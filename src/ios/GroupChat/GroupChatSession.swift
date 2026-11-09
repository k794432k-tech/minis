//
//  GroupChatSession.swift
//  Minis — AI group chat
//
//  Runs the turn-taking of one group chat. After each user message the
//  responders answer one after another, each seeing the whole transcript:
//  its own earlier replies as assistant turns, the user's and the other
//  members' messages as user turns labelled with the speaker's name. With
//  more than one round (or "Continue discussion"), members also react to
//  each other.
//
//  Members are plain chat completions (`LLMProvider.streamMessage`): no
//  agent tools, which keeps a group of very different models — cloud and
//  on-device — on equal footing.
//

import Foundation
import UIKit

private let logger = AppLogger(category: "GroupChat")

/// Keeps each group's session alive after its screen closes, so members
/// finish their turns in the background instead of stopping.
@MainActor
enum GroupChatSessions {
    private static var sessions: [String: GroupChatSession] = [:]

    static func session(for chat: GroupChat) -> GroupChatSession {
        if let existing = sessions[chat.id] { return existing }
        let session = GroupChatSession(chat: chat)
        sessions[chat.id] = session
        return session
    }

    static func isRunning(_ id: String) -> Bool { sessions[id]?.isRunning ?? false }

    /// The group was deleted: stop it and forget it.
    static func discard(_ id: String) {
        sessions[id]?.stop()
        sessions[id] = nil
    }
}

@MainActor
final class GroupChatSession: ObservableObject {
    @Published private(set) var chat: GroupChat
    @Published private(set) var isRunning = false
    @Published private(set) var activeMemberId: String?
    /// A tool call waiting for the user's Allow / Deny.
    @Published private(set) var pendingApproval: GroupToolApproval?

    /// The group's own browser (separate tabs from regular chats).
    lazy var browserPool = BrowserTabPool()

    private var approvalContinuation: CheckedContinuation<Bool, Never>?
    /// Upper bound on tool rounds in one reply.
    private static let maxToolSteps = 8

    private var runTask: Task<Void, Never>?
    /// Identifies the current run, so a cancelled run that finishes late
    /// does not reset the state of the run that replaced it.
    private var runToken = UUID()
    private var lastPersist = Date.distantPast
    /// Text length at the last repetition check, per streaming message.
    private var lastLoopCheck: [String: Int] = [:]

    /// How many recent messages each member sees.
    private static let historyLimit = 40
    /// How many recent images are sent to vision-capable members.
    private static let imageLimit = 3

    init(chat: GroupChat) {
        self.chat = chat
    }

    // MARK: Public actions

    func send(text: String, images: [Data]) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !images.isEmpty else { return }
        stop()
        let files = images.compactMap { GroupChatStore.saveImage($0) }
        chat.messages.append(GroupChatMessage(
            role: .user, memberId: nil, speakerName: AppLocalized("You"),
            text: trimmed, imageFiles: files
        ))
        persist()
        let first = responders(for: trimmed)
        run(firstRound: first, rounds: max(1, chat.rounds))
    }

    /// Everyone takes one more turn without a new user message.
    func continueDiscussion() {
        guard !chat.members.isEmpty else { return }
        stop()
        run(firstRound: chat.members, rounds: 1)
    }

    /// Asks one member to answer again (replaces its failed or unwanted reply).
    func retry(messageId: String) {
        guard let index = chat.messages.firstIndex(where: { $0.id == messageId }),
              let member = chat.member(id: chat.messages[index].memberId) else { return }
        stop()
        chat.messages.remove(at: index)
        persist()
        run(firstRound: [member], rounds: 1)
    }

    /// Asks a specific member to speak next.
    func askMember(_ memberId: String) {
        guard let member = chat.member(id: memberId) else { return }
        stop()
        run(firstRound: [member], rounds: 1)
    }

    func stop() {
        resolveApproval(false)
        runTask?.cancel()
        runTask = nil
        finishStreamingMessages()
        isRunning = false
        activeMemberId = nil
    }

    func deleteMessage(id: String) {
        guard let message = chat.messages.first(where: { $0.id == id }), !message.isStreaming else { return }
        chat.messages.removeAll { $0.id == id }
        persist()
    }

    func clearHistory() {
        stop()
        chat.messages.removeAll()
        persist()
    }

    /// Applies edits made in the setup screen.
    func applySettings(_ edited: GroupChat) {
        chat.title = edited.title
        chat.topic = edited.topic
        chat.members = edited.members
        chat.replyMode = edited.replyMode
        chat.rounds = edited.rounds
        chat.toolsEnabled = edited.toolsEnabled
        chat.enabledTools = edited.enabledTools
        persist()
    }

    // MARK: Tool approval

    /// Called by the Allow / Deny buttons.
    func resolveApproval(_ allowed: Bool) {
        guard let continuation = approvalContinuation else { return }
        approvalContinuation = nil
        pendingApproval = nil
        continuation.resume(returning: allowed)
    }

    private func requestApproval(_ request: GroupToolApproval) async -> Bool {
        // Only one request can be open; a stale one is denied.
        resolveApproval(false)
        return await withCheckedContinuation { continuation in
            approvalContinuation = continuation
            pendingApproval = request
        }
    }

    /// The text of a message, for copying.
    func copyText(_ message: GroupChatMessage) -> String { message.text }

    // MARK: Turn-taking

    /// Members who answer a user message: the @mentioned ones if any,
    /// otherwise everyone or one random member depending on the mode.
    private func responders(for text: String) -> [GroupChatMember] {
        let lower = text.lowercased()
        let mentioned = chat.members.filter { lower.contains("@" + $0.name.lowercased()) }
        if lower.contains("@all") || lower.contains("@所有人") || lower.contains("@everyone") {
            return chat.members
        }
        if !mentioned.isEmpty { return mentioned }
        switch chat.replyMode {
        case .everyone: return chat.members
        case .oneRandom: return chat.members.randomElement().map { [$0] } ?? []
        }
    }

    private func run(firstRound: [GroupChatMember], rounds: Int) {
        guard !firstRound.isEmpty else { return }
        isRunning = true
        let token = UUID()
        runToken = token
        // Ask iOS for time to finish the turn if the app is sent to the
        // background mid-discussion.
        let background = UIApplication.shared.beginBackgroundTask(withName: "GroupChat-\(chat.id)")
        runTask = Task { [weak self] in
            defer {
                if background != .invalid { UIApplication.shared.endBackgroundTask(background) }
            }
            guard let self else { return }
            for round in 0..<rounds {
                let speakers = round == 0 ? firstRound : self.chat.members
                for member in speakers {
                    if Task.isCancelled { break }
                    // A member that has disappeared from the chat is skipped.
                    guard self.chat.member(id: member.id) != nil else { continue }
                    await self.reply(as: member)
                }
                if Task.isCancelled { break }
            }
            guard self.runToken == token else { return }
            self.isRunning = false
            self.activeMemberId = nil
            self.persist()
        }
    }

    private func reply(as member: GroupChatMember) async {
        activeMemberId = member.id
        let placeholder = GroupChatMessage(
            role: .member, memberId: member.id, speakerName: member.name,
            text: "", isStreaming: true
        )
        let history = chat.messages
        chat.messages.append(placeholder)
        let messageId = placeholder.id

        guard let entry = GroupChatModelCatalog.entry(member.entryId) else {
            fail(messageId, AppLocalized("This member's model is no longer available. Pick another model in Members."))
            return
        }

        if !chat.activeTools.isEmpty {
            await replyWithTools(as: member, entry: entry, history: history, messageId: messageId)
            return
        }

        do {
            let provider = try await LLMProviderFactory.makeProvider(for: entry, sessionId: chat.id)
            let messages = buildMessages(for: member, entry: entry, history: history)
            let system = systemPrompt(for: member, entry: entry)
            let maxTokens = min(entry.model.maxOutputTokens ?? 2048, 2048)
            let stream = try await provider.streamMessage(
                messages: messages, systemPrompt: system, maxTokens: maxTokens, temperature: nil
            )
            for try await chunk in stream {
                if Task.isCancelled { break }
                if case .text(let piece) = chunk {
                    appendText(messageId, piece)
                    if shouldCutOff(messageId, member: member) { break }
                }
            }
            finish(messageId)
        } catch is CancellationError {
            finish(messageId)
        } catch {
            if Task.isCancelled {
                finish(messageId)
            } else {
                logger.error("[Reply] \(member.name) failed: \(error.localizedDescription)")
                fail(messageId, error.localizedDescription)
            }
        }
    }

    // MARK: Tool-using reply

    /// A reply that may call tools: the agent provider streams text and tool
    /// calls; each call is shown to the user for approval, run if allowed, and
    /// its result fed back until the model answers without calling a tool.
    private func replyWithTools(as member: GroupChatMember, entry: ModelEntry,
                                history: [GroupChatMessage], messageId: String) async {
        let kinds = chat.activeTools
        let tools = GroupChatToolCatalog.definitions(for: kinds)
        let canSeeImages = entry.model.capabilities.supportedModalities.contains(.imageInput)
        let system = systemPrompt(for: member, entry: entry)
        var conversation = Self.agentMessages(from: buildMessages(for: member, entry: entry, history: history))
        let provider = await AIChatViewModel.makeAgentProvider(for: entry, sessionId: chat.id)
        let maxTokens = min(entry.model.maxOutputTokens ?? 4096, 4096)
        var steps = 0

        do {
            // Tools stay declared on every request (some providers reject a
            // history containing tool calls otherwise); past the step budget
            // further calls are refused and the model is asked to answer.
            while !Task.isCancelled && steps <= Self.maxToolSteps + 1 {
                let stream = try await provider.streamAgentMessage(
                    messages: conversation, systemPrompt: system,
                    tools: tools, maxTokens: maxTokens
                )
                var text = ""
                var calls: [(id: String, name: String, args: [String: Any])] = []
                var reasoning: String?
                var echo: ReasoningEcho?
                var stoppedEarly = false
                for try await event in stream {
                    if Task.isCancelled || stoppedEarly { break }
                    switch event {
                    case .textDelta(let piece):
                        if text.isEmpty, let i = index(of: messageId), !chat.messages[i].text.isEmpty {
                            appendText(messageId, "\n\n")
                        }
                        text += piece
                        appendText(messageId, piece)
                        if shouldCutOff(messageId, member: member) { stoppedEarly = true }
                    case .thinkingDelta(let piece):
                        appendReasoning(messageId, piece)
                    case .toolCallComplete(let id, let name, let args, _):
                        calls.append((id, name, args))
                    case .reasoningContent(let content):
                        reasoning = content
                        if let i = index(of: messageId), (chat.messages[i].reasoning ?? "").isEmpty {
                            chat.messages[i].reasoning = content
                        }
                    case .reasoningEcho(let value):
                        echo = value
                    default:
                        break
                    }
                }
                if Task.isCancelled { break }

                var parts: [AgentContentPart] = []
                if !text.isEmpty { parts.append(.text(text)) }
                for call in calls { parts.append(.toolUse(id: call.id, name: call.name, input: call.args)) }
                var assistant = AgentMessage(role: .assistant, parts: parts)
                assistant.reasoningContent = reasoning
                assistant.reasoningEcho = echo
                conversation.append(assistant)
                if calls.isEmpty || stoppedEarly { break }
                steps += 1

                var results: [AgentContentPart] = []
                for call in calls {
                    if steps > Self.maxToolSteps {
                        results.append(.toolResult(
                            id: call.id, name: call.name,
                            content: "Tool limit for this reply reached. Do not call more tools; give your answer now.",
                            isError: true))
                        continue
                    }
                    let result = await runTool(call: call, kinds: kinds, member: member,
                                               messageId: messageId, canSeeImages: canSeeImages)
                    results.append(result)
                }
                if Task.isCancelled { break }
                conversation.append(AgentMessage(role: .user, parts: results))
            }
            finish(messageId)
        } catch {
            if Task.isCancelled {
                finish(messageId)
            } else {
                logger.error("[ToolReply] \(member.name) failed: \(error.localizedDescription)")
                fail(messageId, error.localizedDescription)
            }
        }
    }

    /// Asks for approval, runs one tool call and records it on the message.
    private func runTool(call: (id: String, name: String, args: [String: Any]), kinds: [GroupChatToolKind],
                         member: GroupChatMember, messageId: String, canSeeImages: Bool) async -> AgentContentPart {
        guard let kind = GroupChatToolKind(rawValue: call.name), kinds.contains(kind) else {
            return .toolResult(id: call.id, name: call.name,
                               content: "Error: the tool '\(call.name)' is not available in this group.", isError: true)
        }
        let (title, detail) = GroupChatToolCatalog.describe(kind, args: call.args)
        let eventId = UUID().uuidString
        addToolEvent(messageId, GroupToolEvent(id: eventId, toolName: kind.rawValue, title: title,
                                               detail: detail, status: .awaitingApproval))

        let allowed = await requestApproval(GroupToolApproval(
            id: eventId, memberId: member.id, memberName: member.name, colorIndex: member.colorIndex,
            tool: kind, title: title, detail: detail
        ))
        guard allowed, !Task.isCancelled else {
            updateToolEvent(messageId, eventId) {
                $0.status = .denied
                $0.output = AppLocalized("Denied")
            }
            return .toolResult(id: call.id, name: call.name,
                               content: "The user denied this tool call. Do not try it again unless the user asks; continue without it.",
                               isError: true)
        }

        updateToolEvent(messageId, eventId) { $0.status = .running }
        let result = await GroupChatToolExecutor.execute(kind, args: call.args, chatId: chat.id, browser: browserPool)
        let imageFile = result.imageData.flatMap { GroupChatStore.saveImage($0) }
        updateToolEvent(messageId, eventId) {
            $0.status = result.isError ? .failed : .succeeded
            $0.output = result.text.count > 2000 ? String(result.text.prefix(2000)) + "…" : result.text
            $0.imageFile = imageFile
        }
        return .toolResult(id: call.id, name: call.name, content: result.text, isError: result.isError,
                           imageData: canSeeImages ? result.imageData : nil,
                           imageMimeType: canSeeImages ? result.imageMimeType : nil)
    }

    /// Plain chat messages → agent messages (text plus attached images).
    static func agentMessages(from messages: [LLMMessage]) -> [AgentMessage] {
        messages.map { message in
            var parts: [AgentContentPart] = []
            if !message.content.isEmpty { parts.append(.text(message.content)) }
            for image in message.images {
                parts.append(.imageData(data: image.data, mimeType: image.mimeType))
            }
            return AgentMessage(role: message.role == .user ? .user : .assistant, parts: parts)
        }
    }

    private func index(of messageId: String) -> Int? {
        chat.messages.firstIndex { $0.id == messageId }
    }

    private func addToolEvent(_ messageId: String, _ event: GroupToolEvent) {
        guard let i = index(of: messageId) else { return }
        chat.messages[i].toolEvents = (chat.messages[i].toolEvents ?? []) + [event]
        persist()
    }

    private func updateToolEvent(_ messageId: String, _ eventId: String, _ change: (inout GroupToolEvent) -> Void) {
        guard let i = index(of: messageId),
              var events = chat.messages[i].toolEvents,
              let j = events.firstIndex(where: { $0.id == eventId }) else { return }
        change(&events[j])
        chat.messages[i].toolEvents = events
        persist()
    }

    // MARK: Prompt building

    private func systemPrompt(for member: GroupChatMember, entry: ModelEntry) -> String {
        let others = chat.members.filter { $0.id != member.id }.map { other -> String in
            let model = GroupChatModelCatalog.entry(other.entryId).map(GroupChatModelCatalog.displayName) ?? "AI"
            return other.persona.isEmpty ? "\(other.name) (\(model))" : "\(other.name) (\(model)): \(other.persona)"
        }
        var lines: [String] = []
        lines.append("You are \(member.name), one of several AI participants in a group chat with a human user (shown as \"User\").")
        if !others.isEmpty {
            lines.append("The other AI participants are: \(others.joined(separator: "; ")).")
        }
        // [T-model-persona] The model's own soul, personality and memories.
        if let modelPersona = ModelPersonaStore.shared.activePersona(for: member.entryId) {
            let section = ModelPersonaStore.promptSection(
                modelPersona, fallbackName: member.name,
                memoryInstruction: PersonaMemoryTags.instruction
            )
            if !section.isEmpty { lines.append(section) }
        }
        if !member.persona.isEmpty {
            lines.append("Your role in this group: \(member.persona)")
        }
        if !chat.topic.isEmpty {
            lines.append("Group topic and rules: \(chat.topic)")
        }
        lines.append("""
        Messages from other participants are prefixed with their name in square brackets. \
        Write only your own reply as \(member.name): do not prefix it with your name, and never write lines for anyone else. \
        Keep it conversational and concise (usually under 150 words) unless asked for detail. \
        You may agree, disagree, or build on what others said and address them by name. \
        Never repeat or reword what you or anyone else already said — add something new, or reply in one short sentence if you have nothing to add. \
        Do not show your reasoning; write only the final reply. \
        Reply in the language the user is using.
        """)
        let tools = chat.activeTools
        if !tools.isEmpty {
            lines.append("""
            You can call these tools: \(tools.map(\.rawValue).joined(separator: ", ")). \
            The user must approve every single call, so use tools only when they clearly help, \
            give each call a clear tool_title, and say in one sentence why you need it. \
            If a call is denied, do not repeat it; continue without it. \
            Files live in /var/minis/workspace, which is shared with the other members of this group, \
            and the browser is shared too: build on what others already did instead of redoing it.
            """)
        }
        return lines.joined(separator: "\n")
    }

    private func buildMessages(for member: GroupChatMember, entry: ModelEntry, history: [GroupChatMessage]) -> [LLMMessage] {
        let canSeeImages = entry.model.capabilities.supportedModalities.contains(.imageInput)
        let recent = history.filter {
            $0.errorText == nil && (!$0.text.isEmpty || !$0.imageFiles.isEmpty || !($0.toolEvents ?? []).isEmpty)
        }
            .suffix(Self.historyLimit)

        // Which images to actually attach: the most recent few.
        let allImageFiles = recent.flatMap(\.imageFiles)
        let attachable = canSeeImages ? Set(allImageFiles.suffix(Self.imageLimit)) : []

        var out: [LLMMessage] = []
        func appendUser(_ text: String, images: [LLMMessage.ImageAttachment]) {
            if let last = out.last, last.role == .user {
                var merged = LLMMessage(role: .user, content: last.content + "\n\n" + text)
                merged.images = last.images + images
                out[out.count - 1] = merged
            } else {
                var message = LLMMessage(role: .user, content: text)
                message.images = images
                out.append(message)
            }
        }

        for message in recent {
            let body = Self.stripThinking(message.text)
            switch message.role {
            case .user:
                var attachments: [LLMMessage.ImageAttachment] = []
                var notes: [String] = []
                for file in message.imageFiles {
                    if attachable.contains(file), let data = GroupChatStore.imageData(file) {
                        attachments.append(.init(mimeType: "image/jpeg", data: data))
                    } else {
                        notes.append(canSeeImages ? "[User shared an earlier image]" : "[User shared an image you cannot see]")
                    }
                }
                let text = ([body.isEmpty ? nil : "[User]: \(body)"] + notes.map { Optional($0) })
                    .compactMap { $0 }.joined(separator: "\n")
                appendUser(text.isEmpty ? "[User]:" : text, images: attachments)
            case .member:
                let summary = Self.toolSummary(message)
                let full = summary.isEmpty ? body : (body.isEmpty ? summary : body + "\n" + summary)
                if message.memberId == member.id {
                    out.append(LLMMessage(role: .assistant, content: full))
                } else {
                    appendUser("[\(message.speakerName)]: \(full)", images: [])
                }
            }
        }

        // Most providers want the conversation to start and end with the user.
        if out.first?.role == .assistant {
            out.insert(LLMMessage(role: .user, content: "(The conversation so far.)"), at: 0)
        }
        if out.isEmpty || out.last?.role == .assistant {
            appendUser("(It is your turn, \(member.name). Continue the discussion.)", images: [])
        }
        return out
    }

    /// A compact record of the tools a reply used, so later turns know what
    /// was done (commands, files, pages) and what came out.
    static func toolSummary(_ message: GroupChatMessage) -> String {
        guard let events = message.toolEvents, !events.isEmpty else { return "" }
        let lines = events.map { e -> String in
            let detail = e.detail.count > 200 ? String(e.detail.prefix(200)) + "…" : e.detail
            let output = e.output.count > 300 ? String(e.output.prefix(300)) + "…" : e.output
            return "- \(e.toolName) [\(e.status.rawValue)] \(detail)" + (output.isEmpty ? "" : " → \(output)")
        }
        return "(Tool calls in this reply:\n" + lines.joined(separator: "\n") + ")"
    }

    static func stripThinking(_ text: String) -> String {
        GroupThinking.split(text).visible
    }

    // MARK: Message updates

    private func appendText(_ id: String, _ piece: String) {
        guard let i = chat.messages.firstIndex(where: { $0.id == id }) else { return }
        chat.messages[i].text += piece
        // Keep the store's copy roughly current without writing on every token.
        if Date().timeIntervalSince(lastPersist) > 2 {
            persist()
        }
    }

    private func appendReasoning(_ id: String, _ piece: String) {
        guard let i = chat.messages.firstIndex(where: { $0.id == id }) else { return }
        chat.messages[i].reasoning = (chat.messages[i].reasoning ?? "") + piece
    }

    /// Moves thinking out of the reply text into `reasoning`, and cleans the
    /// reply: drops a "[Name]:" prefix, cuts lines written for other members
    /// and collapses a reply that got stuck repeating itself.
    private func settleText(at i: Int) {
        let parts = GroupThinking.split(chat.messages[i].text)
        if !parts.thinking.isEmpty {
            let existing = chat.messages[i].reasoning ?? ""
            chat.messages[i].reasoning = existing.isEmpty ? parts.thinking : existing + "\n\n" + parts.thinking
        }
        var text = parts.visible
        if let member = chat.member(id: chat.messages[i].memberId) {
            text = GroupReplyCleaner.clean(text, speaker: member.name,
                                           others: chat.members.filter { $0.id != member.id }.map(\.name))
        }
        text = GroupReplyCleaner.collapseRepetition(text)
        chat.messages[i].text = text
    }

    /// True when the streaming reply has started looping or is writing turns
    /// for other members — the stream is then stopped early.
    private func shouldCutOff(_ id: String, member: GroupChatMember) -> Bool {
        guard let i = index(of: id) else { return false }
        let text = chat.messages[i].text
        let count = text.count
        guard count > 120, count - (lastLoopCheck[id] ?? 0) >= 40 else { return false }
        lastLoopCheck[id] = count
        let visible = GroupThinking.split(text).visible
        if GroupReplyCleaner.isLooping(visible) {
            logger.info("[Reply] \(member.name) is repeating itself — stopping")
            return true
        }
        let others = chat.members.filter { $0.id != member.id }.map(\.name)
        if GroupReplyCleaner.speaksForOthers(visible, others: others) {
            logger.info("[Reply] \(member.name) started writing for others — stopping")
            return true
        }
        return false
    }

    private func finish(_ id: String) {
        guard let i = chat.messages.firstIndex(where: { $0.id == id }) else { return }
        chat.messages[i].isStreaming = false
        settleText(at: i)
        saveRememberTags(at: i)
        // A reply that only repeats what this member just said adds nothing.
        if let memberId = chat.messages[i].memberId, !chat.messages[i].text.isEmpty,
           (chat.messages[i].toolEvents ?? []).isEmpty {
            let mine = GroupReplyCleaner.normalized(chat.messages[i].text)
            let earlier = chat.messages[..<i].suffix(8).filter { $0.memberId == memberId }
            if earlier.contains(where: { GroupReplyCleaner.normalized($0.text) == mine }) {
                logger.info("[Reply] dropped a duplicate reply")
                chat.messages.remove(at: i)
                persist()
                return
            }
        }
        if chat.messages[i].text.isEmpty && chat.messages[i].errorText == nil
            && (chat.messages[i].toolEvents ?? []).isEmpty {
            chat.messages[i].errorText = AppLocalized("No reply")
        }
        persist()
    }

    /// [T-model-persona] Moves <remember> notes out of a finished reply into
    /// the speaking model's persona memory.
    private func saveRememberTags(at i: Int) {
        let (cleaned, memories) = PersonaMemoryTags.extract(from: chat.messages[i].text)
        guard !memories.isEmpty else { return }
        chat.messages[i].text = cleaned
        guard let member = chat.member(id: chat.messages[i].memberId) else { return }
        for memory in memories {
            ModelPersonaStore.shared.addMemory(entryId: member.entryId, text: memory, source: .model,
                                               origin: chat.title)
        }
    }

    private func fail(_ id: String, _ message: String) {
        guard let i = chat.messages.firstIndex(where: { $0.id == id }) else { return }
        chat.messages[i].isStreaming = false
        chat.messages[i].errorText = message
        persist()
    }

    private func finishStreamingMessages() {
        for i in chat.messages.indices where chat.messages[i].isStreaming {
            chat.messages[i].isStreaming = false
            settleText(at: i)
            saveRememberTags(at: i)
            if chat.messages[i].text.isEmpty && (chat.messages[i].toolEvents ?? []).isEmpty {
                chat.messages[i].errorText = AppLocalized("Stopped")
            }
            // Tool calls that never got to run are marked as stopped.
            if var events = chat.messages[i].toolEvents {
                for j in events.indices where events[j].status == .awaitingApproval || events[j].status == .running {
                    events[j].status = .denied
                    events[j].output = AppLocalized("Stopped")
                }
                chat.messages[i].toolEvents = events
            }
        }
        persist()
    }

    private func persist() {
        chat.updatedAt = Date()
        lastPersist = Date()
        GroupChatStore.shared.update(chat)
    }
}
