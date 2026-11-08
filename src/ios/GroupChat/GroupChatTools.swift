//
//  GroupChatTools.swift
//  Minis — AI group chat
//
//  Tools group members may call: run shell commands in the Linux sandbox,
//  read / write files, and drive a browser. Every call waits for the user's
//  explicit approval in the chat (see GroupChatSession.requestApproval) —
//  there is no "always allow" in a group.
//
//  Each group chat gets its own sandbox workspace: shell commands run with the
//  group's id as their session, so /var/minis/workspace etc. resolve to a
//  per-group folder, and the group has its own browser tabs.
//

import Foundation

// MARK: - Tool kinds

enum GroupChatToolKind: String, Codable, CaseIterable, Identifiable {
    case shell = "shell_execute"
    case fileRead = "file_read"
    case fileWrite = "file_write"
    case browser = "browser_use"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .shell: return AppLocalized("Run Commands")
        case .fileRead: return AppLocalized("Read Files")
        case .fileWrite: return AppLocalized("Write Files")
        case .browser: return AppLocalized("Browser")
        }
    }

    var systemImage: String {
        switch self {
        case .shell: return "terminal"
        case .fileRead: return "doc.text"
        case .fileWrite: return "square.and.pencil"
        case .browser: return "globe"
        }
    }
}

// MARK: - Recorded tool activity

/// One tool call made during a member's reply, kept on the message so the
/// user can see what happened and later turns know about it.
struct GroupToolEvent: Codable, Identifiable, Hashable {
    enum Status: String, Codable {
        case awaitingApproval
        case running
        case succeeded
        case failed
        case denied
    }

    var id: String
    var toolName: String
    /// Short human summary ("tool_title" from the model, or derived).
    var title: String
    /// The command, path or browser action being requested.
    var detail: String
    var status: Status
    /// Truncated result text, for display and for later turns.
    var output: String = ""
    /// Screenshot taken by a browser action, stored like chat images.
    var imageFile: String?
}

/// A pending request shown to the user as Allow / Deny.
struct GroupToolApproval: Identifiable {
    let id: String
    let memberId: String
    let memberName: String
    let colorIndex: Int
    let tool: GroupChatToolKind
    let title: String
    let detail: String
}

// MARK: - Definitions

enum GroupChatToolCatalog {
    static func definitions(for kinds: [GroupChatToolKind]) -> [AgentToolDefinition] {
        kinds.map(definition)
    }

    private static let titleParam = AgentToolParam(
        type: .string,
        description: "A concise 5-10 word summary of what this call does, shown to the user when asking for permission. Use the user's language."
    )

    static func definition(_ kind: GroupChatToolKind) -> AgentToolDefinition {
        switch kind {
        case .shell:
            return AgentToolDefinition(
                name: kind.rawValue,
                description: "Run a shell command in an Alpine Linux sandbox (/bin/sh -c, stdout and stderr captured). The group's working folder is /var/minis/workspace. The user must approve every call, so explain briefly why you need it. Default timeout 120 s.",
                parameters: [
                    "tool_title": titleParam,
                    "command": AgentToolParam(type: .string, description: "The shell command to run."),
                    "timeout": AgentToolParam(type: .integer, description: "Timeout in seconds (default 120, max 600)."),
                ],
                required: ["tool_title", "command"],
                propertyOrdering: ["tool_title", "command", "timeout"]
            )
        case .fileRead:
            return AgentToolDefinition(
                name: kind.rawValue,
                description: "Read a text file from the Linux sandbox. Returns at most max_length characters.",
                parameters: [
                    "tool_title": titleParam,
                    "path": AgentToolParam(type: .string, description: "Absolute path, e.g. /var/minis/workspace/notes.md"),
                    "max_length": AgentToolParam(type: .integer, description: "Maximum characters to return (default 15000)."),
                ],
                required: ["tool_title", "path"],
                propertyOrdering: ["tool_title", "path", "max_length"]
            )
        case .fileWrite:
            return AgentToolDefinition(
                name: kind.rawValue,
                description: "Write a text file in the Linux sandbox, creating parent folders. Overwrites unless append is true.",
                parameters: [
                    "tool_title": titleParam,
                    "path": AgentToolParam(type: .string, description: "Absolute path, e.g. /var/minis/workspace/plan.md"),
                    "content": AgentToolParam(type: .string, description: "Text to write."),
                    "append": AgentToolParam(type: .boolean, description: "Append instead of overwriting (default false)."),
                ],
                required: ["tool_title", "path", "content"],
                propertyOrdering: ["tool_title", "path", "content", "append"]
            )
        case .browser:
            return AgentToolDefinition(
                name: kind.rawValue,
                description: "Control a web browser shared by this group. Actions: navigate (url), get_readable (page text as readable content), get_text (selector), screenshot (returns an image), click (selector or coordinate_x/coordinate_y), type (selector + text), scroll (direction, amount), find_elements (selector), list_tabs, new_tab, close_tab (tab_id). Prefer get_readable to read pages.",
                parameters: [
                    "tool_title": titleParam,
                    "action": AgentToolParam(type: .string, description: "The browser action", enumValues: BrowserAction.allCases.map(\.rawValue)),
                    "url": AgentToolParam(type: .string, description: "URL for navigate"),
                    "selector": AgentToolParam(type: .string, description: "CSS selector for click, type, get_text, find_elements, scroll"),
                    "text": AgentToolParam(type: .string, description: "Text for type"),
                    "coordinate_x": AgentToolParam(type: .integer, description: "X coordinate for click"),
                    "coordinate_y": AgentToolParam(type: .integer, description: "Y coordinate for click"),
                    "direction": AgentToolParam(type: .string, description: "Scroll direction", enumValues: ["up", "down"]),
                    "amount": AgentToolParam(type: .integer, description: "Scroll amount in pixels"),
                    "tab_id": AgentToolParam(type: .integer, description: "Target tab (optional)"),
                ],
                required: ["tool_title", "action"],
                propertyOrdering: ["tool_title", "action", "tab_id", "url", "selector", "text", "coordinate_x", "coordinate_y", "direction", "amount"]
            )
        }
    }

    /// Title and detail shown in the approval card.
    static func describe(_ kind: GroupChatToolKind, args: [String: Any]) -> (title: String, detail: String) {
        let modelTitle = (args["tool_title"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let detail: String
        switch kind {
        case .shell:
            detail = args["command"] as? String ?? ""
        case .fileRead:
            detail = args["path"] as? String ?? ""
        case .fileWrite:
            let path = args["path"] as? String ?? ""
            let content = args["content"] as? String ?? ""
            let append = (args["append"] as? Bool) == true
            detail = "\(append ? ">>" : ">") \(path)\n\(String(content.prefix(600)))\(content.count > 600 ? "\n…" : "")"
        case .browser:
            let action = args["action"] as? String ?? ""
            let extra = [args["url"], args["selector"], args["text"]].compactMap { $0 as? String }.filter { !$0.isEmpty }
            detail = ([action] + extra).joined(separator: " ")
        }
        return (modelTitle.isEmpty ? kind.title : modelTitle, detail)
    }
}

// MARK: - Execution

struct GroupToolResult {
    var text: String
    var isError: Bool
    var imageData: Data? = nil
    var imageMimeType: String? = nil
}

@MainActor
enum GroupChatToolExecutor {
    static let maxOutputChars = 12_000

    /// The sandbox session a group's commands run in.
    static func sandboxSessionId(chatId: String) -> String { "group-\(chatId)" }

    static func execute(_ kind: GroupChatToolKind, args: [String: Any], chatId: String,
                        browser: BrowserTabPool) async -> GroupToolResult {
        switch kind {
        case .shell:
            let command = args["command"] as? String ?? ""
            guard !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return GroupToolResult(text: "Error: missing 'command'.", isError: true)
            }
            let timeout = min(max(intArg(args["timeout"]) ?? 120, 1), 600)
            return await runShell(command, timeout: TimeInterval(timeout), chatId: chatId)

        case .fileRead:
            guard let path = args["path"] as? String, path.hasPrefix("/") else {
                return GroupToolResult(text: "Error: 'path' must be an absolute path.", isError: true)
            }
            let limit = min(max(intArg(args["max_length"]) ?? 15_000, 100), 50_000)
            let result = await runShell("head -c \(limit * 4) \(shellQuote(path))", timeout: 30, chatId: chatId)
            guard !result.isError else { return result }
            let text = String(result.text.prefix(limit))
            return GroupToolResult(text: "[\(path)]\n" + text + (result.text.count > limit ? "\n…[truncated]" : ""), isError: false)

        case .fileWrite:
            guard let path = args["path"] as? String, path.hasPrefix("/") else {
                return GroupToolResult(text: "Error: 'path' must be an absolute path.", isError: true)
            }
            let content = args["content"] as? String ?? ""
            let append = (args["append"] as? Bool) == true
            let b64 = Data(content.utf8).base64EncodedString()
            let dir = (path as NSString).deletingLastPathComponent
            let command = "mkdir -p \(shellQuote(dir)) && printf %s '\(b64)' | base64 -d \(append ? ">>" : ">") \(shellQuote(path))"
            let result = await runShell(command, timeout: 60, chatId: chatId)
            if result.isError { return result }
            return GroupToolResult(text: "Wrote \(content.utf8.count) bytes to \(path)\(append ? " (appended)" : "").", isError: false)

        case .browser:
            guard JSONSerialization.isValidJSONObject(args),
                  let data = try? JSONSerialization.data(withJSONObject: args),
                  let json = String(data: data, encoding: .utf8),
                  let input = BrowserActionInput.parse(from: json) else {
                return GroupToolResult(text: "Error: invalid browser_use input; 'action' is required.", isError: true)
            }
            do {
                let result = try await browser.execute(action: input, owner: sandboxSessionId(chatId: chatId))
                var out = GroupToolResult(text: truncate(result.text), isError: !result.success)
                if let b64 = result.base64Image, let image = Data(base64Encoded: b64) {
                    out.imageData = image
                    out.imageMimeType = "image/jpeg"
                }
                return out
            } catch {
                return GroupToolResult(text: "Error: \(error.localizedDescription)", isError: true)
            }
        }
    }

    private static func runShell(_ command: String, timeout: TimeInterval, chatId: String) async -> GroupToolResult {
        guard ISHKernel.shared.isBooted else {
            return GroupToolResult(text: "Error: the Linux sandbox is not running yet. Ask the user to open a regular chat once, then retry.", isError: true)
        }
        let sid = sandboxSessionId(chatId: chatId)
        prepareWorkspace(sid: sid)
        do {
            let result = try await ISHExecutionCoordinator.shared.execute(
                sessionId: sid, command: command, timeout: timeout,
                lineCallback: { _ in }, pidCallback: { _ in }
            )
            let body = result.output.isEmpty ? "(no output)" : truncate(result.output)
            return GroupToolResult(text: "exit code \(result.exitCode)\n\(body)", isError: result.exitCode != 0)
        } catch {
            return GroupToolResult(text: "Error: \(error.localizedDescription)", isError: true)
        }
    }

    /// Creates the group's per-session folders on the host side.
    private static func prepareWorkspace(sid: String) {
        for guest in [AIChatViewModel.minisWorkspaceLinuxDir, AIChatViewModel.minisBrowserLinuxDir,
                      AIChatViewModel.minisAttachmentsLinuxDir, AIChatViewModel.minisOffloadsLinuxDir] {
            if let url = MinisFsRouter.shared.hostURL(forGuest: guest, sid: sid) {
                try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            }
        }
    }

    static func truncate(_ text: String) -> String {
        guard text.count > maxOutputChars else { return text }
        let head = text.prefix(maxOutputChars / 2)
        let tail = text.suffix(maxOutputChars / 2)
        return "\(head)\n…[\(text.count - maxOutputChars) characters omitted]…\n\(tail)"
    }

    static func shellQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private static func intArg(_ raw: Any?) -> Int? {
        if let i = raw as? Int { return i }
        if let d = raw as? Double { return Int(d) }
        if let s = raw as? String { return Int(s) }
        return nil
    }
}
