//
//  GroupChatView.swift
//  Minis — AI group chat
//
//  The conversation screen: message list, @mention chips, photo attachments,
//  send / stop, and the group menu (settings, continue, ask a member, clear).
//

import PhotosUI
import SwiftUI
import UIKit

struct GroupChatView: View {
    @StateObject private var session: GroupChatSession
    @State private var input = ""
    @State private var photoItems: [PhotosPickerItem] = []
    @State private var pendingImages: [Data] = []
    @State private var showSettings = false
    @State private var confirmClear = false
    @State private var showBrowser = false
    @State private var inspectedEvent: GroupToolEvent?
    @FocusState private var inputFocused: Bool

    init(chat: GroupChat) {
        _session = StateObject(wrappedValue: GroupChatSession(chat: chat))
    }

    private var chat: GroupChat { session.chat }

    var body: some View {
        VStack(spacing: 0) {
            messageList
            if let approval = session.pendingApproval {
                GroupToolApprovalCard(
                    approval: approval,
                    onDeny: { session.resolveApproval(false) },
                    onAllow: { session.resolveApproval(true) }
                )
                .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            Divider()
            inputBar
        }
        .animation(.easeOut(duration: 0.2), value: session.pendingApproval?.id)
        .navigationTitle(chat.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) { groupMenu }
        }
        .sheet(isPresented: $showSettings) {
            GroupChatSetupView(chat: chat) { edited in
                session.applySettings(edited)
            }
        }
        .sheet(isPresented: $showBrowser) {
            BrowserSheetView(pool: session.browserPool)
        }
        .sheet(item: $inspectedEvent) { event in
            GroupToolEventDetailView(event: event)
        }
        .confirmationDialog("Clear all messages in this group?", isPresented: $confirmClear, titleVisibility: .visible) {
            Button("Clear Messages", role: .destructive) { session.clearHistory() }
        }
        .onChange(of: photoItems) { items in
            Task { await loadPhotos(items) }
        }
        .onDisappear { session.stop() }
    }

    // MARK: Messages

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    header
                    ForEach(chat.messages) { message in
                        GroupMessageBubble(
                            message: message,
                            member: chat.member(id: message.memberId),
                            onRetry: { session.retry(messageId: message.id) },
                            onDelete: { session.deleteMessage(id: message.id) },
                            onInspect: { inspectedEvent = $0 }
                        )
                        .id(message.id)
                    }
                    Color.clear.frame(height: 4).id("bottom")
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
            }
            .scrollDismissesKeyboard(.immediately)
            // Tapping anywhere in the conversation closes the keyboard — the
            // scroll gesture alone does nothing while the list is too short
            // to scroll.
            .contentShape(Rectangle())
            .simultaneousGesture(TapGesture().onEnded { dismissKeyboard() })
            .onChange(of: chat.messages.count) { _ in
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: chat.messages.last?.text.count ?? 0) { _ in
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            .onAppear { proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }

    private var header: some View {
        VStack(spacing: 8) {
            HStack(spacing: -8) {
                ForEach(chat.members) { member in
                    GroupAvatar(name: member.name, colorIndex: member.colorIndex, size: 34)
                        .overlay(Circle().stroke(Color(.systemBackground), lineWidth: 2))
                }
            }
            Text(chat.members.map(\.name).joined(separator: " · "))
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if !chat.topic.isEmpty {
                Text(chat.topic)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(8)
                    .frame(maxWidth: .infinity)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.bottom, 6)
    }

    // MARK: Input

    private var inputBar: some View {
        VStack(spacing: 8) {
            if !chat.members.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(chat.members) { member in
                            Button {
                                insertMention(member.name)
                            } label: {
                                HStack(spacing: 4) {
                                    Circle().fill(GroupChatPalette.color(member.colorIndex)).frame(width: 8, height: 8)
                                    Text("@\(member.name)").font(.caption)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color(.secondarySystemBackground), in: Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 12)
                }
            }

            if !pendingImages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(pendingImages.enumerated()), id: \.offset) { index, data in
                            if let image = UIImage(data: data) {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 56, height: 56)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                                    .overlay(alignment: .topTrailing) {
                                        Button {
                                            pendingImages.remove(at: index)
                                        } label: {
                                            Image(systemName: "xmark.circle.fill")
                                                .symbolRenderingMode(.palette)
                                                .foregroundStyle(.white, .black.opacity(0.6))
                                        }
                                        .offset(x: 6, y: -6)
                                    }
                            }
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.top, 6)
                }
            }

            HStack(alignment: .bottom, spacing: 8) {
                PhotosPicker(selection: $photoItems, maxSelectionCount: 4, matching: .images) {
                    Image(systemName: "photo.on.rectangle.angled")
                        .font(.system(size: 20))
                        .frame(width: 34, height: 34)
                }
                TextField("Message the group…", text: $input, axis: .vertical)
                    .lineLimit(1...6)
                    .focused($inputFocused)
                    .toolbar {
                        ToolbarItemGroup(placement: .keyboard) {
                            Spacer()
                            Button {
                                dismissKeyboard()
                            } label: {
                                Image(systemName: "keyboard.chevron.compact.down")
                            }
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
                if session.isRunning {
                    Button {
                        session.stop()
                    } label: {
                        Image(systemName: "stop.circle.fill").font(.system(size: 30))
                    }
                    .tint(.red)
                } else {
                    Button {
                        send()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill").font(.system(size: 30))
                    }
                    .disabled(input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && pendingImages.isEmpty)
                }
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
        }
        .padding(.top, 6)
        .background(.bar)
    }

    private var groupMenu: some View {
        Menu {
            Button {
                session.continueDiscussion()
            } label: {
                Label("Continue Discussion", systemImage: "bubble.left.and.bubble.right")
            }
            .disabled(session.isRunning || chat.messages.isEmpty)
            Menu {
                ForEach(chat.members) { member in
                    Button(member.name) { session.askMember(member.id) }
                }
            } label: {
                Label("Ask a Member to Speak", systemImage: "person.wave.2")
            }
            .disabled(session.isRunning)
            if chat.activeTools.contains(.browser) {
                Button {
                    showBrowser = true
                } label: {
                    Label("Show Group Browser", systemImage: "globe")
                }
            }
            Divider()
            Button {
                showSettings = true
            } label: {
                Label("Members & Settings", systemImage: "person.2.badge.gearshape")
            }
            Button(role: .destructive) {
                confirmClear = true
            } label: {
                Label("Clear Messages", systemImage: "trash")
            }
        } label: {
            Image(systemName: "ellipsis.circle")
        }
    }

    // MARK: Actions

    private func dismissKeyboard() {
        inputFocused = false
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }

    private func send() {
        dismissKeyboard()
        let text = input
        let images = pendingImages
        input = ""
        pendingImages = []
        photoItems = []
        session.send(text: text, images: images)
    }

    private func insertMention(_ name: String) {
        let mention = "@\(name) "
        if input.isEmpty || input.hasSuffix(" ") || input.hasSuffix("\n") {
            input += mention
        } else {
            input += " " + mention
        }
        inputFocused = true
    }

    private func loadPhotos(_ items: [PhotosPickerItem]) async {
        guard !items.isEmpty else { return }
        var loaded: [Data] = []
        for item in items {
            if let data = try? await item.loadTransferable(type: Data.self),
               let jpeg = GroupChatStore.normalizedJPEG(data) {
                loaded.append(jpeg)
            }
        }
        pendingImages = Array((pendingImages + loaded).prefix(4))
        photoItems = []
    }
}

// MARK: - Bubble

private struct GroupMessageBubble: View {
    let message: GroupChatMessage
    let member: GroupChatMember?
    let onRetry: () -> Void
    let onDelete: () -> Void
    let onInspect: (GroupToolEvent) -> Void

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 48)
                VStack(alignment: .trailing, spacing: 6) {
                    images
                    if !message.text.isEmpty {
                        Text(message.text)
                            .textSelection(.enabled)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 16))
                            .foregroundStyle(.white)
                    }
                }
            }
            .contextMenu { menuItems }
        } else {
            HStack(alignment: .top, spacing: 8) {
                GroupAvatar(name: message.speakerName, colorIndex: member?.colorIndex ?? 0, size: 30)
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(message.speakerName)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(GroupChatPalette.color(member?.colorIndex ?? 0))
                        if let model = modelName {
                            Text(model).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
                        }
                    }
                    content
                }
                Spacer(minLength: 24)
            }
            .contextMenu { menuItems }
        }
    }

    @ViewBuilder private var content: some View {
        if message.isStreaming && message.text.isEmpty && (message.toolEvents ?? []).isEmpty {
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Thinking…").font(.callout).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
        } else {
            VStack(alignment: .leading, spacing: 6) {
                if let events = message.toolEvents, !events.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(events) { event in
                            Button { onInspect(event) } label: { GroupToolEventRow(event: event) }
                                .buttonStyle(.plain)
                        }
                    }
                }
                if message.isStreaming && message.text.isEmpty {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Working…").font(.caption).foregroundStyle(.secondary)
                    }
                }
                if !message.text.isEmpty {
                    Text(Self.markdown(message.text))
                        .textSelection(.enabled)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
                }
                if let error = message.errorText {
                    HStack(spacing: 8) {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                        Button("Retry", action: onRetry)
                            .font(.caption.weight(.semibold))
                    }
                }
            }
        }
    }

    @ViewBuilder private var images: some View {
        if !message.imageFiles.isEmpty {
            HStack(spacing: 6) {
                ForEach(message.imageFiles, id: \.self) { file in
                    if let image = GroupChatStore.image(file) {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 120, height: 120)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }
            }
        }
    }

    @ViewBuilder private var menuItems: some View {
        if !message.text.isEmpty {
            Button {
                UIPasteboard.general.string = message.text
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
        if message.role == .member && !message.isStreaming {
            Button(action: onRetry) {
                Label("Regenerate", systemImage: "arrow.clockwise")
            }
        }
        if !message.isStreaming {
            Button(role: .destructive, action: onDelete) {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private var modelName: String? {
        guard let member, let entry = GroupChatModelCatalog.entry(member.entryId) else { return nil }
        return GroupChatModelCatalog.displayName(entry)
    }

    private static func markdown(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }
}

// MARK: - Tool UI

/// Shown above the input bar while a member waits for permission to use a tool.
private struct GroupToolApprovalCard: View {
    let approval: GroupToolApproval
    let onDeny: () -> Void
    let onAllow: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                GroupAvatar(name: approval.memberName, colorIndex: approval.colorIndex, size: 26)
                VStack(alignment: .leading, spacing: 1) {
                    Text("\(approval.memberName) wants to use a tool")
                        .font(.subheadline.weight(.semibold))
                    Label(approval.tool.title, systemImage: approval.tool.systemImage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text(approval.title).font(.callout)
            if !approval.detail.isEmpty {
                ScrollView {
                    Text(approval.detail)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 140)
                .padding(8)
                .background(Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
            }
            HStack(spacing: 12) {
                Button(role: .destructive, action: onDeny) {
                    Text("Deny").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                Button(action: onAllow) {
                    Text("Allow Once").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(12)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(GroupChatPalette.color(approval.colorIndex).opacity(0.5), lineWidth: 1))
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}

/// One tool call inside a member's message.
private struct GroupToolEventRow: View {
    let event: GroupToolEvent

    var body: some View {
        HStack(spacing: 6) {
            statusIcon
            Image(systemName: GroupChatToolKind(rawValue: event.toolName)?.systemImage ?? "wrench")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(event.title)
                .font(.caption)
                .lineLimit(1)
            Spacer(minLength: 0)
            Image(systemName: "chevron.right")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(.tertiarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder private var statusIcon: some View {
        switch event.status {
        case .awaitingApproval:
            Image(systemName: "hand.raised.fill").foregroundStyle(.orange).font(.caption)
        case .running:
            ProgressView().controlSize(.mini)
        case .succeeded:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green).font(.caption)
        case .failed:
            Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.red).font(.caption)
        case .denied:
            Image(systemName: "nosign").foregroundStyle(.secondary).font(.caption)
        }
    }
}

/// Full command / output of one tool call.
private struct GroupToolEventDetailView: View {
    @Environment(\.dismiss) private var dismiss
    let event: GroupToolEvent

    var body: some View {
        NavigationStack {
            List {
                Section("Request") {
                    Text(event.detail.isEmpty ? event.title : event.detail)
                        .font(.system(.footnote, design: .monospaced))
                        .textSelection(.enabled)
                }
                if !event.output.isEmpty {
                    Section("Result") {
                        Text(event.output)
                            .font(.system(.footnote, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }
                if let file = event.imageFile, let image = GroupChatStore.image(file) {
                    Section("Screenshot") {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                    }
                }
            }
            .navigationTitle(event.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
