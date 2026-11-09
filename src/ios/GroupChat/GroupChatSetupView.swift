//
//  GroupChatSetupView.swift
//  Minis — AI group chat
//
//  Create or edit a group chat: title, shared topic, members (model, name,
//  persona) and how members take turns.
//

import SwiftUI

struct GroupChatSetupView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var draft: GroupChat
    @State private var showModelPicker = false
    @State private var editingMember: GroupChatMember?
    private let isNew: Bool
    private let onSave: (GroupChat) -> Void

    init(chat: GroupChat? = nil, onSave: @escaping (GroupChat) -> Void) {
        _draft = State(initialValue: chat ?? GroupChat(title: "", members: []))
        isNew = chat == nil
        self.onSave = onSave
    }

    private var canSave: Bool { !draft.members.isEmpty }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Group name", text: $draft.title)
                    TextField("Topic or ground rules (optional)", text: $draft.topic, axis: .vertical)
                        .lineLimit(1...4)
                } footer: {
                    Text("The topic is shared with every member, e.g. \"Debate whether remote work is better\" or \"Plan a 3-day trip to Kyoto together\".")
                }

                Section {
                    ForEach(draft.members) { member in
                        Button {
                            editingMember = member
                        } label: {
                            GroupMemberRow(member: member)
                        }
                        .foregroundStyle(.primary)
                    }
                    .onDelete { draft.members.remove(atOffsets: $0) }
                    .onMove { draft.members.move(fromOffsets: $0, toOffset: $1) }
                    Button {
                        showModelPicker = true
                    } label: {
                        Label("Add Member", systemImage: "plus.circle.fill")
                    }
                } header: {
                    Text("Members")
                } footer: {
                    Text("Add two or more models, from any provider. The same model can join several times with different personas. Members speak in this order.")
                }

                Section {
                    Picker("Who replies", selection: $draft.replyMode) {
                        ForEach(GroupReplyMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    Stepper(value: $draft.rounds, in: 1...3) {
                        HStack {
                            Text("Rounds per message")
                            Spacer()
                            Text("\(draft.rounds)").foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Turn-taking")
                } footer: {
                    Text("Mention a member with @Name to ask only them, or @all for everyone. With more than one round, members respond to each other after answering you.")
                }

                Section {
                    Toggle("Allow Tools", isOn: Binding(
                        get: { draft.toolsOn },
                        set: { draft.toolsEnabled = $0 }
                    ))
                    if draft.toolsOn {
                        ForEach(GroupChatToolKind.allCases) { kind in
                            Toggle(isOn: Binding(
                                get: { draft.activeTools.contains(kind) },
                                set: { on in
                                    var set = draft.activeTools
                                    if on { if !set.contains(kind) { set.append(kind) } } else { set.removeAll { $0 == kind } }
                                    draft.enabledTools = GroupChatToolKind.allCases.filter { set.contains($0) }
                                }
                            )) {
                                Label(kind.title, systemImage: kind.systemImage)
                            }
                        }
                    }
                } header: {
                    Text("Tools")
                } footer: {
                    Text("Members can run commands in the Linux sandbox, read and write files, and use a browser. Every tool call waits for you to tap Allow in the chat. The group has its own workspace folder and browser tabs.")
                }
            }
            .navigationTitle(isNew ? "New Group Chat" : "Group Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(isNew ? "Create" : "Save") {
                        var result = draft
                        if result.title.trimmingCharacters(in: .whitespaces).isEmpty {
                            result.title = result.members.map(\.name).joined(separator: ", ")
                        }
                        onSave(result)
                        dismiss()
                    }
                    .disabled(!canSave)
                }
            }
            .sheet(isPresented: $showModelPicker) {
                GroupModelPickerView { entry in
                    addMember(entry)
                }
            }
            .sheet(item: $editingMember) { member in
                GroupMemberEditView(member: member) { updated in
                    if let i = draft.members.firstIndex(where: { $0.id == updated.id }) {
                        draft.members[i] = updated
                    }
                }
            }
        }
    }

    private func addMember(_ entry: ModelEntry) {
        let base = GroupChatModelCatalog.displayName(entry)
        var name = base
        var n = 2
        while draft.members.contains(where: { $0.name == name }) {
            name = "\(base) \(n)"
            n += 1
        }
        let usedColors = Set(draft.members.map(\.colorIndex))
        let color = (0..<GroupChatPalette.colors.count).first { !usedColors.contains($0) } ?? draft.members.count
        draft.members.append(GroupChatMember(entryId: entry.id, name: name, colorIndex: color))
    }
}

// MARK: - Member row

struct GroupMemberRow: View {
    let member: GroupChatMember

    var body: some View {
        HStack(spacing: 10) {
            GroupAvatar(name: member.name, colorIndex: member.colorIndex, size: 32, entryId: member.entryId)
            VStack(alignment: .leading, spacing: 2) {
                Text(member.name).font(.body)
                Text(modelName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if !member.persona.isEmpty {
                    Text(member.persona)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
    }

    private var modelName: String {
        guard let entry = GroupChatModelCatalog.entry(member.entryId) else {
            return AppLocalized("Model unavailable")
        }
        let provider = ProviderConfigStore.shared.instance(for: entry.providerInstanceId)?.label ?? ""
        let name = GroupChatModelCatalog.displayName(entry)
        return provider.isEmpty ? name : "\(name) · \(provider)"
    }
}

struct GroupAvatar: View {
    let name: String
    let colorIndex: Int
    var size: CGFloat = 30
    /// The member's model: shows the avatar set in its persona, if any.
    var entryId: String? = nil
    @ObservedObject private var personas = ModelPersonaStore.shared

    var body: some View {
        if let image = personas.avatar(for: entryId) {
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: size, height: size)
                .clipShape(Circle())
        } else {
            Circle()
                .fill(GroupChatPalette.color(colorIndex).gradient)
                .frame(width: size, height: size)
                .overlay(
                    Text(String(name.trimmingCharacters(in: .whitespaces).prefix(1)).uppercased())
                        .font(.system(size: size * 0.45, weight: .semibold))
                        .foregroundStyle(.white)
                )
        }
    }
}

// MARK: - Member editor

struct GroupMemberEditView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var member: GroupChatMember
    @State private var showModelPicker = false
    let onSave: (GroupChatMember) -> Void

    private static let personaExamples = [
        "A skeptical critic who points out weaknesses",
        "An optimistic idea generator",
        "A concise summarizer who wraps up the discussion",
        "A domain expert who gives precise facts",
    ]

    init(member: GroupChatMember, onSave: @escaping (GroupChatMember) -> Void) {
        _member = State(initialValue: member)
        self.onSave = onSave
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Name") {
                    TextField("Name", text: $member.name)
                }
                Section {
                    Button {
                        showModelPicker = true
                    } label: {
                        HStack {
                            Text("Model")
                            Spacer()
                            Text(GroupChatModelCatalog.entry(member.entryId).map(GroupChatModelCatalog.displayName) ?? AppLocalized("Model unavailable"))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .foregroundStyle(.primary)
                }
                Section {
                    NavigationLink {
                        ModelPersonaEditView(entryId: member.entryId)
                    } label: {
                        Label("Model Soul, Personality & Memory", systemImage: "theatermasks")
                    }
                } footer: {
                    Text("Shared by this model everywhere: in regular chats and in every group.")
                }
                Section {
                    TextField("Persona (optional)", text: $member.persona, axis: .vertical)
                        .lineLimit(2...6)
                    Menu("Examples") {
                        ForEach(Self.personaExamples, id: \.self) { example in
                            Button(AppLocalized(String.LocalizationValue(example))) {
                                member.persona = AppLocalized(String.LocalizationValue(example))
                            }
                        }
                    }
                } header: {
                    Text("Persona")
                } footer: {
                    Text("Describe this member's role or personality. It is added to this member's instructions only.")
                }
                Section("Color") {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            ForEach(GroupChatPalette.colors.indices, id: \.self) { i in
                                Circle()
                                    .fill(GroupChatPalette.color(i))
                                    .frame(width: 28, height: 28)
                                    .overlay(Circle().stroke(Color.primary, lineWidth: member.colorIndex == i ? 2 : 0))
                                    .onTapGesture { member.colorIndex = i }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle(member.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        member.name = member.name.trimmingCharacters(in: .whitespacesAndNewlines)
                        onSave(member)
                        dismiss()
                    }
                    .disabled(member.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .sheet(isPresented: $showModelPicker) {
                GroupModelPickerView { entry in
                    member.entryId = entry.id
                }
            }
        }
    }
}

// MARK: - Model picker

struct GroupModelPickerView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    let onPick: (ModelEntry) -> Void

    private var sections: [GroupChatModelCatalog.Section] {
        let all = GroupChatModelCatalog.sections()
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return all }
        return all.compactMap { section in
            let entries = section.entries.filter {
                GroupChatModelCatalog.displayName($0).lowercased().contains(q)
                    || $0.model.id.lowercased().contains(q)
                    || section.title.lowercased().contains(q)
            }
            return entries.isEmpty ? nil : GroupChatModelCatalog.Section(id: section.id, title: section.title, entries: entries)
        }
    }

    var body: some View {
        NavigationStack {
            List {
                if GroupChatModelCatalog.sections().isEmpty {
                    Text("No models available. Add a provider in Settings → Manage Providers, or download a model in Settings → Local Models.")
                        .foregroundStyle(.secondary)
                }
                ForEach(sections) { section in
                    Section(section.title) {
                        ForEach(section.entries) { entry in
                            Button {
                                onPick(entry)
                                dismiss()
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(GroupChatModelCatalog.displayName(entry))
                                        Text(entry.model.id)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    if entry.model.capabilities.supportedModalities.contains(.imageInput) {
                                        Image(systemName: "photo").foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .foregroundStyle(.primary)
                        }
                    }
                }
            }
            .searchable(text: $query)
            .navigationTitle("Choose a Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
