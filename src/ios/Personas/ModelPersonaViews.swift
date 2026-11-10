//
//  ModelPersonaViews.swift
//  Minis — per-model soul, personality and memory
//
//  Settings → Model Personas: pick a model, then edit its name, soul,
//  personality and memories.
//

import PhotosUI
import SwiftUI

// MARK: - List

struct ModelPersonaListView: View {
    @ObservedObject private var store = ModelPersonaStore.shared
    @State private var query = ""
    /// Providers whose models are shown; all start folded.
    @State private var expanded: Set<String> = []

    private var isSearching: Bool { !query.trimmingCharacters(in: .whitespaces).isEmpty }

    private func expandedBinding(_ id: String) -> Binding<Bool> {
        Binding(
            get: { isSearching || expanded.contains(id) },
            set: { open in
                if open { expanded.insert(id) } else { expanded.remove(id) }
            }
        )
    }

    private var sections: [GroupChatModelCatalog.Section] {
        let all = GroupChatModelCatalog.sections()
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return all }
        return all.compactMap { section in
            let entries = section.entries.filter {
                GroupChatModelCatalog.displayName($0).lowercased().contains(q)
                    || $0.model.id.lowercased().contains(q)
                    || (store.persona(for: $0.id)?.name.lowercased().contains(q) ?? false)
            }
            return entries.isEmpty ? nil : GroupChatModelCatalog.Section(id: section.id, title: section.title, entries: entries)
        }
    }

    /// Models that already have a persona, shown first.
    private var configured: [ModelEntry] {
        sections.flatMap(\.entries).filter { store.persona(for: $0.id).map { !$0.isEmpty } ?? false }
    }

    var body: some View {
        List {
            Section {
                Text("Give a model its own name, soul, personality and memory. It applies in regular chats when that model is selected (instead of the global Soul) and to that model in group chats.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            if !configured.isEmpty {
                Section("With a Persona") {
                    ForEach(configured) { entry in
                        NavigationLink {
                            ModelPersonaEditView(entryId: entry.id)
                        } label: {
                            ModelPersonaRow(entry: entry, persona: store.persona(for: entry.id))
                        }
                    }
                }
            }
            // One folding section per provider: tap the provider to show
            // its models.
            ForEach(sections) { section in
                Section(isExpanded: expandedBinding(section.id)) {
                    ForEach(section.entries) { entry in
                        NavigationLink {
                            ModelPersonaEditView(entryId: entry.id)
                        } label: {
                            ModelPersonaRow(entry: entry, persona: store.persona(for: entry.id))
                        }
                    }
                } header: {
                    HStack {
                        Text(section.title)
                        Spacer()
                        let withPersona = section.entries.filter { store.persona(for: $0.id).map { !$0.isEmpty } ?? false }.count
                        Text(withPersona > 0 ? "\(withPersona)/\(section.entries.count)" : "\(section.entries.count)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .searchable(text: $query)
        .navigationTitle("Model Personas")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct ModelPersonaRow: View {
    let entry: ModelEntry
    let persona: ModelPersona?

    var body: some View {
        HStack(spacing: 10) {
            GroupAvatar(name: persona?.effectiveName ?? GroupChatModelCatalog.displayName(entry),
                        colorIndex: entry.id.unicodeScalars.reduce(0) { $0 &+ Int($1.value) } % GroupChatPalette.colors.count,
                        size: 30, entryId: entry.id)
                .opacity(persona.map { $0.enabled && !$0.isEmpty } ?? false ? 1 : 0.35)
            VStack(alignment: .leading, spacing: 2) {
                Text(persona?.effectiveName ?? GroupChatModelCatalog.displayName(entry))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    if persona?.effectiveName != nil {
                        Text(GroupChatModelCatalog.displayName(entry)).lineLimit(1)
                    }
                    if let p = persona, !p.memories.isEmpty {
                        Label("\(p.memories.count)", systemImage: "brain")
                    }
                    if let p = persona, !p.isEmpty, !p.enabled {
                        Text("Off")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Editor

struct ModelPersonaEditView: View {
    let entryId: String
    @ObservedObject private var store = ModelPersonaStore.shared
    @State private var draft: ModelPersona
    @State private var newMemory = ""
    @State private var editingMemory: ModelMemoryEntry?
    @State private var confirmClear = false
    @State private var confirmReset = false
    @State private var avatarItem: PhotosPickerItem?
    @Environment(\.dismiss) private var dismiss

    init(entryId: String) {
        self.entryId = entryId
        _draft = State(initialValue: ModelPersonaStore.shared.persona(for: entryId) ?? ModelPersona(id: entryId))
    }

    private var entry: ModelEntry? { GroupChatModelCatalog.entry(entryId) }

    private var modelName: String {
        entry.map(GroupChatModelCatalog.displayName) ?? AppLocalized("Model unavailable")
    }

    private static let soulPlaceholder = "e.g. You are Lumi, a calm astronomer who grew up in a lighthouse. You care about honesty and curiosity, and you treat the user as an old friend."
    private static let personalityPlaceholder = "e.g. Warm and playful, a little dry humor, short sentences, loves metaphors about the sky. Never uses corporate language."

    var body: some View {
        Form {
            Section {
                HStack(spacing: 16) {
                    GroupAvatar(
                        name: draft.effectiveName ?? modelName,
                        colorIndex: entryId.unicodeScalars.reduce(0) { $0 &+ Int($1.value) } % GroupChatPalette.colors.count,
                        size: 64, entryId: entryId
                    )
                    VStack(alignment: .leading, spacing: 8) {
                        PhotosPicker(selection: $avatarItem, matching: .images) {
                            Label(draft.avatarFile == nil ? "Choose Avatar" : "Change Avatar", systemImage: "photo")
                        }
                        if draft.avatarFile != nil {
                            Button(role: .destructive) {
                                ModelPersonaStore.removeAvatarFile(draft.avatarFile)
                                draft.avatarFile = nil
                                store.save(draft)
                            } label: {
                                Label("Remove Avatar", systemImage: "trash")
                            }
                        }
                    }
                    .buttonStyle(.borderless)
                }
                .padding(.vertical, 4)
            } footer: {
                Text("Shown next to this model's messages in group chats.")
            }
            .onChange(of: avatarItem) { item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let file = ModelPersonaStore.saveAvatarImage(data) {
                        ModelPersonaStore.removeAvatarFile(draft.avatarFile)
                        draft.avatarFile = file
                        store.save(draft)
                    }
                    avatarItem = nil
                }
            }

            Section {
                Toggle("Use This Persona", isOn: $draft.enabled)
                LabeledContent("Model", value: modelName)
                TextField("Name (optional)", text: $draft.name)
            } footer: {
                Text("When on, chats with this model use this persona instead of the global Soul. Group chats use it for every member running this model.")
            }

            Section {
                editor($draft.soul, placeholder: Self.soulPlaceholder)
            } header: {
                Text("Soul")
            } footer: {
                Text("Who this model is: identity, background, values, its relationship with you.")
            }

            Section {
                editor($draft.personality, placeholder: Self.personalityPlaceholder)
            } header: {
                Text("Personality")
            } footer: {
                Text("How it behaves and talks: traits, tone, quirks.")
            }

            Section {
                Toggle("Remember Automatically", isOn: $draft.autoMemory)
                HStack {
                    TextField("Add a memory…", text: $newMemory, axis: .vertical)
                        .lineLimit(1...4)
                    Button {
                        addMemory()
                    } label: {
                        Image(systemName: "plus.circle.fill")
                    }
                    .disabled(newMemory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                ForEach(draft.memories.sorted(by: { $0.createdAt > $1.createdAt })) { memory in
                    Button {
                        editingMemory = memory
                    } label: {
                        ModelMemoryRow(memory: memory)
                    }
                    .foregroundStyle(.primary)
                    .swipeActions {
                        Button(role: .destructive) {
                            draft.memories.removeAll { $0.id == memory.id }
                            persist()
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                }
                if !draft.memories.isEmpty {
                    Button("Clear All Memories", role: .destructive) { confirmClear = true }
                }
            } header: {
                Text("Memory (\(draft.memories.count))")
            } footer: {
                Text("Memories are added to this model's instructions. With automatic memory on, the model saves things worth keeping by itself; you can edit or delete any of them here.")
            }

            if !(store.persona(for: entryId)?.isEmpty ?? true) {
                Section {
                    Button("Delete Persona", role: .destructive) { confirmReset = true }
                }
            }
        }
        .navigationTitle(draft.effectiveName ?? modelName)
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { persist() }
        .onReceive(store.$personas) { personas in
            // Memories the model saved while this screen is open.
            if let stored = personas[entryId], stored.memories != draft.memories {
                draft.memories = stored.memories
            }
        }
        .sheet(item: $editingMemory) { memory in
            ModelMemoryEditSheet(memory: memory) { updated in
                if let i = draft.memories.firstIndex(where: { $0.id == updated.id }) {
                    draft.memories[i] = updated
                    persist()
                }
            }
        }
        .confirmationDialog("Delete all memories of this model?", isPresented: $confirmClear, titleVisibility: .visible) {
            Button("Clear All Memories", role: .destructive) {
                draft.memories.removeAll()
                persist()
            }
        }
        .confirmationDialog("Delete this model's persona and memories?", isPresented: $confirmReset, titleVisibility: .visible) {
            Button("Delete Persona", role: .destructive) {
                store.delete(entryId: entryId)
                draft = ModelPersona(id: entryId)
                dismiss()
            }
        }
    }

    @ViewBuilder
    private func editor(_ text: Binding<String>, placeholder: String) -> some View {
        ZStack(alignment: .topLeading) {
            if text.wrappedValue.isEmpty {
                Text(AppLocalized(String.LocalizationValue(placeholder)))
                    .foregroundStyle(.tertiary)
                    .padding(.top, 8)
                    .padding(.leading, 5)
                    .allowsHitTesting(false)
            }
            TextEditor(text: text)
                .frame(minHeight: 110)
        }
    }

    private func addMemory() {
        let text = newMemory.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        draft.memories.append(ModelMemoryEntry(text: text, source: .user))
        newMemory = ""
        persist()
    }

    private func persist() {
        // Do not create an empty file just by opening a model's page.
        if draft.isEmpty && store.persona(for: entryId) == nil { return }
        store.save(draft)
    }
}

private struct ModelMemoryRow: View {
    let memory: ModelMemoryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(memory.text).font(.callout)
            HStack(spacing: 6) {
                Image(systemName: memory.source == .model ? "sparkles" : "person.fill")
                Text(memory.createdAt, style: .date)
                if let origin = memory.origin, !origin.isEmpty {
                    Text("· \(origin)").lineLimit(1)
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}

private struct ModelMemoryEditSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var memory: ModelMemoryEntry
    let onSave: (ModelMemoryEntry) -> Void

    init(memory: ModelMemoryEntry, onSave: @escaping (ModelMemoryEntry) -> Void) {
        _memory = State(initialValue: memory)
        self.onSave = onSave
    }

    var body: some View {
        NavigationStack {
            Form {
                TextField("Memory", text: $memory.text, axis: .vertical)
                    .lineLimit(2...10)
            }
            .navigationTitle("Edit Memory")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        onSave(memory)
                        dismiss()
                    }
                    .disabled(memory.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
        }
        .presentationDetents([.medium])
    }
}
