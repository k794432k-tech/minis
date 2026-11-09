//
//  GroupChatListView.swift
//  Minis — AI group chat
//
//  Home menu → AI Group Chat. Lists group chats and creates new ones.
//

import SwiftUI

struct GroupChatListView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject private var store = GroupChatStore.shared
    @State private var path: [String] = []
    @State private var showCreate = false

    var body: some View {
        NavigationStack(path: $path) {
            List {
                if store.chats.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "person.3.sequence.fill")
                            .font(.system(size: 40))
                            .foregroundStyle(.secondary)
                        Text("Put several AIs in one conversation")
                            .font(.headline)
                        Text("Pick models from any provider (including on-device models), give them personas, and let them answer you and discuss with each other.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                        Button {
                            showCreate = true
                        } label: {
                            Label("New Group Chat", systemImage: "plus")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 32)
                    .listRowBackground(Color.clear)
                } else {
                    ForEach(store.sortedChats) { chat in
                        NavigationLink(value: chat.id) {
                            GroupChatRow(chat: chat)
                        }
                    }
                    .onDelete { offsets in
                        let chats = store.sortedChats
                        for i in offsets { store.delete(id: chats[i].id) }
                    }
                }
            }
            .navigationTitle("AI Group Chat")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: String.self) { id in
                if let chat = store.chat(id: id) {
                    GroupChatView(chat: chat)
                } else {
                    Text("This group chat was deleted.").foregroundStyle(.secondary)
                }
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showCreate = true
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                }
            }
            .sheet(isPresented: $showCreate) {
                GroupChatSetupView { chat in
                    store.update(chat)
                    // Open the new chat once the sheet has gone.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                        path.append(chat.id)
                    }
                }
            }
        }
    }
}

private struct GroupChatRow: View {
    let chat: GroupChat

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                ForEach(Array(chat.members.prefix(3).enumerated()), id: \.offset) { index, member in
                    GroupAvatar(name: member.name, colorIndex: member.colorIndex, size: 26, entryId: member.entryId)
                        .overlay(Circle().stroke(Color(.systemBackground), lineWidth: 1.5))
                        .offset(x: CGFloat(index) * 10 - 10, y: index % 2 == 0 ? -6 : 6)
                }
            }
            .frame(width: 48, height: 44)
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(chat.title).font(.body.weight(.medium)).lineLimit(1)
                    Spacer()
                    Text(chat.updatedAt, style: .relative)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(chat.lastMessagePreview.isEmpty
                     ? chat.members.map(\.name).joined(separator: ", ")
                     : chat.lastMessagePreview)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 2)
    }
}
