//
//  GroupChatHomeRow.swift
//  Minis — AI group chat
//
//  The entry row at the top of the home session list.
//

import SwiftUI

struct GroupChatHomeRow: View {
    @ObservedObject private var store = GroupChatStore.shared
    let onOpen: () -> Void

    private var latest: GroupChat? { store.sortedChats.first }

    var body: some View {
        Button(action: onOpen) {
            HStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Color.accentColor.gradient)
                        .frame(width: 36, height: 36)
                    Image(systemName: "person.3.fill")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.white)
                }
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("AI Group Chat")
                            .font(.body.weight(.semibold))
                        if !store.chats.isEmpty {
                            Text("\(store.chats.count)")
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(Color.secondary.opacity(0.15), in: Capsule())
                        }
                    }
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var subtitle: String {
        if let latest {
            let preview = latest.lastMessagePreview
            return preview.isEmpty ? latest.title : "\(latest.title) · \(preview)"
        }
        return AppLocalized("Put several AIs in one conversation")
    }
}
