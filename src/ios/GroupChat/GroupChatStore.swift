//
//  GroupChatStore.swift
//  Minis — AI group chat
//
//  A group chat puts several models (from any configured provider, on-device
//  models included) in one conversation with the user. Each chat is stored as
//  one JSON file under Application Support/GroupChats, with attached images
//  next to it.
//
//  Map of the GroupChat files:
//    GroupChatStore.swift      data model + persistence
//    GroupChatSession.swift    turn-taking and streaming replies
//    GroupChatListView.swift   list of group chats (home menu → AI Group Chat)
//    GroupChatSetupView.swift  create / edit members, personas, reply mode
//    GroupChatView.swift       the conversation
//

import Foundation
import SwiftUI
import UIKit

// MARK: - Model

struct GroupChatMember: Codable, Identifiable, Hashable {
    var id: String = UUID().uuidString
    /// `ModelEntry.id` (composite "{instanceId}/{modelId}").
    var entryId: String
    var name: String
    /// Optional role / personality, added to this member's system prompt.
    var persona: String = ""
    var colorIndex: Int = 0
}

struct GroupChatMessage: Codable, Identifiable, Hashable {
    enum Role: String, Codable {
        case user
        case member
    }

    var id: String = UUID().uuidString
    var role: Role
    var memberId: String?
    var speakerName: String
    var text: String
    /// File names in `GroupChatStore.imagesDirectory`.
    var imageFiles: [String] = []
    var createdAt: Date = Date()
    var errorText: String?
    var isStreaming: Bool = false
    /// Tool calls made while producing this reply (nil in older files).
    var toolEvents: [GroupToolEvent]? = nil
}

enum GroupReplyMode: String, Codable, CaseIterable, Identifiable {
    /// Every member answers, in order (only the @mentioned ones when the
    /// message mentions someone).
    case everyone
    /// One member answers: the @mentioned one, otherwise a random pick.
    case oneRandom

    var id: String { rawValue }

    var title: LocalizedStringKey {
        switch self {
        case .everyone: return "Everyone replies"
        case .oneRandom: return "One random member replies"
        }
    }
}

struct GroupChat: Codable, Identifiable, Hashable {
    var id: String = UUID().uuidString
    var title: String
    /// Optional shared topic / ground rules shown to every member.
    var topic: String = ""
    var members: [GroupChatMember]
    var messages: [GroupChatMessage] = []
    var replyMode: GroupReplyMode = .everyone
    /// How many times everyone takes a turn after each user message.
    /// More than one lets the AIs respond to each other.
    var rounds: Int = 1
    var createdAt: Date = Date()
    var updatedAt: Date = Date()
    /// Whether members may call tools (each call needs the user's approval).
    /// Optional so files written before tools existed still decode.
    var toolsEnabled: Bool? = nil
    var enabledTools: [GroupChatToolKind]? = nil

    var toolsOn: Bool { toolsEnabled ?? false }

    /// Tools members can actually call right now.
    var activeTools: [GroupChatToolKind] {
        guard toolsOn else { return [] }
        return enabledTools ?? GroupChatToolKind.allCases
    }

    func member(id: String?) -> GroupChatMember? {
        guard let id else { return nil }
        return members.first { $0.id == id }
    }

    var lastMessagePreview: String {
        guard let last = messages.last(where: { !$0.text.isEmpty || !$0.imageFiles.isEmpty }) else { return "" }
        let body = last.text.isEmpty ? AppLocalized("[Image]") : last.text
        return "\(last.speakerName): \(body)"
    }
}

// MARK: - Palette

enum GroupChatPalette {
    static let colors: [Color] = [.blue, .orange, .green, .purple, .pink, .teal, .indigo, .brown, .red, .mint]

    static func color(_ index: Int) -> Color {
        colors[((index % colors.count) + colors.count) % colors.count]
    }
}

// MARK: - Store

@MainActor
final class GroupChatStore: ObservableObject {
    static let shared = GroupChatStore()

    @Published private(set) var chats: [GroupChat] = []

    static let directory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent("GroupChats", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    static let imagesDirectory: URL = {
        let dir = directory.appendingPathComponent("images", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    private init() {
        chats = Self.loadAll()
    }

    var sortedChats: [GroupChat] {
        chats.sorted { $0.updatedAt > $1.updatedAt }
    }

    func chat(id: String) -> GroupChat? {
        chats.first { $0.id == id }
    }

    /// Updates the in-memory copy; `persist` also writes it to disk.
    func update(_ chat: GroupChat, persist: Bool = true) {
        if let i = chats.firstIndex(where: { $0.id == chat.id }) {
            chats[i] = chat
        } else {
            chats.append(chat)
        }
        if persist { Self.write(chat) }
    }

    func delete(id: String) {
        guard let chat = chat(id: id) else { return }
        for message in chat.messages {
            for file in message.imageFiles {
                try? FileManager.default.removeItem(at: Self.imagesDirectory.appendingPathComponent(file))
            }
        }
        chats.removeAll { $0.id == id }
        try? FileManager.default.removeItem(at: Self.fileURL(id: id))
    }

    // MARK: Images

    /// Stores image bytes and returns the file name to keep on the message.
    static func saveImage(_ data: Data) -> String? {
        let name = "\(UUID().uuidString).jpg"
        do {
            try data.write(to: imagesDirectory.appendingPathComponent(name), options: .atomic)
            return name
        } catch {
            return nil
        }
    }

    static func imageData(_ file: String) -> Data? {
        try? Data(contentsOf: imagesDirectory.appendingPathComponent(file))
    }

    static func image(_ file: String) -> UIImage? {
        imageData(file).flatMap(UIImage.init(data:))
    }

    /// Re-encodes a picked photo as a JPEG no larger than `maxSide` pixels.
    static func normalizedJPEG(_ data: Data, maxSide: CGFloat = 1600) -> Data? {
        guard let image = UIImage(data: data) else { return nil }
        let size = image.size
        let scale = min(1, maxSide / max(size.width, size.height))
        let target = CGSize(width: floor(size.width * scale), height: floor(size.height * scale))
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let rendered = UIGraphicsImageRenderer(size: target, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: target))
        }
        return rendered.jpegData(compressionQuality: 0.85)
    }

    // MARK: Disk

    private static func fileURL(id: String) -> URL {
        directory.appendingPathComponent("\(id).json")
    }

    private static func write(_ chat: GroupChat) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(chat) else { return }
        try? data.write(to: fileURL(id: chat.id), options: .atomic)
    }

    private static func loadAll() -> [GroupChat] {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return files.filter { $0.pathExtension == "json" }.compactMap { url in
            guard let data = try? Data(contentsOf: url),
                  var chat = try? decoder.decode(GroupChat.self, from: data) else { return nil }
            // A reply that was streaming when the app quit is finished as-is.
            for i in chat.messages.indices where chat.messages[i].isStreaming {
                chat.messages[i].isStreaming = false
                if chat.messages[i].text.isEmpty {
                    chat.messages[i].errorText = AppLocalized("Interrupted")
                }
            }
            // Tool calls left waiting or running when the app quit never completed.
            for i in chat.messages.indices {
                guard var events = chat.messages[i].toolEvents else { continue }
                for j in events.indices where events[j].status == .awaitingApproval || events[j].status == .running {
                    events[j].status = .failed
                    events[j].output = AppLocalized("Interrupted")
                }
                chat.messages[i].toolEvents = events
            }
            return chat
        }
    }
}

// MARK: - Model choices

/// Text-capable models from enabled providers, for the member picker.
@MainActor
enum GroupChatModelCatalog {
    struct Section: Identifiable {
        let id: String
        let title: String
        let entries: [ModelEntry]
    }

    static func sections() -> [Section] {
        let store = ProviderConfigStore.shared
        return store.instances
            .filter { $0.isEnabled && !$0.providerType.isUnsupported }
            .compactMap { instance in
                let entries = store.visibleEntries(for: instance.id).filter { isChatModel($0) }
                guard !entries.isEmpty else { return nil }
                return Section(id: instance.id, title: instance.label, entries: entries)
            }
    }

    static func isChatModel(_ entry: ModelEntry) -> Bool {
        let modalities = entry.model.capabilities.supportedModalities
        return modalities.contains(.textOutput) && modalities.contains(.textInput)
    }

    static func entry(_ id: String) -> ModelEntry? {
        ProviderConfigStore.shared.entry(for: id)
    }

    static func displayName(_ entry: ModelEntry) -> String {
        entry.model.displayName
    }

    static func supportsImages(_ entryId: String) -> Bool {
        entry(entryId)?.model.capabilities.supportedModalities.contains(.imageInput) ?? false
    }
}
