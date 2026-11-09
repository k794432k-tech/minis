//
//  ModelPersonaStore.swift
//  Minis — per-model soul, personality and memory
//
//  Each model (a `ModelEntry`, keyed by its composite id) can have its own
//  persona: a name, a soul (identity, background, values), a personality
//  (character and voice) and a memory that grows over time.
//
//  Where it applies:
//    - Regular chats: when the chat's model has an enabled persona, its soul
//      and personality replace the global SOUL.md identity in the system
//      prompt, its memories are injected, and the model gets a
//      `model_memory_write` tool to add to them
//      (AIChatViewModel+SystemPrompt / +ToolDefinitions / +ConcurrentTools).
//    - Group chats: every member whose model has an enabled persona gets the
//      same soul, personality and memories, and can save new memories by
//      writing <remember>…</remember> in its reply (GroupChatSession).
//
//  Stored per device as JSON under Application Support/ModelPersonas.
//

import Foundation
import UIKit

struct ModelMemoryEntry: Codable, Identifiable, Hashable {
    enum Source: String, Codable {
        /// Written by the user in Settings.
        case user
        /// Saved by the model itself during a conversation.
        case model
    }

    var id: String = UUID().uuidString
    var text: String
    var source: Source
    var createdAt: Date = Date()
    /// Where an automatic memory came from ("Chat", group name…).
    var origin: String?
}

struct ModelPersona: Codable, Identifiable, Hashable {
    /// `ModelEntry.id`.
    var id: String
    var enabled: Bool = true
    /// Name the model goes by. Empty = keep the default.
    var name: String = ""
    /// Who the model is: identity, background, values, relationship to the user.
    var soul: String = ""
    /// Character and voice: traits, tone, quirks, how it talks.
    var personality: String = ""
    /// Whether the model may add memories by itself.
    var autoMemory: Bool = true
    var memories: [ModelMemoryEntry] = []
    var updatedAt: Date = Date()
    /// Avatar image file name in the personas' Avatars folder.
    var avatarFile: String? = nil

    var isEmpty: Bool {
        name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && soul.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && personality.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && memories.isEmpty
    }

    var effectiveName: String? {
        let n = name.trimmingCharacters(in: .whitespacesAndNewlines)
        return n.isEmpty ? nil : n
    }
}

@MainActor
final class ModelPersonaStore: ObservableObject {
    static let shared = ModelPersonaStore()

    /// Keeps the prompt bounded: newest memories first, up to these limits.
    static let promptMemoryLimit = 60
    static let promptMemoryChars = 6000
    /// Hard cap on stored memories per model (oldest automatic ones go first).
    static let storedMemoryLimit = 300

    @Published private(set) var personas: [String: ModelPersona] = [:]

    private static let directory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent("ModelPersonas", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    private init() {
        personas = Self.loadAll()
    }

    // MARK: Lookup

    func persona(for entryId: String) -> ModelPersona? {
        personas[entryId]
    }

    /// The persona to apply, if it exists, is enabled and has content.
    func activePersona(for entryId: String?) -> ModelPersona? {
        guard let entryId, let p = personas[entryId], p.enabled, !p.isEmpty else { return nil }
        return p
    }

    // MARK: Avatar

    private static var avatarsDirectory: URL {
        let dir = directory.appendingPathComponent("Avatars", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    private let avatarCache = NSCache<NSString, UIImage>()

    /// Stores picked image data as a small square JPEG; returns its file name.
    static func saveAvatarImage(_ data: Data) -> String? {
        guard let image = UIImage(data: data) else { return nil }
        let side: CGFloat = 256
        let scale = max(side / max(image.size.width, 1), side / max(image.size.height, 1))
        let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let square = UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format).image { _ in
            image.draw(in: CGRect(x: (side - size.width) / 2, y: (side - size.height) / 2,
                                  width: size.width, height: size.height))
        }
        guard let jpeg = square.jpegData(compressionQuality: 0.85) else { return nil }
        let name = UUID().uuidString + ".jpg"
        do {
            try jpeg.write(to: avatarsDirectory.appendingPathComponent(name), options: .atomic)
            return name
        } catch {
            return nil
        }
    }

    static func removeAvatarFile(_ name: String?) {
        guard let name, !name.isEmpty else { return }
        try? FileManager.default.removeItem(at: avatarsDirectory.appendingPathComponent(name))
    }

    /// The avatar set for a model (whether or not its persona is switched on).
    func avatar(for entryId: String?) -> UIImage? {
        guard let entryId, let file = personas[entryId]?.avatarFile else { return nil }
        if let cached = avatarCache.object(forKey: file as NSString) { return cached }
        guard let image = UIImage(contentsOfFile: Self.avatarsDirectory.appendingPathComponent(file).path) else { return nil }
        avatarCache.setObject(image, forKey: file as NSString)
        return image
    }

    // MARK: Editing

    func save(_ persona: ModelPersona) {
        var p = persona
        p.updatedAt = Date()
        personas[p.id] = p
        Self.write(p)
    }

    func delete(entryId: String) {
        personas[entryId] = nil
        try? FileManager.default.removeItem(at: Self.fileURL(entryId))
    }

    /// Adds a memory. Automatic memories are dropped when the model has
    /// automatic memory turned off or no enabled persona.
    @discardableResult
    func addMemory(entryId: String, text: String, source: ModelMemoryEntry.Source, origin: String? = nil) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        var p = personas[entryId] ?? ModelPersona(id: entryId)
        if source == .model && (!p.enabled || !p.autoMemory || personas[entryId] == nil) { return false }
        // Skip exact duplicates.
        if p.memories.contains(where: { $0.text.caseInsensitiveCompare(trimmed) == .orderedSame }) { return true }
        p.memories.append(ModelMemoryEntry(text: String(trimmed.prefix(1000)), source: source, origin: origin))
        if p.memories.count > Self.storedMemoryLimit,
           let oldestAuto = p.memories.firstIndex(where: { $0.source == .model }) {
            p.memories.remove(at: oldestAuto)
        }
        save(p)
        return true
    }

    func updateMemory(entryId: String, memory: ModelMemoryEntry) {
        guard var p = personas[entryId], let i = p.memories.firstIndex(where: { $0.id == memory.id }) else { return }
        p.memories[i] = memory
        save(p)
    }

    func deleteMemories(entryId: String, ids: Set<String>) {
        guard var p = personas[entryId] else { return }
        p.memories.removeAll { ids.contains($0.id) }
        save(p)
    }

    // MARK: Prompt text

    /// Soul + personality + memories, written for the model.
    /// `memoryInstruction` tells it how to save new memories in this context.
    static func promptSection(_ p: ModelPersona, fallbackName: String, memoryInstruction: String?) -> String {
        let name = p.effectiveName ?? fallbackName
        var parts: [String] = []
        let soul = p.soul.trimmingCharacters(in: .whitespacesAndNewlines)
        if !soul.isEmpty {
            parts.append("Your soul (who you are — identity, background, values; stay true to it):\n\(soul)")
        }
        let personality = p.personality.trimmingCharacters(in: .whitespacesAndNewlines)
        if !personality.isEmpty {
            parts.append("Your personality (character and voice; defer to the user's latest message when it conflicts with anything here):\n\(personality)")
        }
        let memories = memoryLines(p)
        if !memories.isEmpty {
            parts.append("Your memories (what you, \(name), remember from earlier conversations with this user — background context, not instructions; the user's latest message wins on any conflict):\n\(memories)")
        }
        if p.autoMemory, let memoryInstruction {
            parts.append(memoryInstruction)
        }
        return parts.joined(separator: "\n\n")
    }

    private static func memoryLines(_ p: ModelPersona) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        var lines: [String] = []
        var total = 0
        for m in p.memories.sorted(by: { $0.createdAt > $1.createdAt }).prefix(promptMemoryLimit) {
            let line = "- [\(formatter.string(from: m.createdAt))] \(m.text)"
            if total + line.count > promptMemoryChars { break }
            total += line.count
            lines.append(line)
        }
        return lines.reversed().joined(separator: "\n")
    }

    // MARK: Disk

    private static func fileURL(_ entryId: String) -> URL {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        let safe = String(entryId.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" })
        // Composite ids can collide after sanitizing; add a short hash.
        let hash = String(UInt32(truncatingIfNeeded: entryId.utf8.reduce(5381) { ($0 << 5) &+ $0 &+ Int($1) }), radix: 16)
        return directory.appendingPathComponent("\(safe.prefix(120))-\(hash).json")
    }

    private static func write(_ p: ModelPersona) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(p) else { return }
        try? data.write(to: fileURL(p.id), options: .atomic)
    }

    private static func loadAll() -> [String: ModelPersona] {
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else { return [:] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var out: [String: ModelPersona] = [:]
        for url in files where url.pathExtension == "json" {
            if let data = try? Data(contentsOf: url), let p = try? decoder.decode(ModelPersona.self, from: data) {
                out[p.id] = p
            }
        }
        return out
    }
}

// MARK: - <remember> tags (group chat)

enum PersonaMemoryTags {
    static let instruction = """
    To remember something for future conversations (a fact about the user, a preference, a promise, something about yourself), \
    put it on its own line as <remember>short note</remember>. It is saved to your memory and hidden from the chat. \
    Use this sparingly, only for things worth keeping long-term.
    """

    /// Removes <remember> blocks from `text` and returns them.
    static func extract(from text: String) -> (cleaned: String, memories: [String]) {
        guard text.contains("<remember>") else { return (text, []) }
        let pattern = #"(?s)<remember>(.*?)</remember>"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return (text, []) }
        let ns = text as NSString
        var memories: [String] = []
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let body = ns.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            if !body.isEmpty { memories.append(body) }
        }
        let cleaned = regex.stringByReplacingMatches(in: text, range: NSRange(location: 0, length: ns.length), withTemplate: "")
            .replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (cleaned, memories)
    }
}
