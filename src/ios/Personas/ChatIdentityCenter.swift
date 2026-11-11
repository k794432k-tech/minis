//
//  ChatIdentityCenter.swift
//  Minis — the identity shown in the open chat
//
//  The assistant header (avatar + name), the "thinking" label and the chat
//  title normally come from SOUL.md. When the chat's current model has an
//  enabled persona (Settings → Model Personas), that persona's name and
//  avatar replace them for the whole chat — separate from the built-in soul.
//
//  The active chat keeps `current` pointed at the model it is talking to; the
//  bridged list cells and the title read `effective`, which is the persona's
//  identity when there is one and the SOUL.md identity otherwise.
//

import Foundation

@MainActor
enum ChatIdentityCenter {
    /// `ModelEntry.id` the open chat is currently talking to. The chat sets
    /// this as its model resolves; nil falls back to the SOUL.md identity.
    private(set) static var currentEntryId: String?

    static func setCurrentEntry(_ entryId: String?) {
        guard entryId != currentEntryId else { return }
        currentEntryId = entryId
        NotificationCenter.default.post(name: .chatIdentityChanged, object: nil)
    }

    /// The identity to show right now: the current model's persona when it has
    /// one, otherwise the SOUL.md metadata.
    static var effective: SoulMetadata {
        let base = SoulStore.cachedMetadata
        guard let id = currentEntryId,
              let identity = ModelPersonaStore.shared.effectiveIdentity(for: id) else { return base }
        var meta = base
        if let name = identity.name, !name.isEmpty { meta.name = name }
        if let icon = identity.icon, !icon.isEmpty { meta.icon = icon }
        return meta
    }
}
