//
//  LocalModelProviderRegistrar.swift
//  Minis — on-device models
//
//  Makes the loopback server show up as an ordinary provider: one OpenAI
//  (Chat Completions) API-key instance whose base URL is the local server.
//  Its model list is the server's /v1/models, i.e. the downloaded GGUF files,
//  refreshed whenever a download finishes or a model is deleted.
//
//  The instance is found by its base URL rather than a stored id, so an
//  instance synced from another device is reused instead of duplicated. The
//  server accepts the API key stored for that instance (iCloud Keychain keeps
//  it identical across devices).
//

import Combine
import Foundation
import Security

private let logger = AppLogger(category: "LocalModels")

@MainActor
final class LocalModelProviderRegistrar {
    static let shared = LocalModelProviderRegistrar()
    static let instanceLabel = "On-device (llama.cpp)"

    private var configObserver: AnyCancellable?

    private init() {
        // Keep the server's view of "which instances are local" current.
        configObserver = ProviderConfigStore.shared.$config.sink { config in
            let ids = config.instances.filter { Self.isLocalBaseURL($0.customBaseURL) }.map(\.id)
            LocalProviderInstanceIds.update(ids)
        }
    }

    /// The provider instance that points at the local server, if any.
    var instance: ProviderInstance? {
        ProviderConfigStore.shared.instances.first { Self.isLocalBaseURL($0.customBaseURL) }
    }

    /// The provider entry id of the on-device model currently loaded in
    /// memory, so a newly opened chat can default to "the model I just
    /// loaded". nil when nothing is loaded or it isn't registered.
    func loadedModelEntryId() -> String? {
        guard let path = LlamaEngine.shared.loadedPath,
              let instance,
              let file = LocalModelCatalog.shared.models.first(where: { $0.primaryURL.path == path }) else { return nil }
        let store = ProviderConfigStore.shared
        return store.modelEntries.first {
            $0.providerInstanceId == instance.id && $0.model.id == file.id && !$0.isHidden
        }?.id
    }

    nonisolated static func isLocalBaseURL(_ url: String?) -> Bool {
        guard var s = url?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else { return false }
        while s.hasSuffix("/") { s.removeLast() }
        if s.hasSuffix("/v1") { s.removeLast(3) }
        return s == LocalLLMServer.baseURL.lowercased()
            || s == "http://localhost:\(LocalLLMServer.port)"
    }

    /// Called at launch: re-create the download session and start the server
    /// when there is anything to serve.
    func activateAtLaunch() {
        LocalModelDownloadManager.shared.activate()
        if !LocalModelCatalog.shared.models.isEmpty || instance != nil {
            LocalLLMServer.shared.start()
        }
    }

    /// Called after a download finishes or a model is deleted.
    func modelsChanged() {
        if LocalModelCatalog.shared.models.isEmpty {
            // Keep the instance (the user may have put it in groups); its
            // list simply becomes empty.
            if let instance { refresh(instance) }
            return
        }
        let inst = ensureRegistered()
        refresh(inst)
    }

    /// Creates the provider instance if it does not exist yet.
    @discardableResult
    func ensureRegistered() -> ProviderInstance {
        LocalLLMServer.shared.start()
        if let existing = instance {
            if ProviderKeychainHelper.loadAPIKey(instanceId: existing.id)?.isEmpty ?? true {
                ProviderKeychainHelper.saveAPIKey(Self.makeToken(), instanceId: existing.id)
            }
            if !existing.isEnabled {
                var enabled = existing
                enabled.isEnabled = true
                ProviderConfigStore.shared.updateInstance(enabled)
            }
            return existing
        }
        let inst = ProviderInstance(
            label: Self.instanceLabel,
            providerType: .openAI,
            credentialType: .apiKey,
            customBaseURL: LocalLLMServer.baseURL,
            appendV1Suffix: true
        )
        // Key first: addInstance immediately fetches the model list with it.
        ProviderKeychainHelper.saveAPIKey(Self.makeToken(), instanceId: inst.id)
        // Prompt evaluation of a long agent prompt can take a while on a
        // phone before the first token arrives; the default 120 s idle
        // timeout would cut it off.
        ResponseIdleTimeoutSetting.set(1800, forInstanceId: inst.id)
        ProviderConfigStore.shared.addInstance(inst)
        logger.info("[Registrar] added provider instance \(inst.id.prefix(8)) for the local server")
        return inst
    }

    private func refresh(_ instance: ProviderInstance) {
        Task { @MainActor in
            // Give a just-started listener a moment to come up.
            for _ in 0..<10 where !LocalLLMServer.shared.isRunning {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            await ProviderConfigStore.shared.refreshModels(for: instance, caller: "LocalModels.modelsChanged")
        }
    }

    /// The on-device server's API key (for showing other apps how to connect).
    var serverAPIKey: String? {
        guard let instance else { return nil }
        let key = ProviderKeychainHelper.loadAPIKey(instanceId: instance.id)
        return (key?.isEmpty == false) ? key : nil
    }

    /// Checks the bearer token of a request against the instance's API key.
    nonisolated static func isAuthorized(_ header: String?) -> Bool {
        guard let header, header.lowercased().hasPrefix("bearer ") else { return false }
        let token = String(header.dropFirst("bearer ".count)).trimmingCharacters(in: .whitespaces)
        guard !token.isEmpty else { return false }
        return acceptedTokens().contains(token)
    }

    /// API keys of every instance pointing at the local server. Read from the
    /// persisted config so this works off the main actor.
    nonisolated private static func acceptedTokens() -> Set<String> {
        let ids = LocalProviderInstanceIds.current
        return Set(ids.compactMap { ProviderKeychainHelper.loadAPIKey(instanceId: $0) }.filter { !$0.isEmpty })
    }

    nonisolated private static func makeToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 24)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return "local-" + bytes.map { String(format: "%02x", $0) }.joined()
    }
}

/// Thread-safe snapshot of the local instance ids, kept current from the main
/// actor so the server queue never has to hop to it.
enum LocalProviderInstanceIds {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var ids: [String] = []

    static var current: [String] {
        lock.lock(); defer { lock.unlock() }
        return ids
    }

    static func update(_ newIds: [String]) {
        lock.lock(); ids = newIds; lock.unlock()
    }
}
