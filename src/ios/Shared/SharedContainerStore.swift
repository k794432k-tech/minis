import Foundation

/// Reads and writes PendingShare data to the App Group shared container.
/// Compiled into both the main app target and the Share Extension target.
enum SharedContainerStore {
    static let appGroupID = "group.com.openminis.app"

    private static let pendingShareKey = "pendingShare"

    /// The App Group container, or — when the app runs without the App Group
    /// entitlement (sideloaded with a personal / free signing identity, which
    /// rewrites or drops `com.apple.security.application-groups`) — a folder in
    /// the app's own Library. The share extension and Files provider cannot
    /// see the fallback, but the app itself keeps working instead of crashing
    /// on a nil container at launch.
    static var containerURL: URL {
        let fm = FileManager.default
        if let url = fm.containerURL(forSecurityApplicationGroupIdentifier: appGroupID) {
            return url
        }
        let fallback = fm.urls(for: .libraryDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AppGroupFallback", isDirectory: true)
        try? fm.createDirectory(at: fallback, withIntermediateDirectories: true)
        return fallback
    }

    /// Whether the real App Group container is available.
    static var hasAppGroup: Bool {
        FileManager.default.containerURL(forSecurityApplicationGroupIdentifier: appGroupID) != nil
    }

    static var sharedDefaults: UserDefaults? {
        UserDefaults(suiteName: appGroupID)
    }

    /// Directory in the shared container for transferring attachment files.
    static var sharedFileDirectory: URL? {
        FileManager.default
            .containerURL(forSecurityApplicationGroupIdentifier: appGroupID)?
            .appendingPathComponent("ShareExtension", isDirectory: true)
    }

    // MARK: - Write (called by Share Extension)

    static func savePendingShare(_ share: PendingShare) {
        guard let defaults = sharedDefaults else { return }
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(share) {
            defaults.set(data, forKey: pendingShareKey)
            defaults.synchronize()
        }
    }

    // MARK: - Read & Consume (called by main app)

    static func loadPendingShare() -> PendingShare? {
        guard let defaults = sharedDefaults,
              let data = defaults.data(forKey: pendingShareKey) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(PendingShare.self, from: data)
    }

    static func clearPendingShare() {
        sharedDefaults?.removeObject(forKey: pendingShareKey)
        sharedDefaults?.synchronize()
    }

    /// Remove all files from the shared transfer directory.
    static func cleanSharedFiles() {
        guard let dir = sharedFileDirectory else { return }
        let fm = FileManager.default
        if let files = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) {
            for file in files {
                try? fm.removeItem(at: file)
            }
        }
    }
}
