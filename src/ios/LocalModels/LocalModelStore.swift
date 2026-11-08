//
//  LocalModelStore.swift
//  Minis — on-device models
//
//  Shared, thread-safe description of the GGUF models that live on this
//  device, plus the user-tunable inference settings. Everything here is
//  nonisolated on purpose: the local HTTP server and the llama.cpp engine read
//  it from their own queues, while the SwiftUI layer (MainActor) writes it.
//
//  Map of the LocalModels files:
//    LocalModelStore.swift            catalog on disk + settings + paths
//    LocalModelHub.swift              Hugging Face / ModelScope search + file APIs
//    LocalModelDownloadManager.swift  background URLSession downloads
//    LlamaEngine.swift                llama.cpp wrapper (load, prefix cache, sampling)
//    LocalChatFormatter.swift         OpenAI messages <-> prompt, tool-call parsing
//    LocalLLMServer.swift             loopback OpenAI-compatible HTTP server
//    LocalModelProviderRegistrar.swift  registers the server as a provider instance
//    LocalModelsView.swift / LocalModelBrowserView.swift  UI
//

import Foundation

// MARK: - Source hub

enum LocalModelSource: String, Codable, CaseIterable, Identifiable, Hashable, Sendable {
    case huggingFace
    case modelScope

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .huggingFace: return "Hugging Face"
        case .modelScope: return "ModelScope"
        }
    }
}

// MARK: - A model on disk

/// One runnable model. Split GGUF models (`-00001-of-00003.gguf`) are stored
/// as a folder holding every part; `primaryRelativePath` points at part 1,
/// which is what llama.cpp is given (it finds the other parts itself).
struct LocalModelFile: Codable, Identifiable, Hashable, Sendable {
    /// The model id exposed through `/v1/models` and sent back in requests.
    var id: String
    var displayName: String
    var source: LocalModelSource
    var repoId: String
    /// Repo-relative paths of every file that makes up the model.
    var repoPaths: [String]
    /// Path of the first (or only) GGUF file, relative to `LocalModelPaths.modelsDirectory`.
    var primaryRelativePath: String
    /// Folder (relative to the models directory) that holds the files.
    var folderRelativePath: String
    var sizeBytes: Int64
    var downloadedAt: Date
    /// Vision projector (`mmproj-*.gguf`), relative to the models directory.
    /// Present when the model can look at images.
    var mmprojRelativePath: String?
    var mmprojRepoPath: String?

    var primaryURL: URL {
        LocalModelPaths.modelsDirectory.appendingPathComponent(primaryRelativePath)
    }

    var mmprojURL: URL? {
        mmprojRelativePath.map { LocalModelPaths.modelsDirectory.appendingPathComponent($0) }
    }

    var supportsVision: Bool {
        guard let url = mmprojURL else { return false }
        return FileManager.default.fileExists(atPath: url.path)
    }

    var folderURL: URL {
        LocalModelPaths.modelsDirectory.appendingPathComponent(folderRelativePath, isDirectory: true)
    }
}

// MARK: - Paths

enum LocalModelPaths {
    /// Application Support/LocalModels. Excluded from iCloud/iTunes backup:
    /// model files are large and can always be downloaded again.
    static let rootDirectory: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let dir = base.appendingPathComponent("LocalModels", isDirectory: true)
        ensureDirectory(dir)
        return dir
    }()

    static let modelsDirectory: URL = {
        let dir = rootDirectory.appendingPathComponent("models", isDirectory: true)
        ensureDirectory(dir)
        return dir
    }()

    static var catalogFile: URL { rootDirectory.appendingPathComponent("catalog.json") }
    static var jobsFile: URL { rootDirectory.appendingPathComponent("download-jobs.json") }

    static func ensureDirectory(_ url: URL) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutable = url
        try? mutable.setResourceValues(values)
    }

    /// A filesystem-safe folder name for a repo + model.
    static func sanitize(_ s: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        let mapped = s.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" }
        return String(mapped)
    }
}

// MARK: - Catalog (thread-safe)

/// The list of downloaded models, persisted as JSON. Readable from any thread.
final class LocalModelCatalog: @unchecked Sendable {
    static let shared = LocalModelCatalog()

    private let lock = NSLock()
    private var cache: [LocalModelFile]

    private init() {
        cache = Self.readFromDisk()
    }

    var models: [LocalModelFile] {
        lock.lock(); defer { lock.unlock() }
        return cache
    }

    func model(id: String) -> LocalModelFile? {
        models.first { $0.id == id }
    }

    func upsert(_ model: LocalModelFile) {
        lock.lock()
        if let i = cache.firstIndex(where: { $0.id == model.id }) {
            cache[i] = model
        } else {
            cache.append(model)
        }
        let snapshot = cache
        lock.unlock()
        Self.writeToDisk(snapshot)
    }

    func remove(id: String) {
        lock.lock()
        cache.removeAll { $0.id == id }
        let snapshot = cache
        lock.unlock()
        Self.writeToDisk(snapshot)
    }

    /// A unique, API-friendly model id derived from a GGUF file name.
    func makeUniqueId(for fileName: String) -> String {
        var base = (fileName as NSString).lastPathComponent
        if base.lowercased().hasSuffix(".gguf") { base = String(base.dropLast(5)) }
        if let r = base.range(of: #"-\d{5}-of-\d{5}$"#, options: .regularExpression) {
            base.removeSubrange(r)
        }
        base = base.lowercased()
        let existing = Set(models.map(\.id))
        if !existing.contains(base) { return base }
        var n = 2
        while existing.contains("\(base)-\(n)") { n += 1 }
        return "\(base)-\(n)"
    }

    private static func readFromDisk() -> [LocalModelFile] {
        guard let data = try? Data(contentsOf: LocalModelPaths.catalogFile) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let list = (try? decoder.decode([LocalModelFile].self, from: data)) ?? []
        // Drop entries whose files were removed behind our back.
        return list.filter { FileManager.default.fileExists(atPath: $0.primaryURL.path) }
    }

    private static func writeToDisk(_ list: [LocalModelFile]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(list) else { return }
        try? data.write(to: LocalModelPaths.catalogFile, options: .atomic)
    }
}

// MARK: - Settings

/// User-tunable inference settings, stored in UserDefaults so both the UI
/// (`@AppStorage`) and the server queue can read them.
enum LocalModelSettings {
    static let contextLengthKey = "localLLM.contextLength"
    static let gpuOffloadKey = "localLLM.gpuOffload"
    static let temperatureKey = "localLLM.temperature"
    static let maxOutputTokensKey = "localLLM.maxOutputTokens"
    static let toolCallingKey = "localLLM.toolCalling"
    static let huggingFaceEndpointKey = "localLLM.huggingFaceEndpoint"

    static let contextLengthOptions = [2048, 4096, 8192, 16384, 32768]
    static let maxOutputOptions = [512, 1024, 2048, 4096, 8192]

    static let defaultHuggingFaceEndpoint = "https://huggingface.co"
    static let huggingFaceMirrorEndpoint = "https://hf-mirror.com"

    private static var defaults: UserDefaults { .standard }

    static var contextLength: Int {
        let v = defaults.integer(forKey: contextLengthKey)
        return v > 0 ? v : 8192
    }

    static var gpuOffload: Bool {
        defaults.object(forKey: gpuOffloadKey) as? Bool ?? true
    }

    static var temperature: Double {
        defaults.object(forKey: temperatureKey) as? Double ?? 0.7
    }

    static var maxOutputTokens: Int {
        let v = defaults.integer(forKey: maxOutputTokensKey)
        return v > 0 ? v : 2048
    }

    static var toolCalling: Bool {
        defaults.object(forKey: toolCallingKey) as? Bool ?? true
    }

    static var huggingFaceEndpoint: String {
        let raw = defaults.string(forKey: huggingFaceEndpointKey)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let value = raw.isEmpty ? defaultHuggingFaceEndpoint : raw
        return value.hasSuffix("/") ? String(value.dropLast()) : value
    }
}

// MARK: - Formatting helpers

enum LocalModelFormat {
    static func bytes(_ n: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: n, countStyle: .file)
    }

    /// Rough RAM needed to run a GGUF of this size with a modest context.
    static func estimatedRAM(forFileSize size: Int64) -> Int64 {
        Int64(Double(size) * 1.2) + 300 * 1_048_576
    }

    /// Whether this device probably has enough memory for a model file.
    /// iOS gives an app well under the physical RAM, so be conservative.
    static func fitsInMemory(fileSize: Int64) -> Bool {
        let physical = Int64(ProcessInfo.processInfo.physicalMemory)
        return estimatedRAM(forFileSize: fileSize) < physical / 2
    }

    /// Quantization label pulled from a GGUF file name, e.g. "Q4_K_M".
    static func quantization(of fileName: String) -> String? {
        let name = (fileName as NSString).lastPathComponent
        let pattern = #"(?i)(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|Q\d|BF16|F16|F32|MXFP4|TQ\d_\d)"#
        guard let r = name.range(of: pattern, options: .regularExpression) else { return nil }
        return String(name[r]).uppercased()
    }
}
