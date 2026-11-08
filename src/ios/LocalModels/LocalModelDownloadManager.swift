//
//  LocalModelDownloadManager.swift
//  Minis — on-device models
//
//  Downloads GGUF files with a background URLSession, so a multi-gigabyte
//  model keeps downloading while the app is suspended. Jobs are persisted, so
//  a relaunch reattaches to running tasks; paused or interrupted parts keep
//  their resume data on disk.
//
//  AppDelegate forwards `handleEventsForBackgroundURLSession` here (see
//  `LocalModelDownloadManager.backgroundSessionIdentifier`).
//

import Foundation
import UIKit

private let logger = AppLogger(category: "LocalModels")

// MARK: - Job model

struct LocalModelDownloadJob: Codable, Identifiable, Hashable {
    enum State: String, Codable { case downloading, paused, failed }

    struct Part: Codable, Hashable {
        var repoPath: String
        var url: URL
        var expectedSize: Int64
        /// Destination relative to `LocalModelPaths.modelsDirectory`.
        var relativePath: String
        var done: Bool
        /// The vision projector (`mmproj`) rather than a model file.
        var isProjector: Bool?
    }

    var id: String
    var modelId: String
    var displayName: String
    var source: LocalModelSource
    var repoId: String
    var folderRelativePath: String
    var parts: [Part]
    var state: State
    var errorMessage: String?
    var createdAt: Date
    /// Set when this job only adds a vision projector to an existing model.
    var attachToModelId: String?

    var totalSize: Int64 { parts.reduce(0) { $0 + $1.expectedSize } }
    var modelParts: [Part] { parts.filter { $0.isProjector != true } }
    var projectorPart: Part? { parts.first { $0.isProjector == true } }
}

// MARK: - Manager

@MainActor
final class LocalModelDownloadManager: NSObject, ObservableObject {
    static let shared = LocalModelDownloadManager()
    static let backgroundSessionIdentifier = "com.openminis.app.localmodels.download"

    /// Set by AppDelegate when iOS relaunches the app for background session events.
    static var backgroundCompletionHandler: (() -> Void)?

    @Published private(set) var models: [LocalModelFile] = []
    @Published private(set) var jobs: [LocalModelDownloadJob] = []
    /// Bytes written so far per job (all parts), for progress bars.
    @Published private(set) var progress: [String: Int64] = [:]

    private var partProgress: [String: [Int: Int64]] = [:]
    private var lastProgressPublish: [String: Date] = [:]
    private var session: URLSession!
    private let delegateProxy = DownloadDelegateProxy()

    private override init() {
        super.init()
        models = LocalModelCatalog.shared.models
        jobs = Self.loadJobs()
        delegateProxy.owner = self
        let config = URLSessionConfiguration.background(withIdentifier: Self.backgroundSessionIdentifier)
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.allowsCellularAccess = true
        config.timeoutIntervalForRequest = 120
        session = URLSession(configuration: config, delegate: delegateProxy, delegateQueue: nil)
        reattachTasks()
    }

    /// Call once at launch so the background session is recreated early.
    func activate() {
        models = LocalModelCatalog.shared.models
    }

    // MARK: Public API

    func isDownloading(repo: String, option: GGUFOption, source: LocalModelSource) -> Bool {
        jobs.contains { $0.repoId == repo && $0.source == source && $0.modelParts.first?.repoPath == option.primaryPath }
    }

    func isDownloaded(repo: String, option: GGUFOption, source: LocalModelSource) -> Bool {
        downloadedModel(repo: repo, option: option, source: source) != nil
    }

    func downloadedModel(repo: String, option: GGUFOption, source: LocalModelSource) -> LocalModelFile? {
        models.first { $0.repoId == repo && $0.source == source && $0.repoPaths.first == option.primaryPath }
    }

    func isDownloadingProjector(for modelId: String) -> Bool {
        jobs.contains { $0.attachToModelId == modelId }
    }

    /// Starts downloading every part of a GGUF option, plus the vision
    /// projector when one is given.
    func start(repo: String, option: GGUFOption, projector: HubFile?, source: LocalModelSource) throws {
        guard !isDownloading(repo: repo, option: option, source: source) else { return }
        try checkFreeSpace(option.totalSize + (projector?.size ?? 0))
        let modelId = LocalModelCatalog.shared.makeUniqueId(for: option.primaryPath)
        let folder = LocalModelPaths.sanitize("\(source.rawValue)__\(repo)__\(modelId)")
        var parts: [LocalModelDownloadJob.Part] = []
        for file in option.parts {
            parts.append(try makePart(repo: repo, file: file, source: source, folder: folder, isProjector: false))
        }
        if let projector {
            parts.append(try makePart(repo: repo, file: projector, source: source, folder: folder, isProjector: true))
        }
        LocalModelPaths.ensureDirectory(LocalModelPaths.modelsDirectory.appendingPathComponent(folder, isDirectory: true))
        let job = LocalModelDownloadJob(
            id: UUID().uuidString, modelId: modelId, displayName: option.displayName,
            source: source, repoId: repo, folderRelativePath: folder, parts: parts,
            state: .downloading, errorMessage: nil, createdAt: Date()
        )
        launch(job)
        logger.info("[Download] start \(source.rawValue) \(repo) \(option.primaryPath) parts=\(parts.count) size=\(job.totalSize)")
    }

    /// Adds image support to a model that is already downloaded.
    func startProjector(for model: LocalModelFile, projector: HubFile) throws {
        guard !isDownloadingProjector(for: model.id) else { return }
        try checkFreeSpace(projector.size)
        let part = try makePart(repo: model.repoId, file: projector, source: model.source,
                                folder: model.folderRelativePath, isProjector: true)
        var job = LocalModelDownloadJob(
            id: UUID().uuidString, modelId: model.id,
            displayName: AppLocalized("Vision for \(model.displayName)"),
            source: model.source, repoId: model.repoId, folderRelativePath: model.folderRelativePath,
            parts: [part], state: .downloading, errorMessage: nil, createdAt: Date()
        )
        job.attachToModelId = model.id
        launch(job)
    }

    // MARK: Other sources

    /// Downloads a GGUF from a direct link (Hugging Face / ModelScope file
    /// pages are converted to their download URL).
    func startLinkDownload(_ raw: String) throws {
        guard let url = LocalModelHub.directDownloadURL(from: raw) else {
            throw LocalModelHubError.badURL
        }
        var name = url.lastPathComponent.removingPercentEncoding ?? url.lastPathComponent
        if url.host?.contains("modelscope") == true,
           let filePath = URLComponents(url: url, resolvingAgainstBaseURL: false)?
               .queryItems?.first(where: { $0.name == "FilePath" })?.value {
            name = (filePath as NSString).lastPathComponent
        }
        if !name.lowercased().hasSuffix(".gguf") { name += ".gguf" }
        let isProjector = name.lowercased().contains("mmproj")
        if isProjector {
            throw NSError(domain: "LocalModels", code: 2, userInfo: [
                NSLocalizedDescriptionKey: AppLocalized("This link is a vision projector (mmproj). Download the model first, then add the projector from the model's menu."),
            ])
        }
        let modelId = LocalModelCatalog.shared.makeUniqueId(for: name)
        let folder = LocalModelPaths.sanitize("url__\(url.host ?? "link")__\(modelId)")
        LocalModelPaths.ensureDirectory(LocalModelPaths.modelsDirectory.appendingPathComponent(folder, isDirectory: true))
        let part = LocalModelDownloadJob.Part(repoPath: url.absoluteString, url: url, expectedSize: 0,
                                              relativePath: "\(folder)/\(LocalModelPaths.sanitize(name))",
                                              done: false, isProjector: false)
        let job = LocalModelDownloadJob(
            id: UUID().uuidString, modelId: modelId, displayName: name,
            source: .url, repoId: url.host ?? "link", folderRelativePath: folder, parts: [part],
            state: .downloading, errorMessage: nil, createdAt: Date()
        )
        launch(job)
    }

    /// Copies a GGUF picked in the Files app into the models folder.
    /// `url` must be readable (security scope handled here).
    func importModel(from url: URL) async throws {
        let name = url.lastPathComponent
        guard name.lowercased().hasSuffix(".gguf") else {
            throw NSError(domain: "LocalModels", code: 3, userInfo: [
                NSLocalizedDescriptionKey: AppLocalized("Only .gguf model files can be imported."),
            ])
        }
        let modelId = LocalModelCatalog.shared.makeUniqueId(for: name)
        let folder = LocalModelPaths.sanitize("imported__\(modelId)")
        let relative = "\(folder)/\(LocalModelPaths.sanitize(name))"
        let dest = LocalModelPaths.modelsDirectory.appendingPathComponent(relative)
        let size = try await Self.copyIn(url, to: dest)
        let model = LocalModelFile(
            id: modelId, displayName: name, source: .imported, repoId: AppLocalized("Files"),
            repoPaths: [name], primaryRelativePath: relative, folderRelativePath: folder,
            sizeBytes: size, downloadedAt: Date()
        )
        LocalModelCatalog.shared.upsert(model)
        models = LocalModelCatalog.shared.models
        logger.info("[Import] \(name) -> id=\(modelId) size=\(size)")
        LocalModelProviderRegistrar.shared.modelsChanged()
    }

    /// Adds a vision projector picked in the Files app to an existing model.
    func importProjector(from url: URL, for modelId: String) async throws {
        guard var model = LocalModelCatalog.shared.model(id: modelId) else { return }
        let name = url.lastPathComponent
        guard name.lowercased().hasSuffix(".gguf") else {
            throw NSError(domain: "LocalModels", code: 3, userInfo: [
                NSLocalizedDescriptionKey: AppLocalized("Only .gguf model files can be imported."),
            ])
        }
        let relative = "\(model.folderRelativePath)/mmproj-\(LocalModelPaths.sanitize(name))"
        let dest = LocalModelPaths.modelsDirectory.appendingPathComponent(relative)
        let size = try await Self.copyIn(url, to: dest)
        model.mmprojRelativePath = relative
        model.mmprojRepoPath = name
        model.sizeBytes += size
        LocalModelCatalog.shared.upsert(model)
        models = LocalModelCatalog.shared.models
        LlamaEngine.shared.unloadIfUsing(path: model.primaryURL.path)
        LocalModelProviderRegistrar.shared.modelsChanged()
    }

    /// Copies a (possibly multi-GB) file off the main thread and checks that
    /// it is a GGUF. Returns its size.
    private static func copyIn(_ source: URL, to dest: URL) async throws -> Int64 {
        try await Task.detached(priority: .userInitiated) {
            let accessed = source.startAccessingSecurityScopedResource()
            defer { if accessed { source.stopAccessingSecurityScopedResource() } }
            let fm = FileManager.default
            // Coordinated read so iCloud / provider files are downloaded first.
            var coordError: NSError?
            var copyError: Error?
            NSFileCoordinator().coordinate(readingItemAt: source, options: [], error: &coordError) { readURL in
                do {
                    LocalModelPaths.ensureDirectory(dest.deletingLastPathComponent())
                    if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
                    try fm.copyItem(at: readURL, to: dest)
                } catch {
                    copyError = error
                }
            }
            if let error = coordError ?? copyError { throw error }
            guard let handle = try? FileHandle(forReadingFrom: dest) else {
                throw LlamaEngineError.modelLoadFailed(dest.path)
            }
            let magic = (try? handle.read(upToCount: 4)) ?? Data()
            try? handle.close()
            guard magic == Data("GGUF".utf8) else {
                try? fm.removeItem(at: dest)
                throw NSError(domain: "LocalModels", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: AppLocalized("That file is not a GGUF model."),
                ])
            }
            let attrs = try? fm.attributesOfItem(atPath: dest.path)
            return (attrs?[.size] as? NSNumber)?.int64Value ?? 0
        }.value
    }

    private func checkFreeSpace(_ needed: Int64) throws {
        if let free = Self.freeDiskSpace(), free < needed + 200 * 1_048_576 {
            throw NSError(domain: "LocalModels", code: 1, userInfo: [
                NSLocalizedDescriptionKey: AppLocalized("Not enough free space: \(LocalModelFormat.bytes(needed)) needed, \(LocalModelFormat.bytes(free)) available."),
            ])
        }
    }

    private func makePart(repo: String, file: HubFile, source: LocalModelSource, folder: String, isProjector: Bool) throws -> LocalModelDownloadJob.Part {
        guard let url = LocalModelHub.downloadURL(repo: repo, path: file.path, source: source) else {
            throw LocalModelHubError.badURL
        }
        let name = (file.path as NSString).lastPathComponent
        return .init(repoPath: file.path, url: url, expectedSize: file.size,
                     relativePath: "\(folder)/\(name)", done: false, isProjector: isProjector)
    }

    private func launch(_ job: LocalModelDownloadJob) {
        jobs.append(job)
        saveJobs()
        for index in job.parts.indices { startTask(job: job, partIndex: index) }
    }

    func pause(jobId: String) {
        guard let i = jobs.firstIndex(where: { $0.id == jobId }) else { return }
        jobs[i].state = .paused
        saveJobs()
        session.getAllTasks { tasks in
            for task in tasks {
                guard let info = TaskInfo(task.taskDescription), info.jobId == jobId,
                      let download = task as? URLSessionDownloadTask else { continue }
                download.cancel(byProducingResumeData: { data in
                    if let data { try? data.write(to: Self.resumeDataURL(jobId: jobId, part: info.partIndex)) }
                })
            }
        }
    }

    func resume(jobId: String) {
        guard let i = jobs.firstIndex(where: { $0.id == jobId }) else { return }
        jobs[i].state = .downloading
        jobs[i].errorMessage = nil
        saveJobs()
        let job = jobs[i]
        session.getAllTasks { tasks in
            let running = Set(tasks.compactMap { TaskInfo($0.taskDescription) }
                .filter { $0.jobId == jobId }.map(\.partIndex))
            Task { @MainActor in
                for index in job.parts.indices where !job.parts[index].done && !running.contains(index) {
                    self.startTask(job: job, partIndex: index)
                }
            }
        }
    }

    func cancel(jobId: String) {
        guard let i = jobs.firstIndex(where: { $0.id == jobId }) else { return }
        let job = jobs.remove(at: i)
        progress[jobId] = nil
        partProgress[jobId] = nil
        saveJobs()
        session.getAllTasks { tasks in
            for task in tasks where TaskInfo(task.taskDescription)?.jobId == jobId { task.cancel() }
        }
        for index in job.parts.indices {
            try? FileManager.default.removeItem(at: Self.resumeDataURL(jobId: jobId, part: index))
        }
        if job.attachToModelId != nil {
            // The folder belongs to an existing model: remove only this file.
            for part in job.parts {
                try? FileManager.default.removeItem(at: LocalModelPaths.modelsDirectory.appendingPathComponent(part.relativePath))
            }
        } else {
            try? FileManager.default.removeItem(at: LocalModelPaths.modelsDirectory.appendingPathComponent(job.folderRelativePath, isDirectory: true))
        }
    }

    func deleteModel(id: String) {
        guard let model = models.first(where: { $0.id == id }) else { return }
        // Stop anything running first so the unload is not stuck behind it.
        if LlamaEngine.shared.loadedPath == model.primaryURL.path {
            LlamaEngine.shared.cancelAll()
        }
        LlamaEngine.shared.unloadIfUsing(path: model.primaryURL.path)
        try? FileManager.default.removeItem(at: model.folderURL)
        LocalModelCatalog.shared.remove(id: id)
        models = LocalModelCatalog.shared.models
        LocalModelProviderRegistrar.shared.modelsChanged()
    }

    func renameModel(id: String, displayName: String) {
        guard var model = LocalModelCatalog.shared.model(id: id) else { return }
        model.displayName = displayName
        LocalModelCatalog.shared.upsert(model)
        models = LocalModelCatalog.shared.models
    }

    var totalDiskUsage: Int64 { models.reduce(0) { $0 + $1.sizeBytes } }

    // MARK: Tasks

    private func startTask(job: LocalModelDownloadJob, partIndex: Int) {
        let part = job.parts[partIndex]
        let resumeURL = Self.resumeDataURL(jobId: job.id, part: partIndex)
        let task: URLSessionDownloadTask
        if let data = try? Data(contentsOf: resumeURL) {
            task = session.downloadTask(withResumeData: data)
            try? FileManager.default.removeItem(at: resumeURL)
        } else {
            var request = URLRequest(url: part.url)
            request.setValue("Minis-LocalModels/1.0", forHTTPHeaderField: "User-Agent")
            task = session.downloadTask(with: request)
        }
        task.taskDescription = TaskInfo(jobId: job.id, partIndex: partIndex, relativePath: part.relativePath).encoded
        task.countOfBytesClientExpectsToReceive = part.expectedSize
        task.resume()
    }

    private func reattachTasks() {
        session.getAllTasks { tasks in
            let running = tasks.compactMap { TaskInfo($0.taskDescription) }
            Task { @MainActor in
                for job in self.jobs where job.state == .downloading {
                    for index in job.parts.indices where !job.parts[index].done {
                        let isRunning = running.contains { $0.jobId == job.id && $0.partIndex == index }
                        if !isRunning { self.startTask(job: job, partIndex: index) }
                    }
                }
            }
        }
    }

    // MARK: Delegate callbacks (hopped to the main actor)

    fileprivate func didWrite(info: TaskInfo, written: Int64, expected: Int64) {
        // Link downloads start without a known size; take it from the response.
        if expected > 0, let i = jobs.firstIndex(where: { $0.id == info.jobId }),
           info.partIndex < jobs[i].parts.count, jobs[i].parts[info.partIndex].expectedSize == 0 {
            jobs[i].parts[info.partIndex].expectedSize = expected
            saveJobs()
        }
        partProgress[info.jobId, default: [:]][info.partIndex] = written
        let now = Date()
        if let last = lastProgressPublish[info.jobId], now.timeIntervalSince(last) < 0.3 { return }
        lastProgressPublish[info.jobId] = now
        progress[info.jobId] = partProgress[info.jobId]?.values.reduce(0, +) ?? 0
    }

    fileprivate func didComplete(info: TaskInfo, error: String?, resumeData: Data?, cancelled: Bool) {
        guard let i = jobs.firstIndex(where: { $0.id == info.jobId }) else { return }
        if let error {
            if let resumeData {
                try? resumeData.write(to: Self.resumeDataURL(jobId: info.jobId, part: info.partIndex))
            }
            // A pause cancels with resume data; that is not a failure.
            if cancelled && jobs[i].state == .paused { return }
            jobs[i].state = .failed
            jobs[i].errorMessage = error
            saveJobs()
            logger.error("[Download] job=\(info.jobId.prefix(8)) part=\(info.partIndex) failed: \(error)")
            return
        }
        jobs[i].parts[info.partIndex].done = true
        partProgress[info.jobId, default: [:]][info.partIndex] = jobs[i].parts[info.partIndex].expectedSize
        progress[info.jobId] = partProgress[info.jobId]?.values.reduce(0, +) ?? 0
        if jobs[i].parts.allSatisfy(\.done) {
            finish(job: jobs.remove(at: i))
        }
        saveJobs()
    }

    private func finish(job: LocalModelDownloadJob) {
        let fm = FileManager.default
        progress[job.id] = nil
        partProgress[job.id] = nil
        if let targetId = job.attachToModelId {
            if var model = LocalModelCatalog.shared.model(id: targetId), let projector = job.projectorPart {
                model.mmprojRelativePath = projector.relativePath
                model.mmprojRepoPath = projector.repoPath
                model.sizeBytes += projector.expectedSize
                LocalModelCatalog.shared.upsert(model)
                LlamaEngine.shared.unloadIfUsing(path: model.primaryURL.path)
            }
            models = LocalModelCatalog.shared.models
            LocalModelProviderRegistrar.shared.modelsChanged()
            return
        }
        let size = job.parts.reduce(Int64(0)) { sum, part in
            let path = LocalModelPaths.modelsDirectory.appendingPathComponent(part.relativePath).path
            let attrs = try? fm.attributesOfItem(atPath: path)
            return sum + ((attrs?[.size] as? NSNumber)?.int64Value ?? part.expectedSize)
        }
        let modelParts = job.modelParts
        let model = LocalModelFile(
            id: job.modelId, displayName: job.displayName, source: job.source, repoId: job.repoId,
            repoPaths: modelParts.map(\.repoPath), primaryRelativePath: modelParts[0].relativePath,
            folderRelativePath: job.folderRelativePath, sizeBytes: size, downloadedAt: Date(),
            mmprojRelativePath: job.projectorPart?.relativePath,
            mmprojRepoPath: job.projectorPart?.repoPath
        )
        LocalModelCatalog.shared.upsert(model)
        models = LocalModelCatalog.shared.models
        progress[job.id] = nil
        partProgress[job.id] = nil
        logger.info("[Download] finished \(job.repoId) -> id=\(job.modelId) size=\(size)")
        LocalModelProviderRegistrar.shared.modelsChanged()
    }

    fileprivate func sessionFinishedEvents() {
        Self.backgroundCompletionHandler?()
        Self.backgroundCompletionHandler = nil
    }

    // MARK: Persistence

    private func saveJobs() {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(jobs) {
            try? data.write(to: LocalModelPaths.jobsFile, options: .atomic)
        }
    }

    private static func loadJobs() -> [LocalModelDownloadJob] {
        guard let data = try? Data(contentsOf: LocalModelPaths.jobsFile) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([LocalModelDownloadJob].self, from: data)) ?? []
    }

    nonisolated static func resumeDataURL(jobId: String, part: Int) -> URL {
        let dir = LocalModelPaths.rootDirectory.appendingPathComponent("resume", isDirectory: true)
        LocalModelPaths.ensureDirectory(dir)
        return dir.appendingPathComponent("\(jobId)-\(part).resume")
    }

    private static func freeDiskSpace() -> Int64? {
        let values = try? LocalModelPaths.rootDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }
}

// MARK: - Task description encoding

fileprivate struct TaskInfo {
    let jobId: String
    let partIndex: Int
    let relativePath: String

    init(jobId: String, partIndex: Int, relativePath: String) {
        self.jobId = jobId
        self.partIndex = partIndex
        self.relativePath = relativePath
    }

    init?(_ description: String?) {
        guard let description else { return nil }
        let pieces = description.split(separator: "\n", maxSplits: 2, omittingEmptySubsequences: false)
        guard pieces.count == 3, let index = Int(pieces[1]) else { return nil }
        jobId = String(pieces[0])
        partIndex = index
        relativePath = String(pieces[2])
    }

    var encoded: String { "\(jobId)\n\(partIndex)\n\(relativePath)" }
}

// MARK: - URLSession delegate (runs on the session's queue)

private final class DownloadDelegateProxy: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    weak var owner: LocalModelDownloadManager?

    /// Errors detected while moving a finished file, reported again in didCompleteWithError.
    private let lock = NSLock()
    private var moveErrors: [Int: String] = [:]

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard let info = TaskInfo(downloadTask.taskDescription) else { return }
        Task { @MainActor [weak owner = self.owner] in
            owner?.didWrite(info: info, written: totalBytesWritten, expected: totalBytesExpectedToWrite)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The temporary file is deleted when this method returns, so the move
        // has to happen synchronously here.
        guard let info = TaskInfo(downloadTask.taskDescription) else { return }
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 200
        guard (200..<300).contains(status) else {
            setMoveError(downloadTask.taskIdentifier, status == 401 || status == 403
                ? "HTTP \(status): this repository is gated or private."
                : "HTTP \(status)")
            return
        }
        guard Self.hasGGUFMagic(location) else {
            setMoveError(downloadTask.taskIdentifier, "The downloaded file is not a GGUF model.")
            return
        }
        let dest = LocalModelPaths.modelsDirectory.appendingPathComponent(info.relativePath)
        let fm = FileManager.default
        do {
            LocalModelPaths.ensureDirectory(dest.deletingLastPathComponent())
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: location, to: dest)
        } catch {
            setMoveError(downloadTask.taskIdentifier, error.localizedDescription)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let info = TaskInfo(task.taskDescription) else { return }
        let moveError = takeMoveError(task.taskIdentifier)
        let nsError = error as NSError?
        let resumeData = nsError?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let cancelled = nsError?.code == NSURLErrorCancelled
        let message = moveError ?? error?.localizedDescription
        Task { @MainActor [weak owner = self.owner] in
            owner?.didComplete(info: info, error: message, resumeData: resumeData, cancelled: cancelled)
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor [weak owner = self.owner] in owner?.sessionFinishedEvents() }
    }

    private func setMoveError(_ id: Int, _ message: String) {
        lock.lock(); moveErrors[id] = message; lock.unlock()
    }

    private func takeMoveError(_ id: Int) -> String? {
        lock.lock(); defer { lock.unlock() }
        return moveErrors.removeValue(forKey: id)
    }

    private static func hasGGUFMagic(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        let magic = (try? handle.read(upToCount: 4)) ?? Data()
        return magic == Data("GGUF".utf8)
    }
}
