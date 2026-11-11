//
//  LlamaEngine.swift
//  Minis — on-device models
//
//  A small wrapper over the llama.cpp C API (llama.xcframework, fetched by
//  deps/fetch_llama_xcframework.sh). One model and one context are kept
//  loaded; every call runs on a private serial queue, so concurrent requests
//  simply wait their turn.
//
//  Prefix cache: the agent resends the same long system prompt and history on
//  every turn. The tokens already in the KV cache are compared with the new
//  prompt and only the differing tail is evaluated, which turns a multi-second
//  prefill into a short one from the second turn on.
//

import CoreGraphics
import CryptoKit
import Foundation
import os
import ImageIO
import llama
import UIKit

private let logger = AppLogger(category: "LlamaEngine")

enum LlamaEngineError: LocalizedError {
    case modelLoadFailed(String)
    case contextCreateFailed
    case promptTooLong(promptTokens: Int, contextLength: Int)
    case decodeFailed(Int32)
    case templateFailed
    case visionUnavailable
    case imageDecodeFailed
    case notEnoughMemory(neededMB: Int, availableMB: Int)

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let path):
            return AppLocalized("Could not load the model file \((path as NSString).lastPathComponent). It may be corrupt or too large for this device's memory.")
        case .contextCreateFailed:
            return AppLocalized("Could not create the inference context. Try a smaller context length in Local Models settings.")
        case .promptTooLong(let p, let c):
            return AppLocalized("The conversation (\(p) tokens) does not fit the local model's context (\(c) tokens). Raise the context length in Local Models settings, turn off tool calling, or start a new chat.")
        case .decodeFailed(let code):
            let rc = Int(code)
            return AppLocalized("Local inference failed (llama_decode returned \(rc)). If the app was in the background, bring it to the foreground and retry.")
        case .templateFailed:
            return AppLocalized("Could not format the conversation for this model.")
        case .visionUnavailable:
            return AppLocalized("Could not load the vision projector for this model, so it cannot read images.")
        case .imageDecodeFailed:
            return AppLocalized("Could not read one of the attached images.")
        case .notEnoughMemory(let need, let avail):
            return AppLocalized("This model needs about \(need) MB of memory but iOS only lets the app use about \(avail) MB right now, so loading it would crash the app. Pick a smaller model or quantization (for example a 1–4B Q4_K_M), lower the context length, or close other apps and try again.")
        }
    }
}

struct LlamaChatMessage: Sendable {
    let role: String
    let content: String
}

struct LlamaSampling: Sendable {
    var temperature: Float
    var topP: Float
    var topK: Int32
    var minP: Float
    var maxTokens: Int
}

enum LlamaGenerationEvent: Sendable {
    /// The prompt has been evaluated; `reused` tokens came from the prefix cache.
    case promptEvaluated(promptTokens: Int, reused: Int)
    case text(String)
    case finished(completionTokens: Int, hitLengthLimit: Bool)
}

final class LlamaEngine: @unchecked Sendable {
    static let shared = LlamaEngine()

    struct Config: Equatable {
        var path: String
        var contextLength: Int
        var gpuOffload: Bool
        /// Vision projector; nil for text-only use.
        var mmprojPath: String? = nil
    }

    /// The placeholder that marks where an image goes in the prompt text.
    static let mediaMarker: String = String(cString: mtmd_default_marker())

    /// One evaluated piece of a multimodal prompt, used for prefix reuse.
    private enum ChunkKey: Equatable {
        case text([llama_token])
        case image(String)
    }

    private let queue = DispatchQueue(label: "com.openminis.app.llama", qos: .userInitiated)

    // All state below is only touched on `queue`.
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var vocab: OpaquePointer?
    private var config: Config?
    private var cachedTokens: [llama_token] = []
    private var backendReady = false
    private var mtmdContext: OpaquePointer?
    /// Prefix cache for multimodal prompts: chunk keys and the KV position
    /// after each chunk. Text-only and multimodal caches never coexist.
    private var cachedChunks: [ChunkKey] = []
    private var cachedChunkEnds: [llama_pos] = []

    /// Model currently loaded (path), readable from any thread for the UI.
    private let statusLock = NSLock()
    private var _loadedPath: String?
    var loadedPath: String? {
        statusLock.lock(); defer { statusLock.unlock() }
        return _loadedPath
    }

    /// What the engine is doing right now, for the live status in Settings.
    enum Phase: Equatable {
        case idle
        case loading(String)
        case ready(String)
        case generating(String)
    }
    private var _phase: Phase = .idle
    private var _phaseSince = Date()
    private var _generatedChunks = 0
    var phase: (phase: Phase, since: Date, chunks: Int) {
        statusLock.lock(); defer { statusLock.unlock() }
        return (_phase, _phaseSince, _generatedChunks)
    }
    private func setPhase(_ p: Phase) {
        statusLock.lock()
        if _phase != p { _phase = p; _phaseSince = Date() }
        if case .generating = p {} else { _generatedChunks = 0 }
        statusLock.unlock()
    }
    private func countChunk() {
        statusLock.lock(); _generatedChunks += 1; statusLock.unlock()
    }
    private var loadedName: String {
        ((loadedPath ?? "") as NSString).lastPathComponent
    }
    /// Number of requests waiting behind the running one.
    var queuedCount: Int {
        activeLock.lock(); defer { activeLock.unlock() }
        return max(0, activeFlags.count - (busySince == nil ? 0 : 1))
    }

    /// Cancel flags of requests that are running or waiting for the queue,
    /// so "unload" can stop them instead of waiting behind them.
    private let activeLock = NSLock()
    private var activeFlags: [ObjectIdentifier: LlamaCancelFlag] = [:]

    /// Longest a single request may run before it is stopped, so one runaway
    /// generation cannot hold the engine (and every later load/unload) hostage.
    static let maxRequestSeconds: TimeInterval = 300
    /// The hard limit for one request. With Enhanced Background on, a reply may
    /// legitimately sit paused while the app is in the background waiting to
    /// use the GPU again, so the limit is longer.
    static var requestTimeLimit: TimeInterval {
        enhancedBackgroundEffective ? 1800 : maxRequestSeconds
    }

    private var backgroundObserver: NSObjectProtocol?
    private var foregroundObserver: NSObjectProtocol?
    private var activeObserver: NSObjectProtocol?

    private init() {
        // When the app is backgrounded, a decode in flight can stall or fail
        // because iOS restricts background GPU work. By default we stop
        // generations so the queue isn't left blocked. But if the user turned
        // on Enhanced Background (which keeps the app alive with a silent
        // audio session), let the current generation keep running instead of
        // cutting the reply off mid-sentence.
        backgroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.appInBackground = true
            if Self.enhancedBackgroundEffective {
                logger.info("[Engine] backgrounded with Enhanced Background on — pausing GPU decode until foreground")
            } else {
                self?.cancelAll()
            }
        }
        foregroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.willEnterForegroundNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.appInBackground = false
        }
        activeObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didBecomeActiveNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.appInBackground = false
        }
    }

    /// Nonisolated read of the Enhanced Background setting (master switch plus
    /// the silent-audio keep-alive that actually prevents suspension), so the
    /// background notification can consult it without hopping to the main actor.
    static var enhancedBackgroundEffective: Bool {
        let d = UserDefaults.standard
        return d.bool(forKey: "enhancedBackgroundExecution") && d.bool(forKey: "backgroundSpeakEnabled")
    }

    // MARK: Background-safe decoding

    /// Whether the app is currently in the background. iOS refuses GPU work
    /// there, so a decode (text or image) returns a negative error. Updated by
    /// the notifications set up in init; read off the main thread by the
    /// decode loop, so it is guarded by its own lock.
    private let bgStateLock = NSLock()
    private var _appInBackground = false
    private var appInBackground: Bool {
        get { bgStateLock.lock(); defer { bgStateLock.unlock() }; return _appInBackground }
        set { bgStateLock.lock(); _appInBackground = newValue; bgStateLock.unlock() }
    }

    /// Blocks the engine queue until the app is back in the foreground (so GPU
    /// work is allowed again), the request is cancelled, or the request time
    /// limit passes. Returns true if the foreground was reached.
    private func waitForForeground(cancel: LlamaCancelFlag) -> Bool {
        let deadline = Date().addingTimeInterval(Self.requestTimeLimit)
        while appInBackground, !cancel.isCancelled, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.25)
        }
        return !appInBackground && !cancel.isCancelled
    }

    /// Runs a decode and, if it fails only because the app is in the
    /// background with Enhanced Background on, waits for the foreground and
    /// retries instead of failing the whole reply. A failed decode does not
    /// advance the KV cache, so retrying the same batch is safe.
    private func resilientDecode(cancel: LlamaCancelFlag, _ body: () -> Int32) -> Int32 {
        var rc = body()
        var attempts = 0
        while rc < 0, !cancel.isCancelled, Self.enhancedBackgroundEffective, appInBackground, attempts < 240 {
            logger.info("[Decode] rc=\(rc) in background — waiting for foreground to retry")
            if !waitForForeground(cancel: cancel) { break }
            rc = body()
            attempts += 1
        }
        return rc
    }

    /// True while a request is being processed on the engine queue.
    private let busyLock = NSLock()
    private var _busySince: Date?
    var busySince: Date? { busyLock.lock(); defer { busyLock.unlock() }; return _busySince }
    private func setBusy(_ on: Bool) {
        busyLock.lock(); _busySince = on ? Date() : nil; busyLock.unlock()
        if on {
            setPhase(.generating(loadedName))
        } else {
            setPhase(loadedPath == nil ? .idle : .ready(loadedName))
        }
    }

    private func track(_ flag: LlamaCancelFlag) {
        activeLock.lock(); activeFlags[ObjectIdentifier(flag)] = flag; activeLock.unlock()
    }

    private func untrack(_ flag: LlamaCancelFlag) {
        activeLock.lock(); activeFlags[ObjectIdentifier(flag)] = nil; activeLock.unlock()
    }

    /// Stops every running or queued generation.
    func cancelAll() {
        activeLock.lock()
        let flags = Array(activeFlags.values)
        activeLock.unlock()
        flags.forEach { $0.cancel() }
    }

    // MARK: Public API

    /// Runs a chat completion. Events arrive in order; cancelling the consuming
    /// task stops generation at the next token.
    /// `images` are encoded image files (JPEG/PNG/HEIC…) in the order their
    /// `mediaMarker` placeholders appear in the messages.
    /// Errors that mean "could not use the image", so the request can retry
    /// as a text-only reply instead of failing.
    static func isVisionError(_ error: Error) -> Bool {
        guard let e = error as? LlamaEngineError else { return false }
        switch e {
        case .visionUnavailable, .imageDecodeFailed, .templateFailed: return true
        default: return false
        }
    }

    func generate(messages: [LlamaChatMessage], images: [Data] = [], config: Config, sampling: LlamaSampling)
        -> AsyncThrowingStream<LlamaGenerationEvent, Error>
    {
        AsyncThrowingStream { continuation in
            let cancelFlag = LlamaCancelFlag()
            continuation.onTermination = { _ in cancelFlag.cancel() }
            track(cancelFlag)
            // Hard time limit for this request.
            DispatchQueue.global().asyncAfter(deadline: .now() + Self.requestTimeLimit) {
                cancelFlag.cancel()
            }
            queue.async {
                defer { self.untrack(cancelFlag) }
                // Cancelled while waiting (e.g. "Unload" was tapped): do not load.
                if cancelFlag.isCancelled {
                    continuation.finish(throwing: CancellationError())
                    return
                }
                // The model file may have been deleted while this waited.
                guard FileManager.default.fileExists(atPath: config.path) else {
                    continuation.finish(throwing: LlamaEngineError.modelLoadFailed(config.path))
                    return
                }
                self.setBusy(true)
                defer { self.setBusy(false) }
                do {
                    try self.ensureLoaded(config)
                    let prompt = try self.applyTemplate(messages)
                    self.setPhase(.generating(self.loadedName))
                    let emit: (LlamaGenerationEvent) -> Void = { event in
                        if case .text = event { self.countChunk() }
                        continuation.yield(event)
                    }
                    if images.isEmpty {
                        try self.run(prompt: prompt, sampling: sampling, cancel: cancelFlag, emit: emit)
                    } else {
                        do {
                            try self.runMultimodal(prompt: prompt, images: images, sampling: sampling,
                                                   cancel: cancelFlag, emit: emit)
                        } catch {
                            // Vision errors (missing or bad projector, image
                            // decode, tokenize) all happen before any token is
                            // produced, so it is safe to answer from the text
                            // instead of failing the whole request with a 500.
                            guard Self.isVisionError(error), !cancelFlag.isCancelled else { throw error }
                            logger.error("[Vision] \(error.localizedDescription) — answering from text only")
                            if let context = self.context {
                                llama_memory_clear(llama_get_memory(context), true)
                            }
                            self.resetCaches()
                            let note = AppLocalized("(I couldn't read the attached image on this device, so I'm answering from the text only.)")
                            let textPrompt = prompt.replacingOccurrences(of: Self.mediaMarker, with: note)
                            emit(.text("⚠️ " + note + "\n\n"))
                            try self.run(prompt: textPrompt, sampling: sampling, cancel: cancelFlag, emit: emit)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Loads a model ahead of the first request (optional warm-up).
    func preload(config: Config) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            queue.async {
                do { try self.ensureLoaded(config); cont.resume() } catch { cont.resume(throwing: error) }
            }
        }
    }

    /// Frees the model if it is the given file (when it is being deleted).
    /// Asynchronous: deleting a memory-mapped file is safe on iOS, the pages
    /// stay valid until the model is freed after any running request.
    func unloadIfUsing(path: String) {
        queue.async {
            if self.config?.path == path { self.unload() }
        }
    }

    /// Frees the model and context. Running and queued requests are stopped
    /// first, so this takes effect right away instead of after them.
    func unloadAll(completion: (@Sendable () -> Void)? = nil) {
        cancelAll()
        queue.async {
            self.unload()
            completion?()
        }
    }

    // MARK: Loading (queue)

    private func ensureLoaded(_ wanted: Config) throws {
        if !backendReady {
            llama_backend_init()
            backendReady = true
        }
        if let config, config == wanted, model != nil, context != nil { return }

        // Same file, different context settings: keep the model, rebuild the context.
        if let config, config.path == wanted.path, config.gpuOffload == wanted.gpuOffload, model != nil {
            if config.contextLength != wanted.contextLength {
                if let context { llama_free(context) }
                context = nil
                resetCaches()
                try createContext(contextLength: try fittedContextLength(wanted.contextLength))
            }
            if config.mmprojPath != wanted.mmprojPath {
                if let mtmdContext { mtmd_free(mtmdContext) }
                mtmdContext = nil
                resetCaches()
                if let context { llama_memory_clear(llama_get_memory(context), true) }
                try loadProjector(wanted)
            }
            self.config = wanted
            return
        }

        unload()
        setPhase(.loading((wanted.path as NSString).lastPathComponent))

        // [T-llama-memory-guard] iOS kills the app outright (no error, no
        // crash log in-app) once it passes its memory limit. With Metal the
        // whole weight file is wired into GPU memory and counts against that
        // limit, so a model bigger than what is left crashes ~seconds into
        // loading. Check first: fall back to CPU (weights stay a clean,
        // evictable file mapping), and refuse if even that cannot fit.
        var fileBytes = Self.modelBytes(wanted.path)
        if let mmproj = wanted.mmprojPath {
            fileBytes += Self.modelBytes(mmproj)
        }
        let available = Self.availableMemory()
        var useGPU = wanted.gpuOffload
        var note: String?
        if let available {
            let mb = { (b: Int64) in Int(b / 1_048_576) }
            if useGPU, fileBytes + Self.gpuHeadroom > available {
                useGPU = false
                note = AppLocalized("Not enough memory for GPU offload (model \(mb(fileBytes)) MB, available \(mb(available)) MB) — running on the CPU, which is slower.")
                logger.warning("[Load] GPU offload skipped: model=\(mb(fileBytes))MB available=\(mb(available))MB")
            }
            let physical = Int64(ProcessInfo.processInfo.physicalMemory)
            if fileBytes > physical * 7 / 10 {
                setPhase(.idle)
                throw LlamaEngineError.notEnoughMemory(neededMB: mb(fileBytes), availableMB: mb(available))
            }
        }
        var mparams = llama_model_default_params()
        #if targetEnvironment(simulator)
        mparams.n_gpu_layers = 0
        #else
        mparams.n_gpu_layers = useGPU ? 999 : 0
        #endif
        let start = Date()
        guard let loaded = llama_model_load_from_file(wanted.path, mparams) else {
            setPhase(.idle)
            throw LlamaEngineError.modelLoadFailed(wanted.path)
        }
        model = loaded
        vocab = llama_model_get_vocab(loaded)
        do {
            let fitted = try fittedContextLength(wanted.contextLength)
            if fitted < wanted.contextLength {
                let msg = AppLocalized("Context lowered to \(fitted) tokens to fit in memory.")
                note = note.map { $0 + " " + msg } ?? msg
            }
            try createContext(contextLength: fitted)
            try loadProjector(wanted)
        } catch {
            unload()
            throw error
        }
        setLoadNote(note)
        config = wanted
        statusLock.lock(); _loadedPath = wanted.path; statusLock.unlock()
        setPhase(busySince == nil ? .ready(loadedName) : .generating(loadedName))
        logger.info("[Load] \((wanted.path as NSString).lastPathComponent) ctx=\(wanted.contextLength) gpu=\(wanted.gpuOffload) in \(String(format: "%.1f", Date().timeIntervalSince(start)))s")
    }

    // MARK: Memory (queue)

    /// Room kept free for Metal command buffers, compute buffers and the app.
    private static let gpuHeadroom: Int64 = 700 * 1_048_576
    private static let contextHeadroom: Int64 = 500 * 1_048_576

    /// Bytes the app may still allocate before iOS terminates it.
    static func availableMemory() -> Int64? {
        #if targetEnvironment(simulator)
        return nil
        #else
        let a = Int64(os_proc_available_memory())
        return a > 0 ? a : nil
        #endif
    }

    /// Size of the model file, including the other parts of a split GGUF
    /// (name-00001-of-00003.gguf).
    static func modelBytes(_ path: String) -> Int64 {
        let fm = FileManager.default
        let size = { (p: String) -> Int64 in
            ((try? fm.attributesOfItem(atPath: p))?[.size] as? NSNumber)?.int64Value ?? 0
        }
        let url = URL(fileURLWithPath: path)
        let name = url.lastPathComponent
        guard let r = name.range(of: #"-\d{5}-of-\d{5}\.gguf$"#, options: .regularExpression) else {
            return size(path)
        }
        let stem = String(name[..<r.lowerBound])
        let dir = url.deletingLastPathComponent()
        let parts = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        return parts.filter { $0.hasPrefix(stem + "-") && $0.hasSuffix(".gguf") }
            .reduce(0) { $0 + size(dir.appendingPathComponent($1).path) }
    }

    /// The largest context (up to `wanted`) whose KV cache fits in what is
    /// left after the weights are loaded. Throws if not even 512 fits.
    private func fittedContextLength(_ wanted: Int) throws -> Int {
        guard let model, let available = Self.availableMemory() else { return wanted }
        let layers = Int64(llama_model_n_layer(model))
        let heads = Int64(max(1, llama_model_n_head(model)))
        let kvHeads = Int64(max(1, llama_model_n_head_kv(model)))
        let embd = Int64(llama_model_n_embd(model))
        guard layers > 0, embd > 0 else { return wanted }
        // K and V, f16, per token.
        let perToken = 2 * layers * kvHeads * (embd / heads) * 2
        let budget = available - Self.contextHeadroom
        let fits = perToken > 0 ? Int(max(0, budget) / perToken) : wanted
        let fitted = min(wanted, fits / 256 * 256)
        logger.info("[Load] kv/token=\(perToken / 1024)KB available=\(available / 1_048_576)MB ctx wanted=\(wanted) fits=\(fits)")
        if fitted < 512 {
            throw LlamaEngineError.notEnoughMemory(
                neededMB: Int((Self.contextHeadroom + perToken * 512) / 1_048_576),
                availableMB: Int(available / 1_048_576))
        }
        return fitted
    }

    private var _loadNote: String?
    /// What the last load had to change to fit (CPU fallback, smaller context).
    var loadNote: String? { statusLock.lock(); defer { statusLock.unlock() }; return _loadNote }
    private func setLoadNote(_ note: String?) {
        statusLock.lock(); _loadNote = note; statusLock.unlock()
    }

    private func createContext(contextLength: Int) throws {
        guard let model else { throw LlamaEngineError.contextCreateFailed }
        var cparams = llama_context_default_params()
        let trained = Int(llama_model_n_ctx_train(model))
        let nCtx = trained > 0 ? min(contextLength, trained) : contextLength
        cparams.n_ctx = UInt32(nCtx)
        cparams.n_batch = 512
        let threads = Int32(max(1, min(6, ProcessInfo.processInfo.activeProcessorCount - 2)))
        cparams.n_threads = threads
        cparams.n_threads_batch = threads
        guard let ctx = llama_init_from_model(model, cparams) else {
            throw LlamaEngineError.contextCreateFailed
        }
        context = ctx
        resetCaches()
    }

    private func loadProjector(_ wanted: Config) throws {
        guard let path = wanted.mmprojPath, let model else { return }
        var params = mtmd_context_params_default()
        #if targetEnvironment(simulator)
        params.use_gpu = false
        #else
        params.use_gpu = wanted.gpuOffload
        #endif
        params.print_timings = false
        params.n_threads = Int32(max(1, min(6, ProcessInfo.processInfo.activeProcessorCount - 2)))
        guard let ctx = mtmd_init_from_file(path, model, params) else {
            throw LlamaEngineError.visionUnavailable
        }
        guard mtmd_support_vision(ctx) else {
            mtmd_free(ctx)
            throw LlamaEngineError.visionUnavailable
        }
        mtmdContext = ctx
    }

    private func resetCaches() {
        cachedTokens = []
        cachedChunks = []
        cachedChunkEnds = []
    }

    private func unload() {
        if let mtmdContext { mtmd_free(mtmdContext) }
        mtmdContext = nil
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        context = nil
        model = nil
        vocab = nil
        config = nil
        resetCaches()
        statusLock.lock(); _loadedPath = nil; statusLock.unlock()
        setPhase(.idle)
    }

    // MARK: Chat template (queue)

    private func applyTemplate(_ messages: [LlamaChatMessage]) throws -> String {
        guard let model else { throw LlamaEngineError.templateFailed }
        let template = llama_model_chat_template(model, nil)

        // Keep C strings alive for the duration of the call.
        var cStrings: [UnsafeMutablePointer<CChar>] = []
        defer { cStrings.forEach { free($0) } }
        var chat: [llama_chat_message] = []
        for m in messages {
            guard let role = strdup(m.role), let content = strdup(m.content) else { throw LlamaEngineError.templateFailed }
            cStrings.append(role)
            cStrings.append(content)
            chat.append(llama_chat_message(role: UnsafePointer(role), content: UnsafePointer(content)))
        }

        if let template {
            let totalChars = messages.reduce(0) { $0 + $1.content.utf8.count + $1.role.utf8.count }
            var capacity = totalChars * 2 + 4096
            for _ in 0..<2 {
                var buffer = [CChar](repeating: 0, count: capacity)
                let n = llama_chat_apply_template(template, chat, chat.count, true, &buffer, Int32(capacity))
                if n < 0 { break } // template not recognised by llama.cpp's built-in list
                if Int(n) <= capacity {
                    let bytes = buffer.prefix(Int(n)).map { UInt8(bitPattern: $0) }
                    return String(decoding: bytes, as: UTF8.self)
                }
                capacity = Int(n) + 1
            }
            logger.info("[Template] model template not supported by llama_chat_apply_template, falling back to ChatML")
        }
        return Self.chatML(messages)
    }

    private static func chatML(_ messages: [LlamaChatMessage]) -> String {
        var s = ""
        for m in messages {
            s += "<|im_start|>\(m.role)\n\(m.content)<|im_end|>\n"
        }
        s += "<|im_start|>assistant\n"
        return s
    }

    // MARK: Generation (queue)

    private func run(prompt: String, sampling: LlamaSampling, cancel: LlamaCancelFlag,
                     emit: (LlamaGenerationEvent) -> Void) throws {
        guard let context, let vocab else { throw LlamaEngineError.contextCreateFailed }
        let nCtx = Int(llama_n_ctx(context))

        // Templates that write the BOS text themselves must not get a second one.
        let bosText = tokenToPiece(llama_vocab_bos(vocab), special: true)
        let addSpecial = bosText.isEmpty || !prompt.hasPrefix(bosText)
        var tokens = tokenize(prompt, addSpecial: addSpecial)
        guard !tokens.isEmpty else { throw LlamaEngineError.templateFailed }
        if tokens.count >= nCtx - 16 {
            throw LlamaEngineError.promptTooLong(promptTokens: tokens.count, contextLength: nCtx)
        }
        let maxNew = max(1, min(sampling.maxTokens, nCtx - tokens.count - 1))

        // Coming from a multimodal turn: the KV cache holds image chunks the
        // token-level cache knows nothing about.
        if !cachedChunks.isEmpty {
            llama_memory_clear(llama_get_memory(context), true)
            resetCaches()
        }

        // Reuse the common prefix already in the KV cache.
        var reuse = 0
        let limit = min(cachedTokens.count, tokens.count)
        while reuse < limit && cachedTokens[reuse] == tokens[reuse] { reuse += 1 }
        if reuse == tokens.count { reuse -= 1 } // the last prompt token must be decoded for logits
        let memory = llama_get_memory(context)
        if reuse < cachedTokens.count {
            if !llama_memory_seq_rm(memory, 0, llama_pos(reuse), -1) {
                // Some architectures (recurrent / hybrid) cannot drop a tail.
                llama_memory_clear(memory, true)
                reuse = 0
            }
        }
        cachedTokens = Array(cachedTokens.prefix(reuse))

        // Evaluate the rest of the prompt in n_batch chunks.
        let batchSize = 512
        var index = reuse
        while index < tokens.count {
            if cancel.isCancelled { return }
            let count = min(batchSize, tokens.count - index)
            let rc: Int32 = resilientDecode(cancel: cancel) {
                tokens.withUnsafeMutableBufferPointer { buf in
                    llama_decode(context, llama_batch_get_one(buf.baseAddress! + index, Int32(count)))
                }
            }
            if rc != 0 {
                llama_memory_clear(memory, true)
                cachedTokens = []
                throw LlamaEngineError.decodeFailed(rc)
            }
            cachedTokens.append(contentsOf: tokens[index..<(index + count)])
            index += count
        }
        emit(.promptEvaluated(promptTokens: tokens.count, reused: reuse))
        tokens.removeAll()

        try sampleLoop(sampling: sampling, maxNew: maxNew, cancel: cancel, emit: emit) { token in
            var t = token
            let rc = self.resilientDecode(cancel: cancel) {
                withUnsafeMutablePointer(to: &t) { ptr in
                    llama_decode(context, llama_batch_get_one(ptr, 1))
                }
            }
            if rc == 0 { self.cachedTokens.append(token) }
            return rc
        }
    }

    // MARK: Multimodal generation (queue)

    private func runMultimodal(prompt: String, images: [Data], sampling: LlamaSampling, cancel: LlamaCancelFlag,
                               emit: (LlamaGenerationEvent) -> Void) throws {
        guard let context, let vocab else { throw LlamaEngineError.contextCreateFailed }
        guard let mtmd = mtmdContext else { throw LlamaEngineError.visionUnavailable }
        let nCtx = Int(llama_n_ctx(context))
        let memory = llama_get_memory(context)

        // Coming from a text-only turn: drop its token-level cache.
        if !cachedTokens.isEmpty {
            llama_memory_clear(memory, true)
            resetCaches()
        }

        // Decode the images into RGB bitmaps.
        var bitmaps: [OpaquePointer?] = []
        defer { bitmaps.forEach { if let b = $0 { mtmd_bitmap_free(b) } } }
        var imageKeys: [String] = []
        for data in images {
            guard let rgb = Self.rgbPixels(from: data) else { throw LlamaEngineError.imageDecodeFailed }
            let bitmap = rgb.pixels.withUnsafeBufferPointer { buf in
                mtmd_bitmap_init(UInt32(rgb.width), UInt32(rgb.height), buf.baseAddress)
            }
            guard let bitmap else { throw LlamaEngineError.imageDecodeFailed }
            bitmaps.append(bitmap)
            imageKeys.append(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        }

        // Split the prompt into text and image chunks.
        let chunks = mtmd_input_chunks_init()
        defer { mtmd_input_chunks_free(chunks) }
        let bosText = tokenToPiece(llama_vocab_bos(vocab), special: true)
        let addSpecial = bosText.isEmpty || !prompt.hasPrefix(bosText)
        let rc: Int32 = prompt.withCString { cstr in
            var input = mtmd_input_text(text: cstr, text_len: strlen(cstr), add_special: addSpecial, parse_special: true)
            return mtmd_tokenize(mtmd, chunks, &input, bitmaps, bitmaps.count)
        }
        guard rc == 0 else { throw LlamaEngineError.templateFailed }

        let totalPos = Int(mtmd_helper_get_n_pos(chunks))
        if totalPos >= nCtx - 16 {
            throw LlamaEngineError.promptTooLong(promptTokens: totalPos, contextLength: nCtx)
        }
        let maxNew = max(1, min(sampling.maxTokens, nCtx - totalPos - 1))

        // Describe each chunk so an unchanged prefix (system prompt, earlier
        // turns, already-seen images) is not evaluated again.
        let count = mtmd_input_chunks_size(chunks)
        var keys: [ChunkKey] = []
        var imageIndex = 0
        for i in 0..<count {
            let chunk = mtmd_input_chunks_get(chunks, i)
            if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT {
                var n = 0
                let ptr = mtmd_input_chunk_get_tokens_text(chunk, &n)
                keys.append(.text(ptr.map { Array(UnsafeBufferPointer(start: $0, count: n)) } ?? []))
            } else {
                keys.append(.image(imageIndex < imageKeys.count ? imageKeys[imageIndex] : "?"))
                imageIndex += 1
            }
        }

        var first = 0
        while first < min(keys.count, cachedChunks.count) && keys[first] == cachedChunks[first] { first += 1 }
        if first == keys.count { first = keys.count - 1 } // re-evaluate the last chunk for logits
        var nPast: llama_pos = first > 0 ? cachedChunkEnds[first - 1] : 0
        if first < cachedChunks.count || cachedChunks.isEmpty {
            if !llama_memory_seq_rm(memory, 0, nPast, -1) {
                llama_memory_clear(memory, true)
                first = 0
                nPast = 0
            }
        }
        cachedChunks = Array(cachedChunks.prefix(first))
        cachedChunkEnds = Array(cachedChunkEnds.prefix(first))

        for i in first..<count {
            if cancel.isCancelled { return }
            let chunk = mtmd_input_chunks_get(chunks, i)
            var newPast: llama_pos = 0
            let evalRC = resilientDecode(cancel: cancel) {
                mtmd_helper_eval_chunk_single(mtmd, context, chunk, nPast, 0, 512, i == count - 1, &newPast)
            }
            if evalRC != 0 {
                llama_memory_clear(memory, true)
                resetCaches()
                throw LlamaEngineError.decodeFailed(evalRC)
            }
            nPast = newPast
            cachedChunks.append(keys[i])
            cachedChunkEnds.append(nPast)
        }
        emit(.promptEvaluated(promptTokens: totalPos, reused: first > 0 ? Int(cachedChunkEnds[first - 1]) : 0))

        // Generated tokens are decoded at explicit positions (image chunks can
        // use fewer positions than tokens), and are not part of the chunk
        // cache: the next turn re-evaluates from the last cached chunk.
        var batch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(batch) }
        try sampleLoop(sampling: sampling, maxNew: maxNew, cancel: cancel, emit: emit) { token in
            batch.n_tokens = 1
            batch.token[0] = token
            batch.pos[0] = nPast
            batch.n_seq_id[0] = 1
            batch.seq_id[0]![0] = 0
            batch.logits[0] = 1
            let rc = self.resilientDecode(cancel: cancel) { llama_decode(context, batch) }
            if rc == 0 { nPast += 1 }
            return rc
        }
        // The generated tail sits after the last cached chunk; trim the cache
        // record so the next turn starts from a known position.
        if let last = cachedChunkEnds.last, nPast > last {
            _ = llama_memory_seq_rm(memory, 0, last, -1)
        }
    }

    // MARK: Sampling (queue)

    /// Samples up to `maxNew` tokens. `decode` feeds each sampled token back
    /// into the context and returns llama_decode's result.
    private func sampleLoop(sampling: LlamaSampling, maxNew: Int, cancel: LlamaCancelFlag,
                            emit: (LlamaGenerationEvent) -> Void,
                            decode: (llama_token) -> Int32) throws {
        guard let context, let vocab else { throw LlamaEngineError.contextCreateFailed }
        let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        defer { llama_sampler_free(sampler) }
        if sampling.temperature <= 0.01 {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(sampling.topK))
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(sampling.topP, 1))
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(sampling.minP, 1))
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(sampling.temperature))
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32.random(in: 0...UInt32.max)))
        }

        var pending: [UInt8] = []  // bytes of a not-yet-complete UTF-8 character
        var produced = 0
        var hitLimit = false
        while true {
            if cancel.isCancelled { break }
            if produced >= maxNew { hitLimit = true; break }
            let token = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, token) { break }
            produced += 1
            pending.append(contentsOf: tokenBytes(token))
            let text = Self.takeCompleteUTF8(&pending)
            if !text.isEmpty { emit(.text(text)) }
            let rc = decode(token)
            if rc != 0 {
                llama_memory_clear(llama_get_memory(context), true)
                resetCaches()
                throw LlamaEngineError.decodeFailed(rc)
            }
        }
        if !pending.isEmpty { emit(.text(String(decoding: pending, as: UTF8.self))) }
        emit(.finished(completionTokens: produced, hitLengthLimit: hitLimit))
    }

    // MARK: Images

    /// Decodes an image file into tightly packed RGB bytes, downscaled so the
    /// longest side is at most `maxSide` (the projector resizes anyway).
    static func rgbPixels(from data: Data, maxSide: Int = 1024) -> (pixels: [UInt8], width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn: Bool = rgba.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return (rgb, width, height)
    }

    // MARK: Tokens (queue)

    private func tokenize(_ text: String, addSpecial: Bool) -> [llama_token] {
        guard let vocab else { return [] }
        let byteCount = Int32(text.utf8.count)
        var capacity = Int32(byteCount) + 16
        var tokens = [llama_token](repeating: 0, count: Int(capacity))
        var n = llama_tokenize(vocab, text, byteCount, &tokens, capacity, addSpecial, true)
        if n < 0 && n != Int32.min {
            capacity = -n
            tokens = [llama_token](repeating: 0, count: Int(capacity))
            n = llama_tokenize(vocab, text, byteCount, &tokens, capacity, addSpecial, true)
        }
        return n > 0 ? Array(tokens.prefix(Int(n))) : []
    }

    private func tokenBytes(_ token: llama_token) -> [UInt8] {
        guard let vocab else { return [] }
        var capacity: Int32 = 64
        var buffer = [CChar](repeating: 0, count: Int(capacity))
        var n = llama_token_to_piece(vocab, token, &buffer, capacity, 0, false)
        if n < 0 {
            capacity = -n
            buffer = [CChar](repeating: 0, count: Int(capacity))
            n = llama_token_to_piece(vocab, token, &buffer, capacity, 0, false)
        }
        guard n > 0 else { return [] }
        return buffer.prefix(Int(n)).map { UInt8(bitPattern: $0) }
    }

    private func tokenToPiece(_ token: llama_token, special: Bool) -> String {
        guard let vocab, token >= 0 else { return "" }
        let capacity: Int32 = 64
        var buffer = [CChar](repeating: 0, count: Int(capacity))
        let n = llama_token_to_piece(vocab, token, &buffer, capacity, 0, special)
        guard n > 0 else { return "" }
        return String(decoding: buffer.prefix(Int(n)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Removes and returns the longest prefix of `bytes` made of complete
    /// UTF-8 characters, leaving a trailing partial character in place.
    static func takeCompleteUTF8(_ bytes: inout [UInt8]) -> String {
        guard !bytes.isEmpty else { return "" }
        var end = bytes.count
        // Walk back over at most 3 continuation bytes to find a lead byte.
        var i = bytes.count - 1
        var back = 0
        while i >= 0 && back < 4 && (bytes[i] & 0xC0) == 0x80 { i -= 1; back += 1 }
        if i >= 0 {
            let lead = bytes[i]
            let needed: Int
            if lead & 0x80 == 0 { needed = 1 }
            else if lead & 0xE0 == 0xC0 { needed = 2 }
            else if lead & 0xF0 == 0xE0 { needed = 3 }
            else if lead & 0xF8 == 0xF0 { needed = 4 }
            else { needed = 1 } // invalid lead byte: let the decoder replace it
            if bytes.count - i < needed { end = i }
        }
        let text = String(decoding: bytes[0..<end], as: UTF8.self)
        bytes.removeFirst(end)
        return text
    }
}

/// Cancellation flag shared between the async consumer and the engine queue.
final class LlamaCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    /// Cancels and returns true only for the first caller.
    func claim() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if cancelled { return false }
        cancelled = true
        return true
    }
}
