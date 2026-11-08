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
import ImageIO
import llama

private let logger = AppLogger(category: "LlamaEngine")

enum LlamaEngineError: LocalizedError {
    case modelLoadFailed(String)
    case contextCreateFailed
    case promptTooLong(promptTokens: Int, contextLength: Int)
    case decodeFailed(Int32)
    case templateFailed
    case visionUnavailable
    case imageDecodeFailed

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

    private init() {}

    // MARK: Public API

    /// Runs a chat completion. Events arrive in order; cancelling the consuming
    /// task stops generation at the next token.
    /// `images` are encoded image files (JPEG/PNG/HEIC…) in the order their
    /// `mediaMarker` placeholders appear in the messages.
    func generate(messages: [LlamaChatMessage], images: [Data] = [], config: Config, sampling: LlamaSampling)
        -> AsyncThrowingStream<LlamaGenerationEvent, Error>
    {
        AsyncThrowingStream { continuation in
            let cancelFlag = LlamaCancelFlag()
            continuation.onTermination = { _ in cancelFlag.cancel() }
            queue.async {
                do {
                    try self.ensureLoaded(config)
                    let prompt = try self.applyTemplate(messages)
                    let emit: (LlamaGenerationEvent) -> Void = { continuation.yield($0) }
                    if images.isEmpty {
                        try self.run(prompt: prompt, sampling: sampling, cancel: cancelFlag, emit: emit)
                    } else {
                        try self.runMultimodal(prompt: prompt, images: images, sampling: sampling,
                                               cancel: cancelFlag, emit: emit)
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

    func unloadAll() {
        queue.async { self.unload() }
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
                try createContext(contextLength: wanted.contextLength)
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
        var mparams = llama_model_default_params()
        #if targetEnvironment(simulator)
        mparams.n_gpu_layers = 0
        #else
        mparams.n_gpu_layers = wanted.gpuOffload ? 999 : 0
        #endif
        let start = Date()
        guard let loaded = llama_model_load_from_file(wanted.path, mparams) else {
            throw LlamaEngineError.modelLoadFailed(wanted.path)
        }
        model = loaded
        vocab = llama_model_get_vocab(loaded)
        do {
            try createContext(contextLength: wanted.contextLength)
            try loadProjector(wanted)
        } catch {
            unload()
            throw error
        }
        config = wanted
        statusLock.lock(); _loadedPath = wanted.path; statusLock.unlock()
        logger.info("[Load] \((wanted.path as NSString).lastPathComponent) ctx=\(wanted.contextLength) gpu=\(wanted.gpuOffload) in \(String(format: "%.1f", Date().timeIntervalSince(start)))s")
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
            let rc: Int32 = tokens.withUnsafeMutableBufferPointer { buf in
                llama_decode(context, llama_batch_get_one(buf.baseAddress! + index, Int32(count)))
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
            let rc = withUnsafeMutablePointer(to: &t) { ptr in
                llama_decode(context, llama_batch_get_one(ptr, 1))
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
            let evalRC = mtmd_helper_eval_chunk_single(mtmd, context, chunk, nPast, 0, 512, i == count - 1, &newPast)
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
            let rc = llama_decode(context, batch)
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
}
