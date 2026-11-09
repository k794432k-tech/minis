//
//  LocalLLMServer.swift
//  Minis — on-device models
//
//  A minimal OpenAI-compatible HTTP server bound to the loopback interface.
//  The rest of the app talks to on-device models exactly as it talks to any
//  OpenAI-compatible vendor: a provider instance whose base URL is
//  http://127.0.0.1:<port> (see LocalModelProviderRegistrar). That keeps the
//  chat UI, the agent loop, tool calling, usage accounting and model pickers
//  unchanged.
//
//  Routes:
//    GET  /v1/models
//    POST /v1/chat/completions   (stream and non-stream)
//
//  Every request must carry `Authorization: Bearer <token>`, a random token
//  generated on this device, so other apps cannot use the loopback port.
//

import Foundation
import Network
import UIKit

private let logger = AppLogger(category: "LocalLLMServer")

final class LocalLLMServer: @unchecked Sendable {
    static let shared = LocalLLMServer()
    static let port: UInt16 = 47811
    static var baseURL: String { "http://127.0.0.1:\(port)" }

    private let queue = DispatchQueue(label: "com.openminis.app.localllm.server")
    private var listener: NWListener?
    private var observers: [NSObjectProtocol] = []

    private let stateLock = NSLock()
    private var _isRunning = false
    var isRunning: Bool { stateLock.lock(); defer { stateLock.unlock() }; return _isRunning }

    private init() {}

    /// Starts listening (idempotent). Also restarts after the app returns to
    /// the foreground, since iOS may tear sockets down while suspended.
    func start() {
        queue.async { self.startListener() }
        if observers.isEmpty {
            let center = NotificationCenter.default
            // While suspended iOS can reclaim the listening socket and the
            // listener keeps reporting "ready" although nothing is accepted
            // any more. Close it on the way out and open a fresh one on the
            // way back, then check that it really answers.
            observers.append(center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil) { [weak self] _ in
                self?.queue.async {
                    self?.listener?.cancel()
                    self?.listener = nil
                    self?.setRunning(false)
                    self?.setHealthy(false)
                }
            })
            observers.append(center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: nil) { [weak self] _ in
                self?.restart()
            })
            observers.append(center.addObserver(forName: UIApplication.didBecomeActiveNotification, object: nil, queue: nil) { [weak self] _ in
                self?.queue.async { self?.startListener() }
                self?.queue.asyncAfter(deadline: .now() + 1) { self?.checkHealth() }
            })
        }
    }

    /// Throws the listener away and opens a new one.
    func restart() {
        queue.async {
            self.listener?.cancel()
            self.listener = nil
            self.setRunning(false)
            self.setHealthy(false)
            self.startListener()
        }
    }

    // MARK: Health

    private var _healthy = false
    private var _lastCheck: Date?
    /// Whether the last self-test got an HTTP answer from the port.
    var isHealthy: Bool { stateLock.lock(); defer { stateLock.unlock() }; return _healthy }
    var lastHealthCheck: Date? { stateLock.lock(); defer { stateLock.unlock() }; return _lastCheck }
    private func setHealthy(_ value: Bool) {
        stateLock.lock(); _healthy = value; _lastCheck = Date(); stateLock.unlock()
    }

    /// Connects to the port like a client would. If nothing answers while the
    /// app is in the foreground, the listener is rebuilt.
    func checkHealth(repair: Bool = true, completion: (@Sendable (Bool) -> Void)? = nil) {
        guard let port = NWEndpoint.Port(rawValue: Self.port) else { return }
        let conn = NWConnection(host: "127.0.0.1", port: port, using: .tcp)
        let done = LlamaCancelFlag()
        let finish: @Sendable (Bool) -> Void = { [weak self] ok in
            guard done.claim() else { return }
            conn.cancel()
            self?.setHealthy(ok)
            if !ok && repair {
                logger.error("[Server] health check failed — restarting the listener")
                DispatchQueue.main.async {
                    if UIApplication.shared.applicationState != .background { self?.restart() }
                }
            }
            completion?(ok)
        }
        conn.stateUpdateHandler = { state in
            switch state {
            case .ready:
                let req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
                conn.send(content: Data(req.utf8), completion: .contentProcessed { error in
                    if error != nil { finish(false); return }
                    conn.receive(minimumIncompleteLength: 1, maximumLength: 64) { data, _, _, _ in
                        finish((data?.isEmpty == false))
                    }
                })
            case .failed, .cancelled:
                finish(false)
            case .waiting:
                finish(false)
            default:
                break
            }
        }
        conn.start(queue: DispatchQueue.global(qos: .utility))
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) { finish(false) }
    }

    func stop() {
        queue.async {
            self.listener?.cancel()
            self.listener = nil
            self.setRunning(false)
        }
    }

    private func setRunning(_ value: Bool) {
        stateLock.lock(); _isRunning = value; stateLock.unlock()
    }

    private func startListener() {
        if let listener {
            switch listener.state {
            case .ready, .setup, .waiting: return
            default: break
            }
        }
        listener?.cancel()
        do {
            let params = NWParameters.tcp
            params.requiredInterfaceType = .loopback
            params.allowLocalEndpointReuse = true
            guard let port = NWEndpoint.Port(rawValue: Self.port) else { return }
            let l = try NWListener(using: params, on: port)
            l.newConnectionHandler = { [weak self] conn in self?.accept(conn) }
            l.stateUpdateHandler = { [weak self] state in
                switch state {
                case .ready:
                    self?.setRunning(true)
                    self?.setHealthy(true)
                    logger.info("[Server] listening on 127.0.0.1:\(Self.port)")
                case .failed(let error):
                    self?.setRunning(false)
                    logger.error("[Server] listener failed: \(error)")
                    self?.queue.asyncAfter(deadline: .now() + 2) { self?.startListener() }
                case .cancelled:
                    self?.setRunning(false)
                default:
                    break
                }
            }
            listener = l
            l.start(queue: queue)
        } catch {
            logger.error("[Server] could not create listener: \(error)")
        }
    }

    // MARK: Connections

    private func accept(_ connection: NWConnection) {
        connection.start(queue: queue)
        LocalHTTPRequestReader(connection: connection).read { [weak self] request in
            guard let self else { connection.cancel(); return }
            guard let request else {
                LocalHTTPResponder(connection: connection).sendJSON(status: 400, ["error": ["message": "Malformed request", "type": "invalid_request_error"]])
                return
            }
            self.route(request, connection: connection)
        }
    }

    private func route(_ request: LocalHTTPRequest, connection: NWConnection) {
        let responder = LocalHTTPResponder(connection: connection)
        let path = request.path.split(separator: "?").first.map(String.init) ?? request.path

        guard LocalModelProviderRegistrar.isAuthorized(request.headers["authorization"]) else {
            responder.sendJSON(status: 401, ["error": ["message": "Invalid API key for the on-device model server", "type": "invalid_request_error", "code": "invalid_api_key"]])
            return
        }

        switch (request.method, path) {
        case ("GET", "/v1/models"), ("GET", "/models"):
            responder.sendJSON(status: 200, modelsPayload())
        case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
            Task { await self.chatCompletions(request, responder: responder) }
        default:
            responder.sendJSON(status: 404, ["error": ["message": "Unknown route \(request.method) \(path)", "type": "invalid_request_error"]])
        }
    }

    private func modelsPayload() -> [String: Any] {
        let contextLength = LocalModelSettings.contextLength
        let maxOutput = LocalModelSettings.maxOutputTokens
        let data: [[String: Any]] = LocalModelCatalog.shared.models.map { m -> [String: Any] in
            [
                "id": m.id,
                "object": "model",
                "created": Int(m.downloadedAt.timeIntervalSince1970),
                "owned_by": "on-device",
                "name": m.displayName,
                "context_length": contextLength,
                "max_output_tokens": maxOutput,
                "input_modalities": m.supportsVision ? ["text", "image"] : ["text"],
                "output_modalities": ["text"],
            ]
        }
        return ["object": "list", "data": data]
    }

    // MARK: Chat completions

    private func chatCompletions(_ request: LocalHTTPRequest, responder: LocalHTTPResponder) async {
        guard let body = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
            responder.sendJSON(status: 400, ["error": ["message": "Request body is not valid JSON", "type": "invalid_request_error"]])
            return
        }
        let requestedModel = body["model"] as? String ?? ""
        let catalog = LocalModelCatalog.shared
        guard let model = catalog.model(id: requestedModel) ?? (requestedModel.isEmpty ? catalog.models.first : nil) else {
            responder.sendJSON(status: 404, ["error": ["message": "Model '\(requestedModel)' is not downloaded on this device. Open Settings → Local Models.", "type": "invalid_request_error", "code": "model_not_found"]])
            return
        }

        let stream = body["stream"] as? Bool ?? false
        let includeUsage = ((body["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false
        let tools = (body["tools"] as? [[String: Any]]) ?? []
        let compact = LocalModelSettings.compactPrompt
        let useTools = !compact && LocalModelSettings.toolCalling && !tools.isEmpty
            && (body["tool_choice"] as? String) != "none"
        let vision = model.supportsVision
            ? LocalChatFormatter.Vision(marker: LlamaEngine.mediaMarker, maxImages: 4)
            : nil
        let prompt = LocalChatFormatter.prompt(
            from: (body["messages"] as? [[String: Any]]) ?? [],
            tools: tools, includeTools: useTools, vision: vision, compactSystem: compact
        )

        let requestedMax = (body["max_completion_tokens"] as? Int) ?? (body["max_tokens"] as? Int)
        let cap = LocalModelSettings.maxOutputTokens
        let sampling = LlamaSampling(
            temperature: Float((body["temperature"] as? Double) ?? LocalModelSettings.temperature),
            topP: Float((body["top_p"] as? Double) ?? 0.95),
            topK: 40,
            minP: 0.05,
            maxTokens: min(requestedMax ?? cap, cap)
        )
        // The projector is only loaded for requests that carry images, so
        // text chats do not pay for its memory.
        let config = LlamaEngine.Config(
            path: model.primaryURL.path,
            contextLength: LocalModelSettings.contextLength,
            gpuOffload: LocalModelSettings.gpuOffload,
            mmprojPath: prompt.images.isEmpty ? nil : model.mmprojURL?.path
        )
        let stops = Self.stopStrings(body["stop"])

        let completionId = "chatcmpl-local-\(UUID().uuidString.prefix(12))"
        let created = Int(Date().timeIntervalSince1970)
        let events = LlamaEngine.shared.generate(messages: prompt.messages, images: prompt.images,
                                                 config: config, sampling: sampling)

        if stream {
            await streamResponse(events: events, responder: responder, id: completionId, created: created,
                                 modelId: model.id, parseTools: useTools, stops: stops, includeUsage: includeUsage)
        } else {
            await jsonResponse(events: events, responder: responder, id: completionId, created: created,
                               modelId: model.id, parseTools: useTools, stops: stops)
        }
    }

    private func streamResponse(events: AsyncThrowingStream<LlamaGenerationEvent, Error>, responder: LocalHTTPResponder,
                                id: String, created: Int, modelId: String, parseTools: Bool,
                                stops: [String], includeUsage: Bool) async {
        responder.beginEventStream()

        func chunk(delta: [String: Any], finish: String? = nil) -> [String: Any] {
            let finishReason: Any = finish.map { $0 as Any } ?? NSNull()
            let choice: [String: Any] = ["index": 0, "delta": delta, "finish_reason": finishReason]
            return [
                "id": id, "object": "chat.completion.chunk", "created": created, "model": modelId,
                "choices": [choice],
            ]
        }

        responder.sendEvent(chunk(delta: ["role": "assistant", "content": ""]))

        var parser = ToolCallStreamParser()
        var stopFilter = StopStringFilter(stops: stops)
        var toolIndex = 0
        var promptTokens = 0
        var completionTokens = 0
        var hitLimit = false

        func emit(_ outputs: [ToolCallStreamParser.Output]) {
            for output in outputs {
                switch output {
                case .text(let text):
                    if !text.isEmpty { responder.sendEvent(chunk(delta: ["content": text])) }
                case .toolCall(let name, let arguments):
                    guard parseTools else { continue }
                    let call: [String: Any] = [
                        "index": toolIndex,
                        "id": "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
                        "type": "function",
                        "function": ["name": name, "arguments": arguments],
                    ]
                    toolIndex += 1
                    responder.sendEvent(chunk(delta: ["tool_calls": [call]]))
                }
            }
        }

        do {
            generation: for try await event in events {
                if responder.isClosed { break }
                switch event {
                case .promptEvaluated(let p, _):
                    promptTokens = p
                case .text(let piece):
                    let (visible, stopped) = stopFilter.feed(piece)
                    // The parser also runs without tools so stray tool-call
                    // markup from small models is hidden instead of shown.
                    emit(parser.feed(visible))
                    if stopped { break generation }
                case .finished(let c, let limit):
                    completionTokens = c
                    hitLimit = limit
                }
            }
            let tail = stopFilter.finish()
            emit(parser.feed(tail)); emit(parser.finish())
            let finish = (parseTools && parser.sawToolCall) ? "tool_calls" : (hitLimit ? "length" : "stop")
            responder.sendEvent(chunk(delta: [:], finish: finish))
            if includeUsage {
                responder.sendEvent([
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": modelId,
                    "choices": [Any](),
                    "usage": ["prompt_tokens": promptTokens, "completion_tokens": completionTokens,
                              "total_tokens": promptTokens + completionTokens],
                ])
            }
            responder.endEventStream()
        } catch {
            logger.error("[Chat] generation failed: \(error.localizedDescription)")
            let errorBody: [String: Any] = ["message": error.localizedDescription, "type": "server_error", "code": 500]
            responder.sendEvent(["error": errorBody])
            responder.endEventStream()
        }
    }

    private func jsonResponse(events: AsyncThrowingStream<LlamaGenerationEvent, Error>, responder: LocalHTTPResponder,
                              id: String, created: Int, modelId: String, parseTools: Bool, stops: [String]) async {
        var parser = ToolCallStreamParser()
        var stopFilter = StopStringFilter(stops: stops)
        var text = ""
        var toolCalls: [[String: Any]] = []
        var promptTokens = 0
        var completionTokens = 0
        var hitLimit = false

        func collect(_ outputs: [ToolCallStreamParser.Output]) {
            for output in outputs {
                switch output {
                case .text(let t): text += t
                case .toolCall(let name, let arguments):
                    guard parseTools else { continue }
                    toolCalls.append([
                        "id": "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
                        "type": "function",
                        "function": ["name": name, "arguments": arguments],
                    ])
                }
            }
        }

        do {
            generation: for try await event in events {
                switch event {
                case .promptEvaluated(let p, _): promptTokens = p
                case .text(let piece):
                    let (visible, stopped) = stopFilter.feed(piece)
                    collect(parser.feed(visible))
                    if stopped { break generation }
                case .finished(let c, let limit):
                    completionTokens = c
                    hitLimit = limit
                }
            }
            let tail = stopFilter.finish()
            collect(parser.feed(tail)); collect(parser.finish())
        } catch {
            responder.sendJSON(status: 500, ["error": ["message": error.localizedDescription, "type": "server_error"]])
            return
        }

        var message: [String: Any] = ["role": "assistant", "content": text]
        if !toolCalls.isEmpty { message["tool_calls"] = toolCalls }
        let finish = !toolCalls.isEmpty ? "tool_calls" : (hitLimit ? "length" : "stop")
        responder.sendJSON(status: 200, [
            "id": id, "object": "chat.completion", "created": created, "model": modelId,
            "choices": [["index": 0, "message": message, "finish_reason": finish] as [String: Any]],
            "usage": ["prompt_tokens": promptTokens, "completion_tokens": completionTokens,
                      "total_tokens": promptTokens + completionTokens],
        ])
    }

    private static func stopStrings(_ raw: Any?) -> [String] {
        if let s = raw as? String, !s.isEmpty { return [s] }
        return ((raw as? [String]) ?? []).filter { !$0.isEmpty }
    }
}

// MARK: - Stop strings

/// Cuts generation at the first caller-supplied stop string, holding back a
/// possible partial match between chunks.
struct StopStringFilter {
    let stops: [String]
    private var buffer = ""

    init(stops: [String]) { self.stops = stops }

    mutating func feed(_ text: String) -> (visible: String, stopped: Bool) {
        guard !stops.isEmpty else { return (text, false) }
        buffer += text
        if let r = stops.compactMap({ buffer.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
            let visible = String(buffer[..<r.lowerBound])
            buffer = ""
            return (visible, true)
        }
        let keep = stops.map { ToolCallStreamParser.partialSuffixLength(buffer, of: $0) }.max() ?? 0
        let visible = String(buffer.dropLast(keep))
        buffer = String(buffer.suffix(keep))
        return (visible, false)
    }

    mutating func finish() -> String {
        defer { buffer = "" }
        return buffer
    }
}

// MARK: - Minimal HTTP/1.1

struct LocalHTTPRequest {
    let method: String
    let path: String
    /// Lower-cased header names.
    let headers: [String: String]
    let body: Data
}

/// Reads one request (headers + Content-Length body) from a connection.
private final class LocalHTTPRequestReader {
    private static let maxBodyBytes = 64 * 1024 * 1024
    private let connection: NWConnection
    private var data = Data()

    init(connection: NWConnection) { self.connection = connection }

    func read(_ completion: @escaping (LocalHTTPRequest?) -> Void) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [self] chunk, _, isComplete, error in
            if let chunk { data.append(chunk) }
            if let request = parse() {
                completion(request)
                return
            }
            if error != nil || isComplete || data.count > Self.maxBodyBytes + 65536 {
                completion(nil)
                return
            }
            read(completion)
        }
    }

    /// Returns the request once headers and the full body have arrived.
    private func parse() -> LocalHTTPRequest? {
        let separator = Data("\r\n\r\n".utf8)
        guard let headerEnd = data.range(of: separator) else { return nil }
        guard let head = String(data: data[data.startIndex..<headerEnd.lowerBound], encoding: .utf8) else { return nil }
        let lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.first?.split(separator: " ") ?? []
        guard requestLine.count >= 2 else { return nil }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        let length = Int(headers["content-length"] ?? "0") ?? 0
        guard length <= Self.maxBodyBytes else { return nil }
        let bodyStart = headerEnd.upperBound
        guard data.count - (bodyStart - data.startIndex) >= length else { return nil }
        let body = data[bodyStart..<(bodyStart + length)]
        return LocalHTTPRequest(method: String(requestLine[0]).uppercased(), path: String(requestLine[1]),
                           headers: headers, body: Data(body))
    }
}

/// Writes responses. Streaming responses use a close-delimited body
/// (`Connection: close`, no Content-Length), which every HTTP/1.1 client
/// accepts and keeps the server free of chunked encoding.
final class LocalHTTPResponder: @unchecked Sendable {
    private let connection: NWConnection
    private let lock = NSLock()
    private var closed = false

    init(connection: NWConnection) {
        self.connection = connection
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled: self?.markClosed()
            default: break
            }
        }
    }

    var isClosed: Bool { lock.lock(); defer { lock.unlock() }; return closed }

    private func markClosed() { lock.lock(); closed = true; lock.unlock() }

    func sendJSON(status: Int, _ object: [String: Any]) {
        let body = (try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes])) ?? Data()
        var head = "HTTP/1.1 \(status) \(Self.reason(status))\r\n"
        head += "Content-Type: application/json; charset=utf-8\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Connection: close\r\n\r\n"
        var payload = Data(head.utf8)
        payload.append(body)
        connection.send(content: payload, isComplete: true, completion: .contentProcessed { [weak self] _ in
            self?.close()
        })
    }

    func beginEventStream() {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        connection.send(content: Data(head.utf8), completion: .contentProcessed { _ in })
    }

    func sendEvent(_ object: [String: Any]) {
        guard !isClosed,
              let json = try? JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes]) else { return }
        var payload = Data("data: ".utf8)
        payload.append(json)
        payload.append(Data("\n\n".utf8))
        connection.send(content: payload, completion: .contentProcessed { _ in })
    }

    func endEventStream() {
        guard !isClosed else { return }
        connection.send(content: Data("data: [DONE]\n\n".utf8), isComplete: true, completion: .contentProcessed { [weak self] _ in
            self?.close()
        })
    }

    private func close() {
        markClosed()
        connection.cancel()
    }

    private static func reason(_ status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 404: return "Not Found"
        case 500: return "Internal Server Error"
        default: return "Status"
        }
    }
}
