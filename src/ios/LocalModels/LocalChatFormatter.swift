//
//  LocalChatFormatter.swift
//  Minis — on-device models
//
//  Converts an OpenAI Chat Completions request into the plain role/content
//  list llama.cpp's built-in chat templates understand, and parses tool calls
//  back out of the generated text.
//
//  Tool calling uses the Hermes / Qwen convention, which most current open
//  models (Qwen 2.5/3, Hermes, many Llama and Mistral fine-tunes) are trained
//  on:
//    - tool signatures go into the system prompt inside <tools></tools>;
//    - the model answers with <tool_call>{"name": …, "arguments": {…}}</tool_call>;
//    - tool results come back as a user turn wrapped in <tool_response>.
//

import Foundation

enum LocalChatFormatter {

    struct Prompt {
        var messages: [LlamaChatMessage]
        /// Image files, in the order their markers appear in `messages`.
        var images: [Data]
    }

    /// How images in the request are handled.
    struct Vision {
        /// Placeholder llama.cpp's mtmd replaces with the image embedding.
        let marker: String
        /// Only the most recent images are kept; older ones become a note,
        /// since every image costs hundreds of context tokens.
        let maxImages: Int
    }

    /// Builds the message list for llama.cpp from OpenAI-format messages.
    /// With `vision`, image parts become markers and their bytes are returned
    /// in `images`; without it they become a short text note.
    static func prompt(from openAIMessages: [[String: Any]], tools: [[String: Any]], includeTools: Bool,
                       vision: Vision?, compactSystem: Bool = false) -> Prompt {
        // Pick which image parts survive: the last `maxImages` in the conversation.
        var keepImageOrdinals = Set<Int>()
        if let vision {
            var ordinals: [Int] = []
            var ordinal = 0
            for message in openAIMessages {
                guard let parts = message["content"] as? [[String: Any]] else { continue }
                for part in parts where isImagePart(part) {
                    if imageData(from: part) != nil { ordinals.append(ordinal) }
                    ordinal += 1
                }
            }
            keepImageOrdinals = Set(ordinals.suffix(vision.maxImages))
        }
        var images: [Data] = []
        var imageOrdinal = 0
        let content: (Any?) -> String = { raw in
            guard let vision, let parts = raw as? [[String: Any]] else { return textContent(raw) }
            var pieces: [String] = []
            for part in parts {
                if isImagePart(part) {
                    defer { imageOrdinal += 1 }
                    if keepImageOrdinals.contains(imageOrdinal), let data = imageData(from: part) {
                        images.append(data)
                        pieces.append(vision.marker)
                    } else {
                        pieces.append("[An earlier image was attached here.]")
                    }
                } else {
                    pieces.append(textContent([part]))
                }
            }
            return pieces.filter { !$0.isEmpty }.joined(separator: "\n")
        }
        let messages = build(openAIMessages, tools: tools, includeTools: includeTools, content: content,
                             compactSystem: compactSystem)
        return Prompt(messages: messages, images: images)
    }

    /// Text-only form (images become a note).
    static func messages(from openAIMessages: [[String: Any]], tools: [[String: Any]], includeTools: Bool) -> [LlamaChatMessage] {
        build(openAIMessages, tools: tools, includeTools: includeTools, content: textContent)
    }

    static func isImagePart(_ part: [String: Any]) -> Bool {
        let type = part["type"] as? String ?? ""
        return type == "image_url" || type == "input_image" || type == "image"
    }

    /// Bytes of an inline (data: URL / base64) image part. Remote URLs are
    /// not fetched.
    static func imageData(from part: [String: Any]) -> Data? {
        var urlString: String?
        if let obj = part["image_url"] as? [String: Any] { urlString = obj["url"] as? String }
        else if let s = part["image_url"] as? String { urlString = s }
        else if let s = part["url"] as? String { urlString = s }
        if let s = urlString, s.hasPrefix("data:"), let comma = s.firstIndex(of: ",") {
            return Data(base64Encoded: String(s[s.index(after: comma)...]), options: .ignoreUnknownCharacters)
        }
        // Anthropic-style {"type":"image","source":{"type":"base64","data":…}}
        if let source = part["source"] as? [String: Any], let b64 = source["data"] as? String {
            return Data(base64Encoded: b64, options: .ignoreUnknownCharacters)
        }
        return nil
    }

    /// Where the app's system prompt moves from identity (name, soul,
    /// persona, memories) to agent and tool instructions.
    private static let agentRulesMarker = "You should proactively use shell commands"

    /// A short system prompt for small local models: the identity part of
    /// the app's prompt (so names, souls and personas still apply) plus a few
    /// plain instructions, without the tool and workflow rules.
    static func compactSystemPrompt(from full: String) -> String {
        var identity = full
        if let r = full.range(of: agentRulesMarker) {
            identity = String(full[..<r.lowerBound])
        }
        // Drop the SOUL.md editing hint, which refers to tools.
        if let r = identity.range(of: "\n---\nSOUL.md fields") {
            identity = String(identity[..<r.lowerBound])
        }
        identity = identity
            .replacingOccurrences(of: "with a fully functional iSH Linux shell (Alpine Linux, aarch64)", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if identity.count > 4000 { identity = String(identity.prefix(4000)) }
        let rules = """
        You are running fully offline on the user's iPhone as a small on-device model.
        Answer the user directly in plain text, in the language they use. Be concise and helpful.
        You have no tools in this mode: never output XML, function-call or tool-call markup; just reply.
        """
        return identity.isEmpty ? rules : identity + "\n\n" + rules
    }

    private static func build(_ openAIMessages: [[String: Any]], tools: [[String: Any]], includeTools: Bool,
                              content: (Any?) -> String, compactSystem: Bool = false) -> [LlamaChatMessage] {
        var out: [LlamaChatMessage] = []
        var systemParts: [String] = []
        var pendingToolResponses: [String] = []

        func flushToolResponses() {
            guard !pendingToolResponses.isEmpty else { return }
            let body = pendingToolResponses
                .map { "<tool_response>\n\($0)\n</tool_response>" }
                .joined(separator: "\n")
            out.append(LlamaChatMessage(role: "user", content: body))
            pendingToolResponses.removeAll()
        }

        for message in openAIMessages {
            let role = (message["role"] as? String) ?? "user"
            let text = content(message["content"])
            switch role {
            case "system", "developer":
                if !text.isEmpty { systemParts.append(text) }
            case "tool", "function":
                pendingToolResponses.append(text.isEmpty ? "(empty result)" : text)
            case "assistant":
                flushToolResponses()
                var content = stripThinking(text)
                if let calls = message["tool_calls"] as? [[String: Any]], !calls.isEmpty {
                    let rendered = calls.compactMap(renderToolCall).joined(separator: "\n")
                    content = content.isEmpty ? rendered : content + "\n" + rendered
                }
                out.append(LlamaChatMessage(role: "assistant", content: content))
            default:
                flushToolResponses()
                out.append(LlamaChatMessage(role: "user", content: text))
            }
        }
        flushToolResponses()

        if compactSystem {
            systemParts = [compactSystemPrompt(from: systemParts.joined(separator: "\n\n"))]
        }
        if includeTools && !tools.isEmpty {
            systemParts.append(toolsPrompt(tools))
        }
        if !systemParts.isEmpty {
            out.insert(LlamaChatMessage(role: "system", content: systemParts.joined(separator: "\n\n")), at: 0)
        }
        // Templates expect the conversation to end on a user turn.
        if out.last?.role == "assistant" {
            out.append(LlamaChatMessage(role: "user", content: "Continue."))
        }
        return out
    }

    /// Text of an OpenAI `content` value (string or array of parts). Images,
    /// audio and files are replaced with a short note: this runtime is text-only.
    static func textContent(_ content: Any?) -> String {
        if let s = content as? String { return s }
        guard let parts = content as? [[String: Any]] else { return "" }
        var pieces: [String] = []
        for part in parts {
            let type = part["type"] as? String ?? ""
            switch type {
            case "text", "input_text":
                if let t = part["text"] as? String { pieces.append(t) }
            case "image_url", "input_image", "image":
                pieces.append("[An image was attached, but this on-device model cannot see images.]")
            case "input_audio", "audio":
                pieces.append("[An audio clip was attached, but this on-device model cannot hear audio.]")
            case "file", "input_file":
                pieces.append("[A file was attached, but this on-device model cannot read attachments.]")
            default:
                if let t = part["text"] as? String { pieces.append(t) }
            }
        }
        return pieces.joined(separator: "\n")
    }

    /// Past reasoning is not fed back to the model; it only wastes context.
    static func stripThinking(_ text: String) -> String {
        guard text.contains("<think>") else { return text }
        let pattern = #"(?s)<think>.*?</think>\s*"#
        return text.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
    }

    private static func renderToolCall(_ call: [String: Any]) -> String? {
        guard let fn = call["function"] as? [String: Any], let name = fn["name"] as? String else { return nil }
        var arguments: Any = [String: Any]()
        if let raw = fn["arguments"] as? String, let data = raw.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) {
            arguments = parsed
        } else if let dict = fn["arguments"] as? [String: Any] {
            arguments = dict
        }
        let payload: [String: Any] = ["name": name, "arguments": arguments]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .withoutEscapingSlashes]),
              let json = String(data: data, encoding: .utf8) else { return nil }
        return "<tool_call>\n\(json)\n</tool_call>"
    }

    private static func toolsPrompt(_ tools: [[String: Any]]) -> String {
        var lines: [String] = []
        for tool in tools {
            guard let compact = compactTool(tool),
                  let data = try? JSONSerialization.data(withJSONObject: compact, options: [.sortedKeys, .withoutEscapingSlashes]),
                  let json = String(data: data, encoding: .utf8) else { continue }
            lines.append(json)
        }
        return """
        # Tools

        You may call one or more functions to assist with the user query.

        You are provided with function signatures within <tools></tools> XML tags:
        <tools>
        \(lines.joined(separator: "\n"))
        </tools>

        For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        <tool_call>
        {"name": <function-name>, "arguments": <args-json-object>}
        </tool_call>
        """
    }

    /// Shortens long tool descriptions so a large tool list fits a small context.
    private static func compactTool(_ tool: [String: Any]) -> [String: Any]? {
        guard var fn = tool["function"] as? [String: Any] else { return nil }
        if let desc = fn["description"] as? String, desc.count > 400 {
            fn["description"] = String(desc.prefix(400)) + "…"
        }
        return ["type": "function", "function": fn]
    }
}

// MARK: - Streaming tool-call parser

/// Splits generated text into plain content and tool-call blocks while it
/// streams, holding back only as many characters as could be the start of an
/// opening tag. Besides the Hermes/Qwen `<tool_call>{json}</tool_call>`
/// format it accepts the other markups small models commonly emit:
///   <invoke name="x"><parameter name="k">v</parameter></invoke>   (optionally in <function_calls>)
///   <function=x><parameter=k>v</parameter></function>
/// Bare `<function_calls>` wrappers are dropped from the visible text.
struct ToolCallStreamParser {
    enum Output {
        case text(String)
        case toolCall(name: String, arguments: String)
    }

    private struct Tag {
        let open: String
        let close: String
    }

    private static let tags = [
        Tag(open: "<tool_call>", close: "</tool_call>"),
        Tag(open: "<invoke", close: "</invoke>"),
        Tag(open: "<function=", close: "</function>"),
    ]
    private static let dropped = ["<function_calls>", "</function_calls>"]

    private var buffer = ""
    private var current: Tag?
    private var callBody = ""
    private(set) var sawToolCall = false

    mutating func feed(_ chunk: String) -> [Output] {
        buffer += chunk
        for d in Self.dropped where buffer.contains(d) {
            buffer = buffer.replacingOccurrences(of: d, with: "")
        }
        var outputs: [Output] = []
        while true {
            if let tag = current {
                if let r = buffer.range(of: tag.close) {
                    callBody += buffer[..<r.lowerBound]
                    buffer = String(buffer[r.upperBound...])
                    current = nil
                    outputs.append(contentsOf: finishCall(tag))
                    continue
                }
                let keep = Self.partialSuffixLength(buffer, of: tag.close)
                callBody += buffer.dropLast(keep)
                buffer = String(buffer.suffix(keep))
                return outputs
            }
            // Earliest opening tag in the buffer.
            let found = Self.tags.compactMap { tag in buffer.range(of: tag.open).map { (tag, $0) } }
                .min { $0.1.lowerBound < $1.1.lowerBound }
            if let (tag, r) = found {
                let before = String(buffer[..<r.lowerBound])
                if !before.isEmpty { outputs.append(.text(before)) }
                buffer = String(buffer[r.upperBound...])
                current = tag
                callBody = ""
                continue
            }
            let keep = (Self.tags.map(\.open) + Self.dropped)
                .map { Self.partialSuffixLength(buffer, of: $0) }.max() ?? 0
            let emit = String(buffer.dropLast(keep))
            if !emit.isEmpty { outputs.append(.text(emit)) }
            buffer = String(buffer.suffix(keep))
            return outputs
        }
    }

    /// Flushes whatever is left at the end of generation.
    mutating func finish() -> [Output] {
        var outputs: [Output] = []
        if let tag = current {
            callBody += buffer
            buffer = ""
            current = nil
            outputs.append(contentsOf: finishCall(tag))
        } else if !buffer.isEmpty {
            outputs.append(.text(buffer))
            buffer = ""
        }
        return outputs
    }

    private mutating func finishCall(_ tag: Tag) -> [Output] {
        let body = callBody
        callBody = ""
        let parsed: (name: String, arguments: String)?
        switch tag.open {
        case "<invoke": parsed = Self.parseInvoke(body)
        case "<function=": parsed = Self.parseFunctionEquals(body)
        default: parsed = Self.parseCall(body.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard let parsed else {
            // Not parseable: drop the markup rather than show it to the user.
            return []
        }
        sawToolCall = true
        return [.toolCall(name: parsed.name, arguments: parsed.arguments)]
    }

    /// ` name="x"><parameter name="k">v</parameter>…`
    static func parseInvoke(_ body: String) -> (name: String, arguments: String)? {
        guard let name = firstMatch(#"^\s*name\s*=\s*"([^"]+)""#, in: body) else { return nil }
        let params = allMatches(#"(?s)<parameter\s+name\s*=\s*"([^"]+)"\s*>(.*?)</parameter>"#, in: body)
        return (name, encodeArguments(params))
    }

    /// `x>{json}` or `x><parameter=k>v</parameter>…`
    static func parseFunctionEquals(_ body: String) -> (name: String, arguments: String)? {
        guard let close = body.firstIndex(of: ">") else { return nil }
        let name = body[..<close].trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { return nil }
        let rest = String(body[body.index(after: close)...]).trimmingCharacters(in: .whitespacesAndNewlines)
        if rest.hasPrefix("{"), let data = rest.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data)) != nil {
            return (name, rest)
        }
        let params = allMatches(#"(?s)<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>"#, in: rest)
        return (name, encodeArguments(params))
    }

    private static func encodeArguments(_ params: [(String, String)]) -> String {
        var args: [String: Any] = [:]
        for (key, raw) in params {
            let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if let data = value.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]),
               !(json is String) {
                args[key] = json
            } else {
                args[key] = value
            }
        }
        guard let data = try? JSONSerialization.data(withJSONObject: args, options: [.withoutEscapingSlashes]),
              let json = String(data: data, encoding: .utf8) else { return "{}" }
        return json
    }

    private static func firstMatch(_ pattern: String, in text: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: pattern),
              let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              let r = Range(m.range(at: 1), in: text) else { return nil }
        return String(text[r])
    }

    private static func allMatches(_ pattern: String, in text: String) -> [(String, String)] {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return [] }
        return re.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap { m in
            guard let k = Range(m.range(at: 1), in: text), let v = Range(m.range(at: 2), in: text) else { return nil }
            return (String(text[k]), String(text[v]))
        }
    }

    static func parseCall(_ body: String) -> (name: String, arguments: String)? {
        var text = body
        // Some models wrap the JSON in a code fence.
        if text.hasPrefix("```") {
            text = text.replacingOccurrences(of: #"^```[a-zA-Z]*\s*"#, with: "", options: .regularExpression)
            text = text.replacingOccurrences(of: #"\s*```$"#, with: "", options: .regularExpression)
        }
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = (obj["name"] as? String) ?? ((obj["function"] as? [String: Any])?["name"] as? String)
        else { return nil }
        let rawArgs = obj["arguments"] ?? obj["parameters"] ?? (obj["function"] as? [String: Any])?["arguments"]
        let arguments: String
        if let s = rawArgs as? String {
            arguments = s
        } else if let a = rawArgs, JSONSerialization.isValidJSONObject(a),
                  let d = try? JSONSerialization.data(withJSONObject: a, options: [.withoutEscapingSlashes]),
                  let s = String(data: d, encoding: .utf8) {
            arguments = s
        } else {
            arguments = "{}"
        }
        return (name, arguments)
    }

    /// Length of the longest suffix of `text` that is a proper prefix of `tag`.
    static func partialSuffixLength(_ text: String, of tag: String) -> Int {
        let maxLen = min(text.count, tag.count - 1)
        guard maxLen > 0 else { return 0 }
        for len in stride(from: maxLen, through: 1, by: -1) where tag.hasPrefix(String(text.suffix(len))) {
            return len
        }
        return 0
    }
}
