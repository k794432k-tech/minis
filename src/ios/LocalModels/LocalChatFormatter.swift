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
                       vision: Vision?) -> Prompt {
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
        let messages = build(openAIMessages, tools: tools, includeTools: includeTools, content: content)
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

    private static func build(_ openAIMessages: [[String: Any]], tools: [[String: Any]], includeTools: Bool,
                              content: (Any?) -> String) -> [LlamaChatMessage] {
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

/// Splits generated text into plain content and `<tool_call>` blocks while it
/// streams, holding back only as many characters as could be the start of an
/// opening tag.
struct ToolCallStreamParser {
    enum Output {
        case text(String)
        case toolCall(name: String, arguments: String)
    }

    private static let open = "<tool_call>"
    private static let close = "</tool_call>"

    private var buffer = ""
    private var insideCall = false
    private var callBody = ""
    private(set) var sawToolCall = false

    mutating func feed(_ chunk: String) -> [Output] {
        buffer += chunk
        var outputs: [Output] = []
        while true {
            if insideCall {
                if let r = buffer.range(of: Self.close) {
                    callBody += buffer[..<r.lowerBound]
                    buffer = String(buffer[r.upperBound...])
                    insideCall = false
                    outputs.append(contentsOf: finishCall())
                    continue
                }
                // Keep a possible partial closing tag in the buffer.
                let keep = Self.partialSuffixLength(buffer, of: Self.close)
                callBody += buffer.dropLast(keep)
                buffer = String(buffer.suffix(keep))
                return outputs
            } else {
                if let r = buffer.range(of: Self.open) {
                    let before = String(buffer[..<r.lowerBound])
                    if !before.isEmpty { outputs.append(.text(before)) }
                    buffer = String(buffer[r.upperBound...])
                    insideCall = true
                    callBody = ""
                    continue
                }
                let keep = Self.partialSuffixLength(buffer, of: Self.open)
                let emit = String(buffer.dropLast(keep))
                if !emit.isEmpty { outputs.append(.text(emit)) }
                buffer = String(buffer.suffix(keep))
                return outputs
            }
        }
    }

    /// Flushes whatever is left at the end of generation.
    mutating func finish() -> [Output] {
        var outputs: [Output] = []
        if insideCall {
            callBody += buffer
            buffer = ""
            insideCall = false
            outputs.append(contentsOf: finishCall())
        } else if !buffer.isEmpty {
            outputs.append(.text(buffer))
            buffer = ""
        }
        return outputs
    }

    private mutating func finishCall() -> [Output] {
        let body = callBody.trimmingCharacters(in: .whitespacesAndNewlines)
        callBody = ""
        guard let parsed = Self.parseCall(body) else {
            // Not valid JSON: show it as text rather than losing it.
            return [.text("<tool_call>\n\(body)\n</tool_call>")]
        }
        sawToolCall = true
        return [.toolCall(name: parsed.name, arguments: parsed.arguments)]
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
