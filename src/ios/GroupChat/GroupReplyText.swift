//
//  GroupReplyText.swift
//  Minis — AI group chat
//
//  Text clean-up for members' replies: separating thinking from the answer,
//  and catching replies that loop or start speaking for other members.
//

import Foundation

/// Splits a model's raw output into its thinking and the visible reply.
enum GroupThinking {
    struct Parts {
        var thinking: String
        var visible: String
        /// Thinking has started but not finished yet (still streaming).
        var isThinking: Bool
    }

    /// Opening marker → closing marker. Covers <think> (Qwen, DeepSeek…),
    /// <thinking>, <reasoning>, Gemma's thought channel and gpt-oss's
    /// analysis channel.
    private static let pairs: [(open: String, close: [String])] = [
        ("<think>", ["</think>"]),
        ("<thinking>", ["</thinking>"]),
        ("<reasoning>", ["</reasoning>"]),
        ("<|channel>thought", ["<channel|>"]),
        ("<|channel|>analysis<|message|>", ["<|end|>", "<|channel|>final<|message|>"]),
    ]
    private static let finalMarkers = ["<|start|>assistant<|channel|>final<|message|>", "<|channel|>final<|message|>"]

    static func split(_ raw: String) -> Parts {
        var thinking: [String] = []
        var visible = ""
        var rest = Substring(raw)
        var open = false

        // A template that already opened <think> in the prompt: the output
        // starts mid-thought and only the closing tag appears.
        if !raw.contains("<think>"), let close = raw.range(of: "</think>") {
            thinking.append(String(raw[..<close.lowerBound]))
            rest = raw[close.upperBound...]
        }
        // gpt-oss with the special tokens stripped: "analysis…assistantfinal…"
        if rest.hasPrefix("analysis"), let f = rest.range(of: "assistantfinal") {
            thinking.append(String(rest[rest.index(rest.startIndex, offsetBy: 8)..<f.lowerBound]))
            rest = rest[f.upperBound...]
        }

        while !rest.isEmpty {
            // Earliest opening marker.
            var best: (range: Range<Substring.Index>, close: [String])?
            for pair in pairs {
                if let r = rest.range(of: pair.open), best == nil || r.lowerBound < best!.range.lowerBound {
                    best = (r, pair.close)
                }
            }
            guard let found = best else {
                visible += rest
                break
            }
            visible += rest[..<found.range.lowerBound]
            let after = rest[found.range.upperBound...]
            var closeRange: Range<Substring.Index>?
            for c in found.close {
                if let r = after.range(of: c), closeRange == nil || r.lowerBound < closeRange!.lowerBound {
                    closeRange = r
                }
            }
            if let closeRange {
                thinking.append(String(after[..<closeRange.lowerBound]))
                rest = after[closeRange.upperBound...]
            } else {
                thinking.append(String(after))
                open = true
                rest = ""
            }
        }
        for marker in finalMarkers {
            visible = visible.replacingOccurrences(of: marker, with: "")
        }
        // Leftover special tokens.
        visible = visible.replacingOccurrences(of: #"<\|[a-z_]+\|>"#, with: "", options: .regularExpression)
        let joined = thinking.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        return Parts(thinking: joined,
                     visible: visible.trimmingCharacters(in: .whitespacesAndNewlines),
                     isThinking: open)
    }
}

enum GroupReplyCleaner {
    /// Drops a leading "[Me]:" / "Me:" and cuts the reply where the model
    /// starts writing lines for other members.
    static func clean(_ text: String, speaker: String, others: [String]) -> String {
        var t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["[\(speaker)]:", "[\(speaker)]：", "\(speaker):", "\(speaker)："] where t.hasPrefix(prefix) {
            t = String(t.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let cut = firstForeignLine(t, others: others + ["User", "You", "用户", "你"]) {
            t = String(t[..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return t
    }

    static func speaksForOthers(_ text: String, others: [String]) -> Bool {
        firstForeignLine(text, others: others + ["User"]) != nil
    }

    /// Start of the first line that begins "[Other]:" or "Other:".
    private static func firstForeignLine(_ text: String, others: [String]) -> String.Index? {
        let names = others.filter { !$0.isEmpty }
        guard !names.isEmpty else { return nil }
        var index = text.startIndex
        var first = true
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if !first || trimmed.hasPrefix("[") {
                for name in names {
                    if trimmed.hasPrefix("[\(name)]") || trimmed.hasPrefix("\(name):") || trimmed.hasPrefix("\(name)：") {
                        // A reply that opens by addressing someone ("Ana: good
                        // point") on its first line is fine unless bracketed.
                        if first && !trimmed.hasPrefix("[") { continue }
                        return index == text.startIndex && first ? nil : index
                    }
                }
            }
            first = false
            index = text.index(index, offsetBy: line.count)
            if index < text.endIndex { index = text.index(after: index) }
        }
        return nil
    }

    /// True when the end of the text is one chunk repeated three times or more.
    static func isLooping(_ text: String) -> Bool {
        let tail = Array(text.suffix(900))
        let n = tail.count
        guard n >= 60 else { return false }
        for period in 12...min(300, n / 3) {
            var repeats = true
            for k in 0..<(period * 2) where tail[n - 1 - k] != tail[n - 1 - k - period] {
                repeats = false
                break
            }
            if repeats { return true }
        }
        // The same line over and over (with small differences in spacing).
        let lines = text.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { $0.count >= 8 }
        if lines.count >= 4 {
            let last = lines.suffix(4)
            if Set(last).count == 1 { return true }
        }
        return false
    }

    /// Keeps one copy of a passage the model repeated back to back, and drops
    /// paragraphs that are exact repeats of an earlier paragraph.
    static func collapseRepetition(_ text: String) -> String {
        var chars = Array(text)
        var changed = true
        while changed {
            changed = false
            let n = chars.count
            guard n >= 40 else { break }
            for period in stride(from: min(400, n / 2), through: 12, by: -1) {
                var same = true
                for k in 0..<period where chars[n - 1 - k] != chars[n - 1 - k - period] {
                    same = false
                    break
                }
                if same {
                    chars.removeLast(period)
                    changed = true
                    break
                }
            }
        }
        var seen = Set<String>()
        var paragraphs: [String] = []
        for p in String(chars).components(separatedBy: "\n\n") {
            let key = normalized(p)
            if key.count >= 12 {
                if seen.contains(key) { continue }
                seen.insert(key)
            }
            paragraphs.append(p)
        }
        return paragraphs.joined(separator: "\n\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func normalized(_ text: String) -> String {
        text.lowercased().filter { !$0.isWhitespace && !$0.isPunctuation }
    }
}
