//
//  LocalModelHub.swift
//  Minis — on-device models
//
//  Search and file-listing clients for Hugging Face and ModelScope, limited to
//  what the download page needs: find GGUF repositories, list their GGUF
//  files (grouping split files into one option), and build download URLs.
//
//  Endpoints used:
//    Hugging Face  GET {endpoint}/api/models?search=…&filter=gguf&sort=downloads&direction=-1
//                  GET {endpoint}/api/models/{repo}/tree/main?recursive=true
//                  GET {endpoint}/{repo}/resolve/main/{path}
//    ModelScope    GET https://modelscope.cn/openapi/v1/models?search=…&page_size=…&page_number=1
//                  GET https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true
//                  GET https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={path}
//
//  {endpoint} is huggingface.co by default and can be switched to the
//  hf-mirror.com mirror in settings (it serves the same paths).
//

import Foundation

struct HubRepo: Identifiable, Hashable, Sendable {
    let id: String
    let source: LocalModelSource
    let downloads: Int?
    let likes: Int?
    let summary: String?
}

struct HubFile: Hashable, Sendable {
    let path: String
    let size: Int64
}

/// One downloadable model choice inside a repo: a single GGUF file or every
/// part of a split GGUF.
struct GGUFOption: Identifiable, Hashable, Sendable {
    let parts: [HubFile]

    var id: String { parts.first?.path ?? "" }
    var primaryPath: String { parts.first?.path ?? "" }
    var totalSize: Int64 { parts.reduce(0) { $0 + $1.size } }
    var isSplit: Bool { parts.count > 1 }
    var quantization: String? { LocalModelFormat.quantization(of: primaryPath) }

    var displayName: String {
        var name = (primaryPath as NSString).lastPathComponent
        if let r = name.range(of: #"-\d{5}-of-\d{5}(?=\.gguf$)"#, options: [.regularExpression, .caseInsensitive]) {
            name.removeSubrange(r)
        }
        return name
    }
}

enum LocalModelHubError: LocalizedError {
    case badURL
    case http(Int, String)
    case badResponse(String)

    var errorDescription: String? {
        switch self {
        case .badURL:
            return AppLocalized("Invalid request URL")
        case .http(let code, let body):
            if code == 401 || code == 403 {
                return AppLocalized("Access denied (HTTP \(code)). This repository may be gated or private.")
            }
            return "HTTP \(code): \(body)"
        case .badResponse(let what):
            return AppLocalized("Unexpected response: \(what)")
        }
    }
}

enum LocalModelHub {
    static let modelScopeHost = "https://modelscope.cn"

    private static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = false
        return URLSession(configuration: config)
    }()

    // MARK: Search

    static func search(_ query: String, source: LocalModelSource) async throws -> [HubRepo] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        switch source {
        case .huggingFace: return try await searchHuggingFace(q)
        case .modelScope: return try await searchModelScope(q)
        case .imported, .url: return []
        }
    }

    private static func searchHuggingFace(_ query: String) async throws -> [HubRepo] {
        var comps = URLComponents(string: LocalModelSettings.huggingFaceEndpoint + "/api/models")
        var items = [
            URLQueryItem(name: "filter", value: "gguf"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "50"),
        ]
        if !query.isEmpty { items.append(URLQueryItem(name: "search", value: query)) }
        comps?.queryItems = items
        guard let url = comps?.url else { throw LocalModelHubError.badURL }
        let json = try await getJSON(url)
        guard let array = json as? [[String: Any]] else {
            throw LocalModelHubError.badResponse("model list")
        }
        return array.compactMap { item in
            guard let id = (item["id"] as? String) ?? (item["modelId"] as? String) else { return nil }
            return HubRepo(id: id, source: .huggingFace,
                           downloads: item["downloads"] as? Int,
                           likes: item["likes"] as? Int,
                           summary: item["pipeline_tag"] as? String)
        }
    }

    private static func searchModelScope(_ query: String) async throws -> [HubRepo] {
        // ModelScope search is not filtered by file type, so bias the query
        // toward GGUF repos and filter the result list client-side.
        let effective = query.lowercased().contains("gguf") ? query : (query.isEmpty ? "GGUF" : "\(query) GGUF")
        var comps = URLComponents(string: modelScopeHost + "/openapi/v1/models")
        comps?.queryItems = [
            URLQueryItem(name: "search", value: effective),
            URLQueryItem(name: "page_size", value: "50"),
            URLQueryItem(name: "page_number", value: "1"),
        ]
        guard let url = comps?.url else { throw LocalModelHubError.badURL }
        let json = try await getJSON(url)
        guard let root = json as? [String: Any],
              let data = root["data"] as? [String: Any],
              let models = data["models"] as? [[String: Any]] else {
            throw LocalModelHubError.badResponse("model list")
        }
        let repos: [HubRepo] = models.compactMap { item in
            guard let id = item["id"] as? String else { return nil }
            let desc = (item["display_name"] as? String) ?? (item["description"] as? String)
            return HubRepo(id: id, source: .modelScope,
                           downloads: item["downloads"] as? Int,
                           likes: item["likes"] as? Int,
                           summary: desc)
        }
        let gguf = repos.filter { $0.id.lowercased().contains("gguf") }
        return gguf.isEmpty ? repos : gguf
    }

    // MARK: Files

    struct RepoContents: Sendable {
        /// Model files, smallest first.
        let options: [GGUFOption]
        /// Vision projectors (`mmproj-*.gguf`), smallest first. Downloading one
        /// next to the model lets it read images.
        let projectors: [HubFile]

        /// The projector to use by default: Q8_0 if offered (near-lossless,
        /// half the size of F16), otherwise the smallest.
        var preferredProjector: HubFile? {
            projectors.first { $0.path.lowercased().contains("q8_0") } ?? projectors.first
        }
    }

    static func repoContents(repo: String, source: LocalModelSource) async throws -> RepoContents {
        let files: [HubFile]
        switch source {
        case .huggingFace: files = try await huggingFaceFiles(repo)
        case .modelScope: files = try await modelScopeFiles(repo)
        case .imported, .url: files = []
        }
        let gguf = files.filter { $0.path.lowercased().hasSuffix(".gguf") }
        let isProjector: (HubFile) -> Bool = {
            (($0.path as NSString).lastPathComponent).lowercased().contains("mmproj")
        }
        let models = gguf.filter { !isProjector($0) }
        let projectors = gguf.filter(isProjector).sorted { $0.size < $1.size }
        return RepoContents(options: groupSplits(models).sorted { $0.totalSize < $1.totalSize },
                            projectors: projectors)
    }

    /// Groups `name-00001-of-00003.gguf` style parts into one option.
    static func groupSplits(_ files: [HubFile]) -> [GGUFOption] {
        let pattern = #"^(.*)-(\d{5})-of-(\d{5})\.gguf$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return files.map { GGUFOption(parts: [$0]) }
        }
        var singles: [GGUFOption] = []
        var groups: [String: [(Int, HubFile)]] = [:]
        var expected: [String: Int] = [:]
        for f in files {
            let ns = f.path as NSString
            if let m = regex.firstMatch(in: f.path, range: NSRange(location: 0, length: ns.length)) {
                let key = ns.substring(with: m.range(at: 1))
                let index = Int(ns.substring(with: m.range(at: 2))) ?? 0
                let total = Int(ns.substring(with: m.range(at: 3))) ?? 0
                groups[key, default: []].append((index, f))
                expected[key] = total
            } else {
                singles.append(GGUFOption(parts: [f]))
            }
        }
        for (key, parts) in groups {
            let sorted = parts.sorted { $0.0 < $1.0 }.map(\.1)
            // Only offer complete sets.
            if sorted.count == expected[key] { singles.append(GGUFOption(parts: sorted)) }
        }
        return singles
    }

    private static func huggingFaceFiles(_ repo: String) async throws -> [HubFile] {
        let base = LocalModelSettings.huggingFaceEndpoint
        guard let url = URL(string: "\(base)/api/models/\(encodePath(repo))/tree/main?recursive=true") else {
            throw LocalModelHubError.badURL
        }
        let json = try await getJSON(url)
        guard let array = json as? [[String: Any]] else {
            throw LocalModelHubError.badResponse("file list")
        }
        return array.compactMap { item in
            guard (item["type"] as? String) == "file", let path = item["path"] as? String else { return nil }
            let lfsSize = (item["lfs"] as? [String: Any])?["size"] as? NSNumber
            let size = lfsSize?.int64Value ?? (item["size"] as? NSNumber)?.int64Value ?? 0
            return HubFile(path: path, size: size)
        }
    }

    private static func modelScopeFiles(_ repo: String) async throws -> [HubFile] {
        guard let url = URL(string: "\(modelScopeHost)/api/v1/models/\(encodePath(repo))/repo/files?Revision=master&Recursive=true") else {
            throw LocalModelHubError.badURL
        }
        let json = try await getJSON(url)
        guard let root = json as? [String: Any],
              let data = root["Data"] as? [String: Any],
              let files = data["Files"] as? [[String: Any]] else {
            let message = ((json as? [String: Any])?["Message"] as? String) ?? "file list"
            throw LocalModelHubError.badResponse(message)
        }
        return files.compactMap { item in
            guard (item["Type"] as? String) == "blob", let path = item["Path"] as? String else { return nil }
            let size = (item["Size"] as? NSNumber)?.int64Value ?? 0
            return HubFile(path: path, size: size)
        }
    }

    // MARK: Download URLs

    static func downloadURL(repo: String, path: String, source: LocalModelSource) -> URL? {
        switch source {
        case .huggingFace:
            let base = LocalModelSettings.huggingFaceEndpoint
            return URL(string: "\(base)/\(encodePath(repo))/resolve/main/\(encodePath(path))?download=true")
        case .modelScope:
            var comps = URLComponents(string: "\(modelScopeHost)/api/v1/models/\(encodePath(repo))/repo")
            comps?.queryItems = [
                URLQueryItem(name: "Revision", value: "master"),
                URLQueryItem(name: "FilePath", value: path),
            ]
            return comps?.url
        case .imported:
            return nil
        case .url:
            // For link downloads the "path" is the full URL.
            return URL(string: path)
        }
    }

    /// Normalizes a pasted link to a direct file URL: Hugging Face / hf-mirror
    /// "blob" pages become "resolve" downloads, ModelScope file pages become
    /// the file API.
    static func directDownloadURL(from raw: String) -> URL? {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        if !text.lowercased().hasPrefix("http") { text = "https://" + text }
        guard var comps = URLComponents(string: text), let host = comps.host?.lowercased() else { return nil }
        if host.hasSuffix("huggingface.co") || host.hasSuffix("hf-mirror.com") {
            comps.path = comps.path.replacingOccurrences(of: "/blob/", with: "/resolve/")
        }
        if host.hasSuffix("modelscope.cn"),
           let r = comps.path.range(of: #"^/models/([^/]+/[^/]+)/(?:file/view|resolve)/[^/]+/(.+)$"#, options: .regularExpression) {
            let parts = comps.path[r].split(separator: "/", omittingEmptySubsequences: true).map(String.init)
            // /models/{owner}/{name}/file/view/{rev}/{path…}  or  /models/{owner}/{name}/resolve/{rev}/{path…}
            let start = parts.count > 3 && parts[3] == "file" ? 6 : 5
            if parts.count > start {
                let repo = parts[1] + "/" + parts[2]
                let path = parts[start...].joined(separator: "/")
                return downloadURL(repo: repo, path: path, source: .modelScope)
            }
        }
        return comps.url
    }

    /// Web page for a repo, for "open in browser".
    static func webURL(repo: String, source: LocalModelSource) -> URL? {
        switch source {
        case .huggingFace: return URL(string: "\(LocalModelSettings.huggingFaceEndpoint)/\(encodePath(repo))")
        case .modelScope: return URL(string: "\(modelScopeHost)/models/\(encodePath(repo))")
        case .imported, .url: return nil
        }
    }

    // MARK: Helpers

    /// Percent-encodes each path component but keeps the slashes.
    static func encodePath(_ path: String) -> String {
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove(charactersIn: "?#")
        return path.split(separator: "/", omittingEmptySubsequences: false)
            .map { String($0).addingPercentEncoding(withAllowedCharacters: allowed) ?? String($0) }
            .joined(separator: "/")
    }

    private static func getJSON(_ url: URL) async throws -> Any {
        var request = URLRequest(url: url)
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Minis-LocalModels/1.0", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await session.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? -1
        guard (200..<300).contains(code) else {
            let body = String(data: data.prefix(300), encoding: .utf8) ?? ""
            throw LocalModelHubError.http(code, body)
        }
        do {
            return try JSONSerialization.jsonObject(with: data)
        } catch {
            throw LocalModelHubError.badResponse(String(data: data.prefix(120), encoding: .utf8) ?? "non-JSON body")
        }
    }
}
