//
//  LocalModelBrowserView.swift
//  Minis — on-device models
//
//  Search Hugging Face or ModelScope for GGUF repositories, then pick a
//  quantization to download.
//

import SwiftUI

struct LocalModelBrowserView: View {
    @AppStorage("localLLM.browserSource") private var sourceRaw = LocalModelSource.huggingFace.rawValue
    @State private var query = ""
    @State private var results: [HubRepo] = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var searchTask: Task<Void, Never>?

    private var source: LocalModelSource {
        LocalModelSource(rawValue: sourceRaw) ?? .huggingFace
    }

    private static let suggestions = ["Qwen3", "Qwen2.5", "Llama 3.2", "Gemma 3", "Phi-4", "DeepSeek R1 Distill", "MiniCPM", "SmolLM"]

    var body: some View {
        List {
            Section {
                Picker("Source", selection: $sourceRaw) {
                    ForEach(LocalModelSource.hubs) { s in
                        Text(s.displayName).tag(s.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets())
            }

            if query.isEmpty && results.isEmpty && !isLoading {
                Section("Suggestions") {
                    ForEach(Self.suggestions, id: \.self) { s in
                        Button {
                            query = s
                            runSearch()
                        } label: {
                            Label(s, systemImage: "magnifyingglass")
                        }
                    }
                }
            }

            if let errorMessage {
                Section {
                    Text(errorMessage).foregroundStyle(.red).font(.callout)
                    if source == .huggingFace && LocalModelSettings.huggingFaceEndpoint == LocalModelSettings.defaultHuggingFaceEndpoint {
                        Button("Switch to hf-mirror.com") {
                            UserDefaults.standard.set(LocalModelSettings.huggingFaceMirrorEndpoint, forKey: LocalModelSettings.huggingFaceEndpointKey)
                            runSearch()
                        }
                    }
                }
            }

            if isLoading {
                Section {
                    HStack { Spacer(); ProgressView(); Spacer() }
                }
            } else if !results.isEmpty {
                Section {
                    ForEach(results) { repo in
                        NavigationLink {
                            LocalModelRepoView(repo: repo)
                        } label: {
                            HubRepoRow(repo: repo)
                        }
                    }
                } header: {
                    Text("\(results.count) repositories")
                }
            } else if !query.isEmpty && errorMessage == nil {
                Section {
                    Text("No GGUF repositories found.").foregroundStyle(.secondary)
                }
            }
        }
        .navigationTitle("Download Models")
        .navigationBarTitleDisplayMode(.inline)
        .searchable(text: $query, prompt: Text("Search models, e.g. Qwen3 1.7B"))
        .onSubmit(of: .search) { runSearch() }
        .onChange(of: sourceRaw) { _ in
            if !query.isEmpty || !results.isEmpty { runSearch() }
        }
        .onDisappear { searchTask?.cancel() }
    }

    private func runSearch() {
        searchTask?.cancel()
        let q = query
        let s = source
        isLoading = true
        errorMessage = nil
        searchTask = Task { @MainActor in
            do {
                let found = try await LocalModelHub.search(q, source: s)
                guard !Task.isCancelled else { return }
                results = found
            } catch {
                guard !Task.isCancelled else { return }
                results = []
                errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }
}

private struct HubRepoRow: View {
    let repo: HubRepo

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(repo.id)
                .font(.callout)
                .lineLimit(2)
            HStack(spacing: 10) {
                if let d = repo.downloads {
                    Label(d.formatted(.number.notation(.compactName)), systemImage: "arrow.down.circle")
                }
                if let l = repo.likes, l > 0 {
                    Label(l.formatted(.number.notation(.compactName)), systemImage: "heart")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .labelStyle(.titleAndIcon)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Repository files

struct LocalModelRepoView: View {
    let repo: HubRepo
    @ObservedObject private var downloads = LocalModelDownloadManager.shared
    @State private var options: [GGUFOption] = []
    @State private var projector: HubFile?
    @State private var includeVision = true
    @State private var isLoading = true
    @State private var errorMessage: String?
    @State private var confirmLarge: GGUFOption?
    @State private var downloadError: String?

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 4) {
                    Text(repo.id).font(.headline)
                    Text(repo.source.displayName).font(.caption).foregroundStyle(.secondary)
                }
                if let url = LocalModelHub.webURL(repo: repo.id, source: repo.source) {
                    Link(destination: url) {
                        Label("Open Model Page", systemImage: "safari")
                    }
                }
            }

            if isLoading {
                Section { HStack { Spacer(); ProgressView(); Spacer() } }
            } else if let errorMessage {
                Section {
                    Text(errorMessage).foregroundStyle(.red).font(.callout)
                    Button("Retry") { Task { await load() } }
                }
            } else if options.isEmpty {
                Section {
                    Text("This repository has no GGUF model files.").foregroundStyle(.secondary)
                }
            } else {
                if let projector {
                    Section {
                        Toggle(isOn: $includeVision) {
                            VStack(alignment: .leading, spacing: 2) {
                                Label("Image Support", systemImage: "photo")
                                Text("Also download the vision projector \((projector.path as NSString).lastPathComponent) (\(LocalModelFormat.bytes(projector.size)))")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    } footer: {
                        Text("This model can read images. The projector is loaded only when a message contains an image.")
                    }
                }
                Section {
                    ForEach(options) { option in
                        optionRow(option)
                    }
                } header: {
                    Text("Files")
                } footer: {
                    Text("Q4_K_M is a good balance of quality and size. Files marked \"Too large\" probably exceed this device's memory.")
                }
            }
        }
        .navigationTitle("Choose a File")
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
        .alert("Large Model", isPresented: Binding(get: { confirmLarge != nil }, set: { if !$0 { confirmLarge = nil } })) {
            Button("Download Anyway") {
                if let o = confirmLarge { start(o) }
                confirmLarge = nil
            }
            Button("Cancel", role: .cancel) { confirmLarge = nil }
        } message: {
            Text("This model probably needs more memory than this device can give an app, and may fail to load.")
        }
        .alert("Download Failed", isPresented: Binding(get: { downloadError != nil }, set: { if !$0 { downloadError = nil } })) {
            Button("OK", role: .cancel) { downloadError = nil }
        } message: {
            Text(downloadError ?? "")
        }
    }

    @ViewBuilder
    private func optionRow(_ option: GGUFOption) -> some View {
        let fits = LocalModelFormat.fitsInMemory(fileSize: option.totalSize)
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(option.displayName).font(.callout).lineLimit(2)
                HStack(spacing: 6) {
                    if let q = option.quantization { LocalModelBadge(text: q) }
                    Text(LocalModelFormat.bytes(option.totalSize))
                    if option.isSplit { Text("· \(option.parts.count) parts") }
                    if !fits { LocalModelBadge(text: AppLocalized("Too large"), tint: .orange) }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            if let model = downloads.downloadedModel(repo: repo.id, option: option, source: repo.source) {
                if let projector, !model.supportsVision {
                    if downloads.isDownloadingProjector(for: model.id) {
                        ProgressView()
                    } else {
                        Button {
                            do { try downloads.startProjector(for: model, projector: projector) }
                            catch { downloadError = error.localizedDescription }
                        } label: {
                            Label("Add Images", systemImage: "photo.badge.plus")
                                .labelStyle(.iconOnly)
                                .font(.title3)
                        }
                        .buttonStyle(.borderless)
                    }
                }
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            } else if downloads.isDownloading(repo: repo.id, option: option, source: repo.source) {
                ProgressView()
            } else {
                Button {
                    if fits { start(option) } else { confirmLarge = option }
                } label: {
                    Image(systemName: "arrow.down.circle").font(.title3)
                }
                .buttonStyle(.borderless)
            }
        }
    }

    private func start(_ option: GGUFOption) {
        do {
            try downloads.start(repo: repo.id, option: option,
                                projector: includeVision ? projector : nil, source: repo.source)
        } catch {
            downloadError = error.localizedDescription
        }
    }

    private func load() async {
        isLoading = true
        errorMessage = nil
        do {
            let contents = try await LocalModelHub.repoContents(repo: repo.id, source: repo.source)
            options = contents.options
            projector = contents.preferredProjector
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
