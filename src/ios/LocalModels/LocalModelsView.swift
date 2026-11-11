//
//  LocalModelsView.swift
//  Minis — on-device models
//
//  Settings → Local Models: downloaded models, running downloads, the
//  entry to the Hugging Face / ModelScope browser, and inference settings.
//

import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct LocalModelsView: View {
    @ObservedObject private var downloads = LocalModelDownloadManager.shared

    @AppStorage(LocalModelSettings.contextLengthKey) private var contextLength = 8192
    @AppStorage(LocalModelSettings.maxOutputTokensKey) private var maxOutputTokens = 2048
    @AppStorage(LocalModelSettings.gpuOffloadKey) private var gpuOffload = true
    @AppStorage(LocalModelSettings.temperatureKey) private var temperature = 0.7
    @AppStorage(LocalModelSettings.toolCallingKey) private var toolCalling = true
    @AppStorage(LocalModelSettings.huggingFaceEndpointKey) private var hfEndpoint = LocalModelSettings.defaultHuggingFaceEndpoint

    @AppStorage(LocalModelSettings.compactPromptKey) private var compactPrompt = true
    @State private var serverRunning = LocalLLMServer.shared.isRunning
    @State private var serverHealthy = LocalLLMServer.shared.isHealthy
    @State private var engine = LlamaEngine.shared.phase
    @State private var queued = 0
    @State private var now = Date()
    @State private var loadError: String?
    @State private var showLinkSheet = false
    @State private var linkText = ""
    @State private var importing: String?
    @State private var pendingDelete: LocalModelFile?
    @State private var errorMessage: String?
    @State private var backgroundServer = LocalModelSettings.backgroundServer
    @State private var showKey = false

    var body: some View {
        List {
            statusSection
            backgroundServerSection
            engineSection
            modelsSection
            if !downloads.jobs.isEmpty { downloadsSection }
            Section {
                NavigationLink {
                    LocalModelBrowserView()
                } label: {
                    Label("Download Models", systemImage: "arrow.down.circle.fill")
                }
                Button {
                    DocumentPickerPresenter.present { urls in
                        guard let url = urls.first else { return }
                        runImport(url.lastPathComponent) { try await downloads.importModel(from: url) }
                    }
                } label: {
                    Label("Import from Files", systemImage: "folder.badge.plus")
                }
                .disabled(importing != nil)
                Button {
                    linkText = UIPasteboard.general.string?.hasPrefix("http") == true ? (UIPasteboard.general.string ?? "") : ""
                    showLinkSheet = true
                } label: {
                    Label("Download from Link", systemImage: "link.badge.plus")
                }
                if let importing {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text("Importing \(importing)…").foregroundStyle(.secondary).lineLimit(1)
                    }
                }
            } footer: {
                Text("Search GGUF models on Hugging Face or ModelScope, import a .gguf file from Files (iCloud Drive, other apps, USB drives, network shares), or paste a direct download link. Small quantized models (1–4B parameters, Q4_K_M) run best on a phone.")
            }
            settingsSection
        }
        .navigationTitle("Local Models")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            LocalModelDownloadManager.shared.activate()
            if !downloads.models.isEmpty {
                LocalModelProviderRegistrar.shared.ensureRegistered()
            }
        }
        .task {
            // Live status: engine and listener state live off the main actor;
            // read them every second, and test the port itself every 10 s
            // (a listener can claim "ready" after a long suspension while
            // nothing answers — the check rebuilds it then).
            var tick = 0
            while !Task.isCancelled {
                if tick % 10 == 0 { LocalLLMServer.shared.checkHealth() }
                tick += 1
                serverRunning = LocalLLMServer.shared.isRunning
                serverHealthy = LocalLLMServer.shared.isHealthy
                engine = LlamaEngine.shared.phase
                queued = LlamaEngine.shared.queuedCount
                now = Date()
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
        .onChange(of: contextLength) { _ in LocalModelProviderRegistrar.shared.modelsChanged() }
        .onChange(of: maxOutputTokens) { _ in LocalModelProviderRegistrar.shared.modelsChanged() }
        .alert("Delete Model?", isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })) {
            Button("Delete", role: .destructive) {
                if let m = pendingDelete { downloads.deleteModel(id: m.id) }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: {
            if let m = pendingDelete {
                Text("\(m.displayName) (\(LocalModelFormat.bytes(m.sizeBytes))) will be removed from this device.")
            }
        }
        .alert("Download from Link", isPresented: $showLinkSheet) {
            TextField("https://…/model.gguf", text: $linkText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button("Download") {
                do { try downloads.startLinkDownload(linkText) }
                catch { errorMessage = error.localizedDescription }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Paste a direct link to a .gguf file. Hugging Face and ModelScope file pages work too.")
        }
        .alert("Error", isPresented: Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private func runImport(_ name: String, _ work: @escaping () async throws -> Void) {
        importing = name
        Task { @MainActor in
            do { try await work() }
            catch { errorMessage = error.localizedDescription }
            importing = nil
        }
    }

    // MARK: Sections

    private var statusSection: some View {
        Section {
            HStack {
                Label("Local Server", systemImage: "server.rack")
                Spacer()
                Circle()
                    .fill(serverRunning && serverHealthy ? Color.green : (serverRunning ? Color.orange : Color.secondary.opacity(0.4)))
                    .frame(width: 8, height: 8)
                (serverRunning && serverHealthy ? Text("Running")
                    : serverRunning ? Text("Not responding") : Text("Stopped"))
                    .foregroundStyle(.secondary)
            }
            if !(serverRunning && serverHealthy) {
                Button {
                    LocalModelProviderRegistrar.shared.ensureRegistered()
                    LocalLLMServer.shared.restart()
                } label: {
                    Label("Restart Local Server", systemImage: "arrow.clockwise")
                }
            }
            HStack {
                Label("Device Memory", systemImage: "memorychip")
                Spacer()
                Text(LocalModelFormat.bytes(Int64(ProcessInfo.processInfo.physicalMemory)))
                    .foregroundStyle(.secondary)
            }
        } footer: {
            Text("Downloaded models appear under the provider \"\(LocalModelProviderRegistrar.instanceLabel)\". Pick one in the model picker like any other model. Everything runs on this device; no data leaves it.")
        }
    }

    // MARK: Engine (live)

    // MARK: Background server

    @ViewBuilder private var backgroundServerSection: some View {
        Section {
            Toggle(isOn: Binding(
                get: { backgroundServer },
                set: { on in
                    backgroundServer = on
                    LocalModelSettings.backgroundServer = on
                    if on {
                        LocalModelProviderRegistrar.shared.ensureRegistered()
                        LocalLLMServer.shared.start()
                    }
                }
            )) {
                Label("Keep Server Running in Background", systemImage: "server.rack")
            }
            if backgroundServer {
                LabeledContent("Address") {
                    Text(LocalLLMServer.baseURL + "/v1")
                        .font(.footnote.monospaced())
                        .textSelection(.enabled)
                }
                if let key = LocalModelProviderRegistrar.shared.serverAPIKey {
                    HStack {
                        Text("API Key")
                        Spacer()
                        Text(showKey ? key : "••••••••")
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Button {
                            UIPasteboard.general.string = key
                        } label: { Image(systemName: "doc.on.doc") }
                            .buttonStyle(.borderless)
                    }
                    .onTapGesture { showKey.toggle() }
                }
            }
        } header: {
            Text("On-device API Server")
        } footer: {
            Text("Other apps on this device can call the loaded model at this OpenAI-compatible address while Minis is in the background. It uses the background keep-alive (some battery) and needs a model already loaded. Note: image recognition and GPU inference don't run while in the background — use a CPU model for background serving. Turn this off when you don't need it.")
        }
    }

    private var engineSection: some View {
        Section {
            HStack(spacing: 10) {
                Circle()
                    .fill(phaseColor)
                    .frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 2) {
                    phaseTitle
                    if let detail = phaseDetail {
                        Text(detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Spacer()
                if isWorking {
                    ProgressView().controlSize(.small)
                }
            }
            if case .generating = engine.phase {
                Button(role: .destructive) {
                    LlamaEngine.shared.cancelAll()
                } label: {
                    Label("Stop Generating", systemImage: "stop.circle")
                }
            }
            Menu {
                if downloads.models.isEmpty {
                    Text("No models downloaded yet.")
                }
                ForEach(downloads.models) { model in
                    Button {
                        load(model)
                    } label: {
                        if isLoaded(model) {
                            Label(model.displayName, systemImage: "checkmark")
                        } else {
                            Text(model.displayName)
                        }
                    }
                }
            } label: {
                Label("Load Model", systemImage: "square.and.arrow.down.on.square")
            }
            .disabled(downloads.models.isEmpty || isLoading)
            Button {
                LlamaEngine.shared.unloadAll()
            } label: {
                Label("Unload Model from Memory", systemImage: "eject")
            }
            .disabled(engine.phase == .idle)
            if let loadError {
                Text(loadError).font(.caption).foregroundStyle(.red)
            } else if engine.phase != .idle, let note = LlamaEngine.shared.loadNote {
                Text(note).font(.caption).foregroundStyle(.orange)
            }
        } header: {
            Text("Model Status")
        } footer: {
            Text("Live state of the on-device engine. Loading a model here makes the first reply faster; a chat that uses another model swaps it automatically.")
        }
    }

    private var elapsed: Int { max(0, Int(now.timeIntervalSince(engine.since))) }
    private var isLoading: Bool { if case .loading = engine.phase { return true }; return false }
    private var isWorking: Bool {
        switch engine.phase {
        case .loading, .generating: return true
        default: return false
        }
    }
    /// A generation that has run well past the per-request limit is stuck.
    private var isStuck: Bool {
        if case .generating = engine.phase { return elapsed > Int(LlamaEngine.maxRequestSeconds) + 30 }
        return false
    }

    private func isLoaded(_ model: LocalModelFile) -> Bool {
        LlamaEngine.shared.loadedPath == model.primaryURL.path
    }

    private var phaseColor: Color {
        if isStuck { return .red }
        switch engine.phase {
        case .idle: return Color.secondary.opacity(0.4)
        case .loading: return .orange
        case .ready: return .green
        case .generating: return .blue
        }
    }

    @ViewBuilder private var phaseTitle: some View {
        if isStuck {
            Text("Not responding — restart the app")
        } else {
            switch engine.phase {
            case .idle: Text("Nothing loaded")
            case .loading: Text("Loading… \(elapsed)s")
            case .ready: Text("Loaded, idle")
            case .generating: Text("Generating… \(elapsed)s")
            }
        }
    }

    private var phaseDetail: String? {
        var parts: [String] = []
        switch engine.phase {
        case .idle: return nil
        case .loading(let name), .ready(let name): parts.append(name)
        case .generating(let name):
            parts.append(name)
            if engine.chunks > 0 { parts.append(AppLocalized("\(engine.chunks) tokens")) }
        }
        if queued > 0 { parts.append(AppLocalized("\(queued) waiting")) }
        return parts.joined(separator: " · ")
    }

    private func load(_ model: LocalModelFile) {
        loadError = nil
        LocalModelProviderRegistrar.shared.ensureRegistered()
        let config = LlamaEngine.Config(
            path: model.primaryURL.path,
            contextLength: LocalModelSettings.contextLength,
            gpuOffload: LocalModelSettings.gpuOffload,
            mmprojPath: nil
        )
        Task {
            do {
                try await LlamaEngine.shared.preload(config: config)
            } catch {
                loadError = error.localizedDescription
            }
            engine = LlamaEngine.shared.phase
        }
    }

    private var modelsSection: some View {
        Section {
            if downloads.models.isEmpty {
                Text("No models downloaded yet.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(downloads.models) { model in
                    LocalModelRow(model: model)
                        .swipeActions {
                            Button(role: .destructive) { pendingDelete = model } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                        .contextMenu {
                            if !model.supportsVision {
                                Button {
                                    let id = model.id
                                    DocumentPickerPresenter.present { urls in
                                        guard let url = urls.first else { return }
                                        runImport(url.lastPathComponent) { try await downloads.importProjector(from: url, for: id) }
                                    }
                                } label: {
                                    Label("Add Vision Projector from Files", systemImage: "photo.badge.plus")
                                }
                            }
                            Button(role: .destructive) { pendingDelete = model } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                }
            }
        } header: {
            Text("Downloaded")
        } footer: {
            if !downloads.models.isEmpty {
                Text("Storage used: \(LocalModelFormat.bytes(downloads.totalDiskUsage))")
            }
        }
    }

    private var downloadsSection: some View {
        Section("Downloading") {
            ForEach(downloads.jobs) { job in
                LocalDownloadRow(job: job, written: downloads.progress[job.id] ?? 0)
            }
        }
    }

    private var settingsSection: some View {
        Section {
            Picker("Context Length", selection: $contextLength) {
                ForEach(LocalModelSettings.contextLengthOptions, id: \.self) { n in
                    Text("\(n) tokens").tag(n)
                }
            }
            Picker("Max Output", selection: $maxOutputTokens) {
                ForEach(LocalModelSettings.maxOutputOptions, id: \.self) { n in
                    Text("\(n) tokens").tag(n)
                }
            }
            VStack(alignment: .leading) {
                HStack {
                    Text("Temperature")
                    Spacer()
                    Text(String(format: "%.2f", temperature)).foregroundStyle(.secondary)
                }
                Slider(value: $temperature, in: 0...1.5, step: 0.05)
            }
            Toggle("Metal GPU Acceleration", isOn: $gpuOffload)
            Toggle("Compact Prompt (Small Models)", isOn: $compactPrompt)
            Toggle("Tool Calling", isOn: $toolCalling)
                .disabled(compactPrompt)
            Picker("Hugging Face Endpoint", selection: $hfEndpoint) {
                Text("huggingface.co").tag(LocalModelSettings.defaultHuggingFaceEndpoint)
                Text("hf-mirror.com").tag(LocalModelSettings.huggingFaceMirrorEndpoint)
            }
        } header: {
            Text("Inference")
        } footer: {
            Text("Compact Prompt replaces the app's long agent instructions (over 10K tokens) with a short one so small models answer normally. With Tool Calling on, it still gives the model a shell and file read/write, so it can run commands and install software; turn Compact Prompt off only for capable models (7B+) to give them every tool. A longer context keeps more history but uses more memory. Use hf-mirror.com if huggingface.co is unreachable.")
        }
    }
}

// MARK: - Rows

private struct LocalModelRow: View {
    let model: LocalModelFile

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(model.displayName)
                .font(.body)
                .lineLimit(2)
            HStack(spacing: 6) {
                if let q = LocalModelFormat.quantization(of: model.primaryRelativePath) {
                    LocalModelBadge(text: q)
                }
                if model.supportsVision {
                    LocalModelBadge(text: AppLocalized("Vision"), tint: .purple)
                }
                Text(LocalModelFormat.bytes(model.sizeBytes))
                Text("·")
                Text(model.source.displayName)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            Text(model.repoId)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .lineLimit(1)
        }
        .padding(.vertical, 2)
    }
}

private struct LocalDownloadRow: View {
    @ObservedObject private var downloads = LocalModelDownloadManager.shared
    let job: LocalModelDownloadJob
    let written: Int64

    private var fraction: Double {
        let total = job.totalSize
        guard total > 0 else { return 0 }
        return min(1, Double(written) / Double(total))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(job.displayName).lineLimit(1)
                Spacer()
                controls
            }
            ProgressView(value: fraction)
            HStack {
                Text("\(LocalModelFormat.bytes(written)) / \(LocalModelFormat.bytes(job.totalSize))")
                Spacer()
                switch job.state {
                case .downloading: Text("\(Int(fraction * 100))%")
                case .paused: Text("Paused")
                case .failed: Text("Failed").foregroundStyle(.red)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            if let error = job.errorMessage, job.state == .failed {
                Text(error).font(.caption2).foregroundStyle(.red)
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder private var controls: some View {
        HStack(spacing: 16) {
            switch job.state {
            case .downloading:
                Button { downloads.pause(jobId: job.id) } label: { Image(systemName: "pause.circle") }
            case .paused, .failed:
                Button { downloads.resume(jobId: job.id) } label: { Image(systemName: "arrow.clockwise.circle") }
            }
            Button(role: .destructive) { downloads.cancel(jobId: job.id) } label: { Image(systemName: "xmark.circle") }
        }
        .buttonStyle(.borderless)
        .font(.title3)
    }
}

struct LocalModelBadge: View {
    let text: String
    var tint: Color = .accentColor

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 5)
            .padding(.vertical, 1)
            .background(tint.opacity(0.15), in: RoundedRectangle(cornerRadius: 4))
            .foregroundStyle(tint)
    }
}
