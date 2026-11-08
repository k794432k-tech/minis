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
    @State private var loadedName: String? = LlamaEngine.shared.loadedPath.map { ($0 as NSString).lastPathComponent }
    @State private var unloadMessage: String?
    @State private var busySeconds: Int?
    @State private var showLinkSheet = false
    @State private var linkText = ""
    @State private var importing: String?
    @State private var pendingDelete: LocalModelFile?
    @State private var errorMessage: String?

    var body: some View {
        List {
            statusSection
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
            // The listener state lives off the main actor; poll it lightly.
            while !Task.isCancelled {
                serverRunning = LocalLLMServer.shared.isRunning
                loadedName = LlamaEngine.shared.loadedPath.map { ($0 as NSString).lastPathComponent }
                busySeconds = LlamaEngine.shared.busySince.map { Int(Date().timeIntervalSince($0)) }
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
                    .fill(serverRunning ? Color.green : Color.secondary.opacity(0.4))
                    .frame(width: 8, height: 8)
                (serverRunning ? Text("Running") : Text("Stopped"))
                    .foregroundStyle(.secondary)
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
            HStack {
                Text("In Memory")
                Spacer()
                Text(loadedName ?? AppLocalized("Nothing loaded"))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if let busySeconds {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("Generating… \(busySeconds)s")
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Stop") { LlamaEngine.shared.cancelAll() }
                        .buttonStyle(.borderless)
                }
            }
            Button("Unload Model from Memory") {
                LlamaEngine.shared.unloadAll {
                    Task { @MainActor in
                        loadedName = nil
                        unloadMessage = AppLocalized("Model unloaded. It loads again the next time a chat uses it.")
                    }
                }
            }
            // Always tappable: it also stops a running generation.
            if let unloadMessage {
                Text(unloadMessage).font(.caption).foregroundStyle(.secondary)
            }
        } header: {
            Text("Inference")
        } footer: {
            Text("Compact Prompt replaces the app's long agent instructions (over 10K tokens) with a short one and sends no tools, so small models answer normally; turn it off only for capable models (7B+) if you want them to use tools. A longer context keeps more history but uses more memory. Use hf-mirror.com if huggingface.co is unreachable.")
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
