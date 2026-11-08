//
//  DocumentPickerPresenter.swift
//  Minis
//
//  Presents UIDocumentPickerViewController straight from UIKit — on top of
//  whatever is frontmost — instead of wrapping it in a SwiftUI sheet. A picker
//  embedded in a sheet could show but not hand back what the user chose
//  (tapping a file or "Open" did nothing) because the picker dismissing itself
//  also tore down the sheet and its coordinator mid-callback.
//
//  Files are imported as copies: the system copies the picked file into the
//  app's own tmp/ before calling back, so callers get a plain local URL with
//  no security scope or provider placeholder to deal with.
//

import UIKit
import UniformTypeIdentifiers

@MainActor
final class DocumentPickerPresenter: NSObject, UIDocumentPickerDelegate {
    private static var active: DocumentPickerPresenter?

    private let onPick: ([URL]) -> Void
    private let onCancel: () -> Void

    private init(onPick: @escaping ([URL]) -> Void, onCancel: @escaping () -> Void) {
        self.onPick = onPick
        self.onCancel = onCancel
    }

    /// Shows the picker. `onPick` receives local copies (on the main actor).
    static func present(
        types: [UTType] = [.item],
        multiple: Bool = false,
        onCancel: @escaping () -> Void = {},
        onPick: @escaping ([URL]) -> Void
    ) {
        guard let host = topViewController() else {
            minisLogger.error("[DocPicker] no view controller to present from")
            onCancel()
            return
        }
        let presenter = DocumentPickerPresenter(onPick: onPick, onCancel: onCancel)
        active = presenter // keep the delegate alive until a callback
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: types.isEmpty ? [.item] : types, asCopy: true)
        picker.delegate = presenter
        picker.allowsMultipleSelection = multiple
        picker.modalPresentationStyle = .formSheet
        minisLogger.info("[DocPicker] presenting from \(String(describing: type(of: host)))")
        host.present(picker, animated: true)
    }

    nonisolated func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        Task { @MainActor in
            minisLogger.info("[DocPicker] picked \(urls.count) file(s)")
            Self.active = nil
            // Give the picker's own dismissal a moment so whatever the caller
            // presents next (an alert, a progress row) is not swallowed by it.
            try? await Task.sleep(nanoseconds: 350_000_000)
            if urls.isEmpty { self.onCancel() } else { self.onPick(urls) }
        }
    }

    nonisolated func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        Task { @MainActor in
            Self.active = nil
            self.onCancel()
        }
    }

    private static func topViewController() -> UIViewController? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let scene = scenes.first { $0.activationState == .foregroundActive } ?? scenes.first
        let window = scene?.windows.first { $0.isKeyWindow } ?? scene?.windows.first
        var top = window?.rootViewController
        while let next = top?.presentedViewController, !next.isBeingDismissed {
            top = next
        }
        return top
    }
}
