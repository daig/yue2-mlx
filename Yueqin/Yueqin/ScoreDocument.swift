import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class ScoreDocument {
  private(set) var abc = ""
  private(set) var id = UUID().uuidString
  private(set) var revision = 0
  private(set) var title = "Untitled score"
  private(set) var fileURL: URL?
  private(set) var sourceURL: URL?
  private(set) var hasDocument = false
  private(set) var isDirty = false
  private(set) var canUndo = false
  private(set) var canRedo = false
  private(set) var selectionStart: Int?
  private(set) var truncated = false
  var errorMessage: String?
  @ObservationIgnored let undoManager = UndoManager()
  @ObservationIgnored var flush: (() async throws -> Void)?
  @ObservationIgnored private var savedBytes = Data()
  @ObservationIgnored private var targetBytes: Data?
  @ObservationIgnored private var replacing = false
  @ObservationIgnored private var saving = false

  init() {
    undoManager.groupsByEvent = false
  }

  @discardableResult
  func applyEdit(
    id: String, baseVersion: Int, start: Int, end: Int, text: String,
    label: String, selectionStart: Int?
  ) -> Bool {
    guard id == self.id, baseVersion == revision,
      validRange(start, end, in: abc), revision < Int.max
    else { return false }
    mutate(start: start, end: end, text: text, label: label, selection: selectionStart)
    return true
  }

  @discardableResult
  func acceptSnapshot(id: String, version: Int, abc source: String) -> Bool {
    guard id == self.id, version >= revision else { return false }
    let same = abc.utf8.elementsEqual(source.utf8)
    guard version != revision || same else { return false }
    if !same {
      guard revision < Int.max else { return false }
      let before = Array(abc.utf16)
      let after = Array(source.utf16)
      var start = 0
      while start < min(before.count, after.count), before[start] == after[start] { start += 1 }
      if start > 0, start < before.count, isLowSurrogate(before[start]) { start -= 1 }
      var oldEnd = before.count
      var newEnd = after.count
      while oldEnd > start, newEnd > start, before[oldEnd - 1] == after[newEnd - 1] {
        oldEnd -= 1
        newEnd -= 1
      }
      if oldEnd < before.count, oldEnd > start, isLowSurrogate(before[oldEnd]) {
        oldEnd += 1
        newEnd += 1
      }
      let replacement = (source as NSString).substring(
        with: NSRange(location: start, length: newEnd - start))
      mutate(start: start, end: oldEnd, text: replacement, label: "Edit score", selection: nil)
    }
    revision = version
    return true
  }

  func undo() {
    guard undoManager.canUndo else { return }
    undoManager.undo()
    refreshState()
  }

  func redo() {
    guard undoManager.canRedo else { return }
    undoManager.redo()
    refreshState()
  }

  func newScore() async {
    _ = await replace(
      abc:
        "X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=90\nV: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\nK:C\nV: Vocal\nz32|\nV: Ins\nz32|\n",
      sourceURL: nil, title: "Untitled score", truncated: false)
  }

  func openScore() async {
    guard !replacing, !saving else { return }
    replacing = true
    defer { replacing = false }
    guard await confirmReplacement() else { return }
    let epoch = id
    let version = revision
    let panel = NSOpenPanel()
    panel.title = "Open ABC score"
    panel.message = "The original file is preserved. Save creates an edited copy."
    panel.allowedContentTypes = [UTType(filenameExtension: "abc") ?? .plainText, .plainText]
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    guard await panel.begin() == .OK, let url = panel.url else { return }
    do {
      let source = try await Task.detached(priority: .userInitiated) {
        let bytes = try Data(contentsOf: url)
        let source = String(decoding: bytes, as: UTF8.self)
        guard Data(source.utf8) == bytes else {
          throw DocumentError.invalidUTF8
        }
        return source
      }.value
      guard await flushEdits(), id == epoch else { return }
      if revision != version, !(await confirmReplacement()) { return }
      load(
        source, sourceURL: url, title: url.deletingPathExtension().lastPathComponent,
        truncated: false)
    } catch {
      errorMessage = "Could not open score: \(error.localizedDescription)"
    }
  }

  @discardableResult
  func replace(abc: String, sourceURL: URL?, title: String, truncated: Bool) async -> Bool {
    guard !replacing, !saving else { return false }
    replacing = true
    defer { replacing = false }
    guard await confirmReplacement() else { return false }
    load(abc, sourceURL: sourceURL, title: title, truncated: truncated)
    return true
  }

  @discardableResult
  func save(as saveAs: Bool = false) async -> Bool {
    guard !saving else { return false }
    saving = true
    defer { saving = false }
    guard await flushEdits(), hasDocument else { return false }
    let epoch = id
    var chooseTarget = saveAs || fileURL == nil
    while true {
      var target = fileURL
      if chooseTarget {
        let panel = NSSavePanel()
        panel.title = "Save ABC score"
        panel.allowedContentTypes = [UTType(filenameExtension: "abc") ?? .plainText]
        panel.canCreateDirectories = true
        if let fileURL {
          panel.directoryURL = fileURL.deletingLastPathComponent()
          panel.nameFieldStringValue = fileURL.lastPathComponent
        } else if let sourceURL {
          panel.directoryURL = await Task.detached(priority: .userInitiated) {
            var destination = sourceURL.deletingLastPathComponent()
            var ancestor = destination.resolvingSymlinksInPath()
            while true {
              let parent = ancestor.deletingLastPathComponent()
              if FileManager.default.fileExists(
                atPath: ancestor.appendingPathComponent("plan_manifest.json").path)
              {
                destination = parent
              }
              if parent == ancestor { break }
              ancestor = parent
            }
            return destination
          }.value
          panel.nameFieldStringValue =
            sourceURL.deletingPathExtension().lastPathComponent + "-edited.abc"
        } else {
          panel.nameFieldStringValue = "Untitled score.abc"
        }
        guard await panel.begin() == .OK, let chosen = panel.url else { return false }
        target = chosen
      }
      guard id == epoch, let target, await flushEdits(), id == epoch else { return false }
      let bytes = Data(abc.utf8)
      let expected = chooseTarget ? nil : targetBytes
      do {
        try await Task.detached(priority: .userInitiated) {
          try Self.write(bytes, to: target, expected: expected)
        }.value
        guard id == epoch else { return false }
        fileURL = target
        targetBytes = bytes
        savedBytes = bytes
        title = target.deletingPathExtension().lastPathComponent
        errorMessage = nil
        refreshState()
        return true
      } catch DocumentError.externalChange {
        let alert = NSAlert()
        alert.messageText = "This score changed outside Yueqin"
        alert.informativeText = "Save your edits to another file to preserve the external changes."
        alert.addButton(withTitle: "Save As…")
        alert.addButton(withTitle: "Cancel")
        guard await response(to: alert) == .alertFirstButtonReturn else { return false }
        chooseTarget = true
      } catch {
        errorMessage = "Could not save score: \(error.localizedDescription)"
        return false
      }
    }
  }

  func confirmReplacement() async -> Bool {
    guard await flushEdits() else { return false }
    while isDirty {
      let epoch = id
      let version = revision
      let alert = NSAlert()
      alert.messageText = "Save changes to \"\(title)\"?"
      alert.informativeText = "Your score edits will be lost if you don’t save them."
      alert.addButton(withTitle: "Save…")
      alert.addButton(withTitle: "Don’t Save")
      alert.addButton(withTitle: "Cancel")
      let answer = await response(to: alert)
      guard id == epoch else { return false }
      if answer == .alertFirstButtonReturn {
        guard await save() else { return false }
      } else if answer == .alertSecondButtonReturn {
        guard await flushEdits() else { return false }
        if revision == version { return true }
      } else {
        return false
      }
    }
    return true
  }

  private func flushEdits() async -> Bool {
    do {
      try await flush?()
      return true
    } catch {
      errorMessage = "Could not synchronize score edits: \(error.localizedDescription)"
      return false
    }
  }

  private func response(to alert: NSAlert) async -> NSApplication.ModalResponse {
    if let window = NSApp.keyWindow, window.attachedSheet == nil {
      return await alert.beginSheetModal(for: window)
    }
    return alert.runModal()
  }

  private func load(_ source: String, sourceURL: URL?, title: String, truncated: Bool) {
    undoManager.removeAllActions()
    abc = source
    id = UUID().uuidString
    revision = 0
    self.title = title
    self.sourceURL = sourceURL
    fileURL = nil
    targetBytes = nil
    savedBytes = Data(source.utf8)
    hasDocument = true
    selectionStart = nil
    self.truncated = truncated
    errorMessage = nil
    refreshState()
  }

  private func mutate(start: Int, end: Int, text: String, label: String, selection: Int?) {
    let range = NSRange(location: start, length: end - start)
    let removed = (abc as NSString).substring(with: range)
    let oldSelection = selectionStart
    let newEnd = start + text.utf16.count
    let normalEdit = !undoManager.isUndoing && !undoManager.isRedoing
    if normalEdit { undoManager.beginUndoGrouping() }
    undoManager.registerUndo(withTarget: self) { document in
      document.mutate(
        start: start, end: newEnd, text: removed, label: label, selection: oldSelection)
    }
    undoManager.setActionName(label.isEmpty ? "Edit score" : label)
    if normalEdit { undoManager.endUndoGrouping() }
    abc = (abc as NSString).replacingCharacters(in: range, with: text)
    revision += 1
    hasDocument = true
    selectionStart = selection.flatMap { validRange($0, $0, in: abc) ? $0 : nil }
    refreshState()
  }

  private func refreshState() {
    isDirty = !savedBytes.elementsEqual(abc.utf8)
    canUndo = undoManager.canUndo
    canRedo = undoManager.canRedo
  }

  private func validRange(_ start: Int, _ end: Int, in source: String) -> Bool {
    let string = source as NSString
    guard start >= 0, end >= start, end <= string.length else { return false }
    return (start == string.length || !isLowSurrogate(string.character(at: start)))
      && (end == string.length || !isLowSurrogate(string.character(at: end)))
  }

  private func isLowSurrogate(_ unit: UInt16) -> Bool { (0xDC00...0xDFFF).contains(unit) }

  private enum DocumentError: LocalizedError {
    case invalidUTF8, protectedPlan, externalChange

    var errorDescription: String? {
      switch self {
      case .invalidUTF8:
        "The file is not lossless UTF-8. Convert its encoding explicitly before importing it."
      case .protectedPlan:
        "This folder contains an integrity-protected plan. Save the edited ABC outside that plan folder."
      case .externalChange: "The saved file was changed or removed by another application."
      }
    }
  }

  nonisolated private static func write(_ bytes: Data, to target: URL, expected: Data?) throws {
    let manager = FileManager.default
    // Check both the chosen path and resolved symlink path, including every ancestor.
    for location in [
      target.standardizedFileURL, target.resolvingSymlinksInPath().standardizedFileURL,
    ] {
      var directory = location.deletingLastPathComponent()
      while true {
        if manager.fileExists(atPath: directory.appendingPathComponent("plan_manifest.json").path) {
          throw DocumentError.protectedPlan
        }
        let parent = directory.deletingLastPathComponent()
        if parent == directory { break }
        directory = parent
      }
    }
    if let expected {
      guard let current = try? Data(contentsOf: target), current == expected else {
        throw DocumentError.externalChange
      }
    }
    try bytes.write(to: target, options: .atomic)
  }
}

/// Interposes only close approval; all other optional delegate methods retain SwiftUI's delegate.
@MainActor
struct ScoreWindowGuard: NSViewRepresentable {
  let document: ScoreDocument

  func makeCoordinator() -> Coordinator { Coordinator(document: document) }

  func makeNSView(context: Context) -> GuardView {
    let view = GuardView()
    view.coordinator = context.coordinator
    return view
  }

  func updateNSView(_ nsView: GuardView, context: Context) {
    context.coordinator.document = document
    context.coordinator.attach(to: nsView.window)
    nsView.window?.isDocumentEdited = document.isDirty
  }

  static func dismantleNSView(_ nsView: GuardView, coordinator: Coordinator) {
    coordinator.detach()
  }

  final class GuardView: NSView {
    weak var coordinator: Coordinator?
    override func viewDidMoveToWindow() {
      super.viewDidMoveToWindow()
      coordinator?.attach(to: window)
    }
  }

  final class Coordinator: NSObject, NSWindowDelegate {
    var document: ScoreDocument
    private weak var window: NSWindow?
    private weak var previous: (any NSWindowDelegate)?
    private var confirming = false
    private var approved = false

    init(document: ScoreDocument) { self.document = document }

    @MainActor
    func attach(to window: NSWindow?) {
      guard let window else {
        detach()
        return
      }
      if self.window === window, window.delegate === self { return }
      detach()
      self.window = window
      previous = window.delegate
      window.delegate = self
    }

    @MainActor
    func detach() {
      if let window, window.delegate === self { window.delegate = previous }
      window = nil
      previous = nil
    }

    override func responds(to selector: Selector!) -> Bool {
      super.responds(to: selector) || (previous?.responds(to: selector) ?? false)
    }

    override func forwardingTarget(for selector: Selector!) -> Any? {
      if previous?.responds(to: selector) == true { return previous }
      return super.forwardingTarget(for: selector)
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
      if approved {
        approved = false
        return previous?.windowShouldClose?(sender) ?? true
      }
      guard !confirming else { return false }
      confirming = true
      Task { @MainActor [weak self, weak sender] in
        guard let self else { return }
        let mayClose = await document.confirmReplacement()
        confirming = false
        guard mayClose, let sender, window === sender else { return }
        approved = true
        sender.performClose(nil)
        approved = false
      }
      return false
    }
  }
}
