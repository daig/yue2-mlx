import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class YueqinWorkspace {
  var selection: WorkflowKind? = .generate
  var inputError: String?
  let settings = EngineSettings()
  let runner = RunController()
  let scoreDocument: ScoreDocument
  let scoreSession: ABCScoreSession
  var showPlanningControls = true
  private(set) var pendingPlannedScore: PlannedScore?
  private var planningDocumentID: String?
  private var planningDocumentRevision: Int?
  private let generate = WorkflowDraft(kind: .generate)
  private let plan = WorkflowDraft(kind: .plan)
  private let renderPlan = WorkflowDraft(kind: .renderPlan)
  private let replay = WorkflowDraft(kind: .replay)
  private let batch = WorkflowDraft(kind: .batch)
  private let prepare = WorkflowDraft(kind: .prepare)
  private let doctor = WorkflowDraft(kind: .doctor)

  init() {
    let document = ScoreDocument()
    scoreDocument = document
    scoreSession = ABCScoreSession(document: document)
  }

  var kind: WorkflowKind { selection ?? .generate }
  var draft: WorkflowDraft { draft(for: kind) }

  func draft(for kind: WorkflowKind) -> WorkflowDraft {
    switch kind {
    case .generate: generate
    case .plan: plan
    case .renderPlan: renderPlan
    case .replay: replay
    case .batch: batch
    case .prepare: prepare
    case .doctor: doctor
    }
  }

  func runActive() {
    guard !runner.isRunning else { return }
    inputError = nil
    do {
      let submission = try draft.makeSubmission(settings: settings)
      draft.save()
      settings.save()
      if kind == .plan {
        planningDocumentID = scoreDocument.id
        planningDocumentRevision = scoreDocument.revision
      }
      runner.start(submission)
    } catch {
      inputError = error.localizedDescription
      if kind == .plan { showPlanningControls = true }
    }
  }

  func useOutput(_ url: URL, for kind: WorkflowKind) {
    guard !runner.isRunning else { return }
    let next = draft(for: kind)
    next.inputPath = url.path
    next.outputPath = ""
    next.save()
    selection = kind
    inputError = nil
  }

  func receivePlannedScore(_ score: PlannedScore) async {
    do {
      try await scoreSession.flush()
      if scoreDocument.isDirty
        || planningDocumentID != scoreDocument.id
        || planningDocumentRevision != scoreDocument.revision
      {
        pendingPlannedScore = score
        return
      }
      await openPlannedScore(score)
    } catch {
      pendingPlannedScore = score
      scoreDocument.errorMessage = error.localizedDescription
    }
  }

  func openPlannedScore(_ score: PlannedScore) async {
    if await scoreDocument.replace(
      abc: score.abc, sourceURL: score.outputURL.appendingPathComponent("score.abc"),
      title: "Planned score", truncated: score.truncated)
    {
      pendingPlannedScore = nil
      showPlanningControls = false
    } else {
      pendingPlannedScore = score
    }
  }

  func newScore() async {
    selection = .plan
    await scoreDocument.newScore()
    if scoreDocument.hasDocument { showPlanningControls = false }
  }

  func openScore() async {
    selection = .plan
    await scoreDocument.openScore()
    if scoreDocument.hasDocument { showPlanningControls = false }
  }

  func useEditorScore() async {
    guard !runner.isRunning else { return }
    do {
      try await scoreSession.flush()
      guard scoreSession.canUseScore else {
        scoreDocument.errorMessage =
          "Resolve the score's compatibility issues before using it in a song."
        return
      }
      // Snapshot the editor into a fresh request. Never modify saved plan tokens
      // or render an old prefix while implying that it includes these edits.
      var request = plan.request
      request.scoreSource = "text"
      request.abc = scoreDocument.abc
      request.abcPath = ""
      request.cot = "full"
      if request.source == "file" {
        request.overrideABC = true
        request.overrideCot = true
      }
      generate.request = request
      generate.outputPath = ""
      generate.resume = false
      generate.save()
      scoreSession.command("stop")
      selection = .generate
      inputError = nil
    } catch {
      scoreDocument.errorMessage = error.localizedDescription
    }
  }

  func openRequest() {
    guard !runner.isRunning else { return }
    let panel = NSOpenPanel()
    panel.title = "Open a song request"
    panel.message = "Relative lyrics and score paths remain relative to this JSON file."
    panel.allowedContentTypes = [.json]
    panel.canChooseDirectories = false
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    var request = generate.request
    request.source = "file"
    request.filePath = url.path
    request.overrideID = false
    request.overrideStyle = false
    request.overrideLyrics = false
    request.overrideCot = false
    request.overrideSeed = false
    request.overrideCFG = false
    request.overrideABC = false
    generate.request = request
    generate.save()
    selection = .generate
    inputError = nil
  }

  func revealOutputs() {
    let path = (settings.outputRoot as NSString).expandingTildeInPath
    guard path.hasPrefix("/") else {
      inputError = "Choose an absolute output root in Engine settings."
      return
    }
    let url = URL(fileURLWithPath: path, isDirectory: true)
    do {
      try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
      NSWorkspace.shared.open(url)
    } catch {
      inputError = "Could not open the output folder: \(error.localizedDescription)"
    }
  }
}

@MainActor
final class YueqinAppDelegate: NSObject, NSApplicationDelegate {
  weak var workspace: YueqinWorkspace?
  private var confirmingTermination = false

  func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
    guard let workspace else { return .terminateNow }
    guard !confirmingTermination else { return .terminateLater }
    confirmingTermination = true
    Task { @MainActor [self] in
      guard await workspace.scoreDocument.confirmReplacement() else {
        confirmingTermination = false
        sender.reply(toApplicationShouldTerminate: false)
        return
      }
      let runner = workspace.runner
      if runner.isRunning {
        let alert = NSAlert()
        alert.messageText = "Cancel the active workflow and quit?"
        alert.informativeText =
          "The engine will stop at its next checkpoint. Completed recordings are kept."
        alert.addButton(withTitle: "Keep Running")
        alert.addButton(withTitle: "Cancel and Quit")
        guard alert.runModal() == .alertSecondButtonReturn else {
          confirmingTermination = false
          sender.reply(toApplicationShouldTerminate: false)
          return
        }
      }
      if runner.isRunning {
        runner.onIdle = { sender.reply(toApplicationShouldTerminate: true) }
        runner.cancel()
      } else {
        sender.reply(toApplicationShouldTerminate: true)
      }
    }
    return .terminateLater
  }
}

@main
struct YueqinApp: App {
  @NSApplicationDelegateAdaptor(YueqinAppDelegate.self) private var appDelegate
  @State private var workspace = YueqinWorkspace()

  var body: some Scene {
    Window("Yueqin", id: "main") {
      ContentView(workspace: workspace)
        .background(ScoreWindowGuard(document: workspace.scoreDocument).frame(width: 0, height: 0))
        .onAppear { appDelegate.workspace = workspace }
    }
    .defaultSize(width: 1220, height: 800)
    .windowResizability(.contentMinSize)
    .commands {
      CommandGroup(replacing: .newItem) {
        Button("New Score") { Task { await workspace.newScore() } }
          .keyboardShortcut("n", modifiers: .command)
        Button("Open ABC Score…") { Task { await workspace.openScore() } }
          .keyboardShortcut("o", modifiers: .command)
        Button("Save Score") { Task { await workspace.scoreDocument.save() } }
          .keyboardShortcut("s", modifiers: .command)
          .disabled(!workspace.scoreDocument.hasDocument)
        Button("Save Score As…") { Task { await workspace.scoreDocument.save(as: true) } }
          .keyboardShortcut("s", modifiers: [.command, .shift])
          .disabled(!workspace.scoreDocument.hasDocument)
        Divider()
        Button("Open Request…") { workspace.openRequest() }
          .keyboardShortcut("o", modifiers: [.command, .shift])
          .disabled(workspace.runner.isRunning)
        Button("Show Output Folder") { workspace.revealOutputs() }
      }
      CommandMenu("Score") {
        Group {
          Button("Undo Score Edit") { workspace.scoreDocument.undo() }
            .disabled(!workspace.scoreDocument.canUndo)
          Button("Redo Score Edit") { workspace.scoreDocument.redo() }
            .disabled(!workspace.scoreDocument.canRedo)
          Divider()
          Button("Note Input") { workspace.scoreSession.command("input") }
          Button("Select All in Voice") { workspace.scoreSession.command("selectAll") }
          Button("Copy Notes") { workspace.scoreSession.command("copy") }
          Button("Cut Notes to Rests") { workspace.scoreSession.command("cut") }
          Button("Paste Notes") { workspace.scoreSession.receiveNative("paste") }
          Divider()
          Button("Preview Tones") { workspace.scoreSession.command("play") }
          Button("Stop Preview") { workspace.scoreSession.command("stop") }
          Button("ABC Source") { workspace.scoreSession.command("source") }
          Button("Keyboard Reference") { workspace.scoreSession.command("help") }
        }
        .disabled(workspace.kind != .plan || !workspace.scoreDocument.hasDocument)
      }
      CommandMenu("Workflow") {
        Button(workspace.kind.actionTitle) { workspace.runActive() }
          .disabled(workspace.runner.isRunning)
        Button("Cancel Workflow") { workspace.runner.cancel() }
          .keyboardShortcut(".", modifiers: .command)
          .disabled(!workspace.runner.isRunning)
        Divider()
        ForEach(WorkflowKind.allCases) { kind in
          Button(kind.title) { workspace.selection = kind }
        }
      }
    }
  }
}
