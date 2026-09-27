import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

enum ScoreWorkspaceView: String, CaseIterable, Identifiable {
  case generate, edit
  var id: Self { self }
  var title: String { self == .generate ? "Generate" : "Edit" }
}

struct ScoreCandidate: Identifiable {
  let id = UUID()
  let score: PlannedScore
}

private struct SongScoreSnapshot {
  let documentID: String
  let abc: String
}

@MainActor
@Observable
final class YueqinWorkspace {
  var selection: WorkflowKind? = .generate
  var inputError: String?
  let settings = EngineSettings()
  let runner = RunController()
  let scoreDocument: ScoreDocument
  let scoreSession: ABCScoreSession
  var scoreView: ScoreWorkspaceView = .generate {
    didSet {
      if scoreView == .generate { hasUnreadScore = false }
    }
  }
  private(set) var candidate: ScoreCandidate?
  private(set) var hasUnreadScore = false
  private(set) var scoreGenerationError: String?
  private(set) var usesEditorScore = false
  private(set) var isPreparingRun = false
  private(set) var isAdoptingCandidate = false
  private var openedCandidateID: UUID?
  private var openedCandidateDocumentID: String?
  private var submittedSongScore: SongScoreSnapshot?
  private var recordedSongScore: SongScoreSnapshot?
  private let generate: WorkflowDraft
  private let plan: WorkflowDraft
  private let renderPlan = WorkflowDraft(kind: .renderPlan)
  private let replay = WorkflowDraft(kind: .replay)
  private let batch = WorkflowDraft(kind: .batch)
  private let prepare = WorkflowDraft(kind: .prepare)
  private let doctor = WorkflowDraft(kind: .doctor)

  init() {
    let document = ScoreDocument()
    scoreDocument = document
    scoreSession = ABCScoreSession(document: document)
    let song = WorkflowDraft(kind: .generate)
    let brief = WorkingBrief(fallback: song.request)
    song.attachBrief(brief)
    generate = song
    plan = WorkflowDraft(kind: .plan, brief: brief)
  }

  var kind: WorkflowKind { selection ?? .generate }
  var draft: WorkflowDraft { draft(for: kind) }
  var isGeneratingScore: Bool { runner.isRunning && runner.lastSubmission?.kind == .plan }
  var isShowingScoreGeneration: Bool { kind == .plan && scoreView == .generate }
  var candidateIsOpen: Bool {
    candidate?.id == openedCandidateID && openedCandidateID != nil
      && openedCandidateDocumentID == scoreDocument.id
  }
  var candidateActionTitle: String {
    if candidateIsOpen { return "Return to editor" }
    return scoreDocument.hasDocument ? "Replace current score…" : "Edit this score"
  }
  var recordingUsesEarlierScore: Bool {
    guard let recordedSongScore else { return false }
    return recordedSongScore.documentID != scoreDocument.id
      || !recordedSongScore.abc.utf8.elementsEqual(scoreDocument.abc.utf8)
  }
  var primaryActionTitle: String {
    guard kind == .plan else { return kind.actionTitle }
    if scoreView == .edit { return "Create song…" }
    if isGeneratingScore {
      return runner.cancellationRequested ? "Cancelling…" : "Cancel generation"
    }
    if candidate != nil && scoreGenerationError == nil { return candidateActionTitle }
    return "Generate score"
  }
  var primaryActionSymbol: String {
    if kind == .plan {
      if isGeneratingScore && scoreView == .generate { return "stop.fill" }
      if scoreView == .edit || (candidate != nil && scoreGenerationError == nil) {
        return "arrow.right"
      }
    }
    return kind == .doctor ? "stethoscope" : "play.fill"
  }
  var primaryActionDisabled: Bool {
    if isPreparingRun || isAdoptingCandidate { return true }
    if kind == .plan {
      if scoreView == .edit { return !scoreSession.canUseScore }
      if isGeneratingScore { return runner.cancellationRequested }
      if candidate != nil && scoreGenerationError == nil { return false }
    }
    return runner.isRunning || (kind == .generate && usesEditorScore && !scoreSession.canUseScore)
  }

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

  func showScoreGenerator() {
    selection = .plan
    scoreView = .generate
    hasUnreadScore = false
  }

  func showScoreEditor() {
    selection = .plan
    scoreView = .edit
  }

  func performPrimaryAction() {
    if kind == .plan {
      if scoreView == .edit {
        Task { await createSong() }
      } else if isGeneratingScore {
        runner.cancel()
      } else if candidate != nil && scoreGenerationError == nil {
        Task { await adoptCandidate() }
      } else {
        generateScore()
      }
    } else {
      startRun(kind)
    }
  }

  func generateScore() {
    guard !runner.isRunning, !isPreparingRun, !isAdoptingCandidate else { return }
    showScoreGenerator()
    startRun(.plan)
  }

  private func startRun(_ kind: WorkflowKind) {
    guard !runner.isRunning, !isPreparingRun, !isAdoptingCandidate else { return }
    isPreparingRun = true
    inputError = nil
    Task { @MainActor in
      defer { isPreparingRun = false }
      do {
        var score: SongScoreSnapshot?
        if kind == .generate && usesEditorScore {
          let documentID = scoreDocument.id
          try await scoreSession.flush()
          guard documentID == scoreDocument.id, scoreSession.canUseScore else {
            inputError = "Review the attached score before generating the song."
            return
          }
          score = SongScoreSnapshot(documentID: documentID, abc: scoreDocument.abc)
        }
        let draft = draft(for: kind)
        let submission = try draft.makeSubmission(settings: settings, scoreABC: score?.abc)
        draft.save()
        settings.save()
        if kind == .plan {
          scoreGenerationError = nil
          hasUnreadScore = false
        } else if kind == .generate {
          submittedSongScore = score
        }
        runner.start(submission)
      } catch {
        inputError = error.localizedDescription
      }
    }
  }

  func receiveCompletedRun() {
    switch runner.lastSubmission?.kind {
    case .plan:
      if case .failed(let message) = runner.planScoreStatus {
        scoreGenerationError = message
      } else if let score = runner.plannedScore,
        !score.abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      {
        receivePlannedScore(score)
      } else {
        scoreGenerationError = "The run produced no notation. Your current score is unchanged."
      }
    case .generate:
      if runner.succeeded { recordedSongScore = submittedSongScore }
    case .prepare:
      if let model = runner.preparedModelPath {
        settings.model = model
        settings.convertedDirectory = model
      }
      if let vae = runner.preparedVAEPath { settings.vae = vae }
    default: break
    }
  }

  func receivePlannedScore(_ score: PlannedScore) {
    candidate = ScoreCandidate(score: score)
    scoreGenerationError = nil
    hasUnreadScore = !isShowingScoreGeneration
  }

  func adoptCandidate() async {
    guard let candidate, !isGeneratingScore, !isAdoptingCandidate else { return }
    if candidateIsOpen {
      showScoreEditor()
      return
    }
    isAdoptingCandidate = true
    defer { isAdoptingCandidate = false }
    if await scoreDocument.replace(
      abc: candidate.score.abc,
      sourceURL: candidate.score.outputURL.appendingPathComponent("score.abc"),
      title: "Generated score", truncated: candidate.score.truncated)
    {
      openedCandidateID = candidate.id
      openedCandidateDocumentID = scoreDocument.id
      hasUnreadScore = false
      showScoreEditor()
    }
  }

  func newScore() async {
    let previousID = scoreDocument.id
    await scoreDocument.newScore()
    if scoreDocument.id != previousID {
      openedCandidateID = nil
      openedCandidateDocumentID = nil
      showScoreEditor()
    }
  }

  func openScore() async {
    let previousID = scoreDocument.id
    await scoreDocument.openScore()
    if scoreDocument.id != previousID {
      openedCandidateID = nil
      openedCandidateDocumentID = nil
      showScoreEditor()
    }
  }

  func createSong() async {
    do {
      try await scoreSession.flush()
      guard scoreSession.canUseScore else {
        scoreDocument.errorMessage =
          "Review the score's compatibility issues before creating a song."
        return
      }
      usesEditorScore = true
      if generate.request.cot == "off" { generate.request.cot = "full" }
      scoreSession.command("stop")
      selection = .generate
      inputError = nil
    } catch {
      scoreDocument.errorMessage = error.localizedDescription
    }
  }

  func removeScoreAttachment() {
    usesEditorScore = false
    inputError = nil
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

  func openRequest() {
    guard !runner.isRunning, !isPreparingRun else { return }
    let panel = NSOpenPanel()
    panel.title = "Open a song request"
    panel.message =
      "The style and lyrics are shared with score generation. Relative paths stay relative to this JSON file."
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
    usesEditorScore = false
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
        Button("Show Score Generation") { workspace.showScoreGenerator() }
        Button("Show Score Editor") { workspace.showScoreEditor() }
        Divider()
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
          Button("Preview Notation") { workspace.scoreSession.command("play") }
          Button("Stop Preview") { workspace.scoreSession.command("stop") }
          Button("ABC Source") { workspace.scoreSession.command("source") }
          Button("Review Compatibility") { workspace.scoreSession.command("showCompatibility") }
          Button("Keyboard Reference") { workspace.scoreSession.command("help") }
        }
        .disabled(
          workspace.kind != .plan || workspace.scoreView != .edit
            || !workspace.scoreDocument.hasDocument)
      }
      CommandMenu("Workflow") {
        Button(workspace.primaryActionTitle) { workspace.performPrimaryAction() }
          .disabled(workspace.primaryActionDisabled)
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
