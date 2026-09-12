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
  private let generate = WorkflowDraft(kind: .generate)
  private let plan = WorkflowDraft(kind: .plan)
  private let renderPlan = WorkflowDraft(kind: .renderPlan)
  private let replay = WorkflowDraft(kind: .replay)
  private let batch = WorkflowDraft(kind: .batch)
  private let prepare = WorkflowDraft(kind: .prepare)
  private let doctor = WorkflowDraft(kind: .doctor)

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
      runner.start(submission)
    } catch {
      inputError = error.localizedDescription
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
  weak var runner: RunController?

  func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
    guard let runner, runner.isRunning else { return .terminateNow }
    let alert = NSAlert()
    alert.messageText = "Cancel the active workflow and quit?"
    alert.informativeText =
      "The engine will stop at its next checkpoint. Completed recordings are kept. An interrupted recording can be retried with matching resume settings."
    alert.addButton(withTitle: "Keep Running")
    alert.addButton(withTitle: "Cancel and Quit")
    guard alert.runModal() == .alertSecondButtonReturn else { return .terminateCancel }
    guard runner.isRunning else { return .terminateNow }
    runner.onIdle = { sender.reply(toApplicationShouldTerminate: true) }
    runner.cancel()
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
        .onAppear { appDelegate.runner = workspace.runner }
    }
    .defaultSize(width: 1220, height: 800)
    .windowResizability(.contentMinSize)
    .commands {
      CommandGroup(replacing: .newItem) {
        Button("Open Request…") { workspace.openRequest() }
          .keyboardShortcut("o", modifiers: .command)
          .disabled(workspace.runner.isRunning)
        Button("Show Output Folder") { workspace.revealOutputs() }
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
