import AppKit
import SwiftUI

@MainActor
struct ContentView: View {
  @Bindable var workspace: YueqinWorkspace
  @State private var showsRunDetails = false

  var body: some View {
    NavigationSplitView {
      List(selection: $workspace.selection) {
        workflowSection("Create", [.generate, .plan])
        workflowSection("Tools", [.renderPlan, .replay, .batch])
        workflowSection("Setup", [.prepare, .doctor])
      }
      .listStyle(.sidebar)
      .navigationSplitViewColumnWidth(min: 175, ideal: 195, max: 240)
      .safeAreaInset(edge: .bottom) {
        VStack(alignment: .leading, spacing: 6) {
          if workspace.runner.isRunning {
            Button {
              if workspace.isGeneratingScore {
                workspace.showScoreGenerator()
              } else {
                workspace.selection = workspace.runner.lastSubmission?.kind
              }
            } label: {
              Label(
                workspace.isGeneratingScore ? "Generating score…" : "Workflow running",
                systemImage: "waveform")
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("workflow.running_destination")
          } else if workspace.hasUnreadScore {
            Button("New score ready") { workspace.showScoreGenerator() }
              .accessibilityIdentifier("score.ready_destination")
          } else {
            Label("Native YuE2 engine", systemImage: "circle")
              .foregroundStyle(.secondary)
          }
        }
        .font(.caption)
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
      }
    } detail: {
      if workspace.kind == .plan { scoreWorkspace } else { workflowWorkspace }
    }
    .navigationSplitViewStyle(.balanced)
    .navigationTitle("Yueqin")
    .toolbar {
      ToolbarItemGroup(placement: .primaryAction) {
        if workspace.runner.isRunning
          && !(workspace.isShowingScoreGeneration && workspace.isGeneratingScore)
        {
          Button("Cancel run", systemImage: "stop.fill") { workspace.runner.cancel() }
            .disabled(workspace.runner.cancellationRequested)
            .accessibilityIdentifier("workflow.cancel")
        }
        Button(workspace.primaryActionTitle, systemImage: workspace.primaryActionSymbol) {
          workspace.performPrimaryAction()
        }
        .buttonStyle(.borderedProminent)
        .labelStyle(.titleAndIcon)
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(workspace.primaryActionDisabled)
        .accessibilityIdentifier("workflow.primary_action")
        .help("\(workspace.primaryActionTitle) (Command–Return)")
      }
    }
    .frame(minWidth: 1020, minHeight: 680)
    .onChange(of: workspace.settings.snapshot) { _, _ in workspace.settings.save() }
    .onChange(of: workspace.draft.snapshot) { _, _ in workspace.draft.save() }
    .onChange(of: workspace.selection) { _, _ in
      workspace.inputError = nil
      if workspace.isShowingScoreGeneration { workspace.showScoreGenerator() }
    }
    .onChange(of: workspace.scoreView) { _, _ in workspace.scoreSession.command("stop") }
    .onChange(of: workspace.runner.completedRunID) { _, _ in workspace.receiveCompletedRun() }
  }

  private var scoreWorkspace: some View {
    VStack(spacing: 0) {
      HStack(spacing: 16) {
        Text("Scores").font(.title2.weight(.semibold))
        Picker("Score view", selection: $workspace.scoreView) {
          ForEach(ScoreWorkspaceView.allCases) { view in Text(view.title).tag(view) }
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .frame(width: 210)
        .accessibilityIdentifier("score.view")
        if workspace.hasUnreadScore {
          Button("New result") { workspace.showScoreGenerator() }
            .controlSize(.small)
        }
        Spacer(minLength: 8)
        Button("New blank score") { Task { await workspace.newScore() } }
          .accessibilityIdentifier("score.new")
        Button("Open ABC…") { Task { await workspace.openScore() } }
          .accessibilityIdentifier("score.open")
      }
      .padding(.horizontal, 20)
      .padding(.vertical, 14)
      .disabled(workspace.isAdoptingCandidate)
      if let error = workspace.inputError {
        errorBanner(error) { workspace.inputError = nil }
          .accessibilityIdentifier("workflow.input_error")
      }
      if let error = workspace.scoreDocument.errorMessage {
        errorBanner(error) { workspace.scoreDocument.errorMessage = nil }
          .accessibilityIdentifier("score.document_error")
      }
      Divider()
      if workspace.scoreView == .generate { scoreGeneration } else { scoreEditor }
    }
  }

  private var scoreGeneration: some View {
    HStack(spacing: 0) {
      VStack(alignment: .leading, spacing: 8) {
        HStack {
          Text("Generation settings").font(.headline)
          Spacer()
          if workspace.candidate != nil {
            Button("Generate another") { workspace.generateScore() }
              .disabled(
                workspace.runner.isRunning || workspace.isPreparingRun
                  || workspace.isAdoptingCandidate
              )
              .accessibilityIdentifier("score.generate_another")
          }
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
        Text("Create new music from your brief, not a revision of the notes in the editor.")
          .font(.callout).foregroundStyle(.secondary)
          .padding(.horizontal, 20)
        workflowForm
      }
      .frame(minWidth: 320, idealWidth: 350, maxWidth: 400, maxHeight: .infinity)
      Divider()
      VStack(alignment: .leading, spacing: 12) {
        if let error = workspace.scoreGenerationError {
          Label(error, systemImage: "exclamationmark.triangle")
            .font(.callout).foregroundStyle(.orange).textSelection(.enabled)
            .accessibilityIdentifier("score.generation_error")
          Text("Your working score is unchanged. Adjust the settings and try again.")
            .font(.caption).foregroundStyle(.secondary)
        }
        if workspace.isGeneratingScore {
          VStack(spacing: 14) {
            Text("Generating a new score").font(.title3.weight(.semibold))
            Text(
              workspace.runner.cancellationRequested
                ? "Waiting for generation to stop…" : workspace.runner.statusTitle
            )
            .foregroundStyle(.secondary)
            RunProgressView(progress: workspace.runner.progress)
              .frame(maxWidth: 320)
            if workspace.scoreDocument.hasDocument {
              Text("Your current score is unchanged.").foregroundStyle(.secondary)
            }
          }
          .frame(maxWidth: .infinity, maxHeight: .infinity)
          .accessibilityIdentifier("score.generation_progress")
        } else if let candidate = workspace.candidate {
          HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
              Text(
                workspace.scoreGenerationError == nil
                  ? "Generated result" : "Previous generated result"
              )
              .font(.headline)
              Text(
                workspace.candidateIsOpen
                  ? "Opened in Edit. Your edits are kept there."
                  : "Preview only — this has not replaced your working score."
              )
              .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            if workspace.scoreDocument.hasDocument && !workspace.candidateIsOpen {
              Button("Keep current") { workspace.showScoreEditor() }
                .accessibilityIdentifier("score.keep_current")
            }
          }
          if candidate.score.truncated {
            Label(
              "This result reached the token limit and may be incomplete.",
              systemImage: "exclamationmark.triangle"
            )
            .font(.caption).foregroundStyle(.orange)
          }
          ABCScoreView(abc: candidate.score.abc)
            .id(candidate.id)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityIdentifier("score.candidate")
          if workspace.scoreGenerationError != nil {
            Button(workspace.candidateActionTitle) { Task { await workspace.adoptCandidate() } }
              .disabled(workspace.isAdoptingCandidate)
          }
        } else {
          ContentUnavailableView {
            Label("Generate a starting point", systemImage: "music.note.list")
          } description: {
            Text(
              "Describe the song, then generate a score to review. Or open an ABC file or start with a blank score."
            )
          }
          .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        if workspace.runner.lastSubmission != nil {
          DisclosureGroup("Run details", isExpanded: $showsRunDetails) {
            results.frame(height: 230)
          }
          .font(.callout)
          .accessibilityIdentifier("score.run_details")
        }
      }
      .padding(16)
      .frame(minWidth: 400, maxWidth: .infinity, maxHeight: .infinity)
    }
    .accessibilityIdentifier("score.generation_view")
  }

  private var scoreEditor: some View {
    VStack(alignment: .leading, spacing: 12) {
      if workspace.scoreDocument.hasDocument {
        scoreDocumentHeader
        if workspace.scoreDocument.truncated {
          Label(
            "This score reached the token limit and may be incomplete.",
            systemImage: "exclamationmark.triangle"
          )
          .font(.caption).foregroundStyle(.orange)
        }
        ABCScoreView(abc: workspace.scoreDocument.abc, session: workspace.scoreSession)
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      } else {
        ContentUnavailableView {
          Label("Start your score", systemImage: "music.note.list")
        } description: {
          Text(
            "Write a blank score, open an ABC file, or generate a starting point. Editing does not require a model."
          )
        } actions: {
          Button("Generate a score") { workspace.showScoreGenerator() }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }
    .padding(16)
    .accessibilityIdentifier("score.editor_view")
  }

  private var scoreDocumentHeader: some View {
    HStack(spacing: 12) {
      VStack(alignment: .leading, spacing: 3) {
        Text(workspace.scoreDocument.title).font(.headline).lineLimit(1)
        Text(
          workspace.scoreDocument.isDirty
            ? "Unsaved changes"
            : workspace.scoreDocument.fileURL != nil
              ? "Saved"
              : workspace.scoreDocument.sourceURL != nil
                ? "Original preserved · save an edited copy" : "Not saved"
        )
        .font(.caption).foregroundStyle(.secondary)
      }
      Spacer()
      Button("Save", systemImage: "square.and.arrow.down") {
        Task { await workspace.scoreDocument.save() }
      }
      .disabled(workspace.scoreDocument.fileURL != nil && !workspace.scoreDocument.isDirty)
      .accessibilityIdentifier("score.save")
      Menu {
        Button("Save Score As…") { Task { await workspace.scoreDocument.save(as: true) } }
        Divider()
        Button("ABC Source") { workspace.scoreSession.command("source") }
        Button("Review Compatibility") { workspace.scoreSession.command("showCompatibility") }
        Button("Keyboard Reference") { workspace.scoreSession.command("help") }
      } label: {
        Image(systemName: "ellipsis.circle")
      }
      .accessibilityLabel("More score actions")
      .accessibilityIdentifier("score.more")
    }
  }

  private var workflowWorkspace: some View {
    HSplitView {
      VStack(alignment: .leading, spacing: 0) {
        VStack(alignment: .leading, spacing: 6) {
          Text(workspace.kind.title).font(.title.weight(.semibold))
          Text(workspace.kind.subtitle).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 24).padding(.top, 22).padding(.bottom, 12)
        if let error = workspace.inputError {
          errorBanner(error) { workspace.inputError = nil }
            .accessibilityIdentifier("workflow.input_error")
        }
        if workspace.kind == .generate {
          if workspace.usesEditorScore {
            scoreAttachment
          } else if workspace.scoreDocument.hasDocument {
            HStack {
              Text("Working score: \(workspace.scoreDocument.title)")
                .lineLimit(1).font(.callout).foregroundStyle(.secondary)
              Spacer()
              Button("Use this score") { Task { await workspace.createSong() } }
                .disabled(!workspace.scoreSession.canUseScore || workspace.runner.isRunning)
            }
            .padding(.horizontal, 24).padding(.bottom, 10)
          }
        }
        workflowForm
      }
      .frame(minWidth: 440, maxWidth: .infinity, maxHeight: .infinity)
      results.frame(minWidth: 320, idealWidth: 350, maxWidth: 440, maxHeight: .infinity)
    }
  }

  private var scoreAttachment: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Label(workspace.scoreDocument.title, systemImage: "music.note.list")
          .font(.headline).lineLimit(1)
        Spacer()
        Button("Edit…") { workspace.showScoreEditor() }
          .accessibilityIdentifier("song.edit_score")
        Button("Remove") { workspace.removeScoreAttachment() }
          .disabled(workspace.runner.isRunning || workspace.isPreparingRun)
          .accessibilityIdentifier("song.remove_score")
      }
      Text("Using your working score. Its exact contents are captured when you generate the song.")
        .font(.caption).foregroundStyle(.secondary)
      if !workspace.scoreSession.canUseScore {
        Label(
          "Review the score's compatibility issues before generating.",
          systemImage: "exclamationmark.triangle"
        )
        .font(.caption).foregroundStyle(.orange)
      }
      if workspace.recordingUsesEarlierScore {
        Label(
          "The recording uses an earlier score. Generate again to hear your changes.",
          systemImage: "arrow.triangle.2.circlepath"
        )
        .font(.caption).foregroundStyle(.secondary)
        .accessibilityIdentifier("song.earlier_score")
      }
    }
    .padding(12)
    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    .padding(.horizontal, 24).padding(.bottom, 10)
    .accessibilityIdentifier("song.score_attachment")
  }

  private var workflowForm: some View {
    WorkflowForm(
      kind: workspace.kind, draft: workspace.draft, settings: workspace.settings,
      usesEditorScore: workspace.kind == .generate && workspace.usesEditorScore,
      editorScoreABC: { workspace.scoreDocument.abc }
    )
    .disabled(workspace.runner.isRunning || workspace.isPreparingRun)
  }

  private var results: some View {
    RunResultsView(controller: workspace.runner) { kind, url in workspace.useOutput(url, for: kind)
    }
  }

  private func errorBanner(_ message: String, dismiss: @escaping () -> Void) -> some View {
    HStack(alignment: .top, spacing: 10) {
      Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
        .textSelection(.enabled)
      Spacer(minLength: 0)
      Button(action: dismiss) { Image(systemName: "xmark") }
        .buttonStyle(.plain).accessibilityLabel("Dismiss error")
    }
    .font(.callout).padding(.horizontal, 20).padding(.vertical, 10)
  }

  private func workflowSection(_ title: String, _ kinds: [WorkflowKind]) -> some View {
    Section(title) {
      ForEach(kinds) { kind in
        NavigationLink(value: kind) { Label(kind.title, systemImage: kind.symbol) }
          .accessibilityIdentifier("workflow.\(kind.rawValue)")
      }
    }
  }
}
