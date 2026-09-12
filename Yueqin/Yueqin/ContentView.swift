import AppKit
import SwiftUI

@MainActor
struct ContentView: View {
  @Bindable var workspace: YueqinWorkspace

  var body: some View {
    NavigationSplitView {
      List(selection: $workspace.selection) {
        workflowSection("Create", [.generate, .plan, .renderPlan])
        workflowSection("Work with artifacts", [.replay, .batch])
        workflowSection("Setup", [.prepare, .doctor])
      }
      .listStyle(.sidebar)
      .navigationSplitViewColumnWidth(min: 175, ideal: 195, max: 240)
      .safeAreaInset(edge: .bottom) {
        HStack(spacing: 8) {
          Image(systemName: workspace.runner.isRunning ? "waveform" : "circle")
            .foregroundStyle(workspace.runner.isRunning ? Color.accentColor : .secondary)
          Text(workspace.runner.isRunning ? "Workflow running" : "Native YuE2 engine")
            .font(.caption)
            .foregroundStyle(.secondary)
          Spacer(minLength: 0)
        }
        .padding(16)
      }
    } detail: {
      GeometryReader { detail in
        HSplitView {
          if workspace.kind != .plan || workspace.showPlanningControls {
            VStack(alignment: .leading, spacing: 0) {
              VStack(alignment: .leading, spacing: 6) {
                Text(workspace.kind.title)
                  .font(.title.weight(.semibold))
                  .accessibilityAddTraits(.isHeader)
                Text(workspace.kind.subtitle)
                  .foregroundStyle(.secondary)
                  .fixedSize(horizontal: false, vertical: true)
              }
              .frame(maxWidth: .infinity, alignment: .leading)
              .padding(.horizontal, 24)
              .padding(.top, 22)
              .padding(.bottom, 8)

              if let error = workspace.inputError {
                HStack(alignment: .top, spacing: 10) {
                  Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                  Text(error).textSelection(.enabled)
                  Spacer(minLength: 0)
                  Button {
                    workspace.inputError = nil
                  } label: {
                    Image(systemName: "xmark")
                  }
                  .buttonStyle(.plain)
                  .accessibilityLabel("Dismiss input error")
                }
                .font(.callout)
                .padding(12)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 24)
                .padding(.vertical, 8)
                .accessibilityIdentifier("workflow.input_error")
              }

              if workspace.kind == .plan {
                GeometryReader { geometry in
                  VSplitView {
                    workflowForm
                      .frame(width: geometry.size.width)
                      .frame(minHeight: 220, idealHeight: 400, maxHeight: .infinity)
                    results
                      .frame(width: geometry.size.width)
                      .frame(minHeight: 160, idealHeight: 220, maxHeight: .infinity)
                  }
                  .frame(width: geometry.size.width, height: geometry.size.height)
                }
              } else {
                workflowForm
              }
            }
            .frame(
              minWidth: workspace.kind == .plan ? 340 : 440,
              idealWidth: workspace.kind == .plan ? 380 : nil,
              maxWidth: workspace.kind == .plan ? 400 : .infinity, maxHeight: .infinity)
          }

          if workspace.kind == .plan {
            planningScore
              .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)
          } else {
            results
              .frame(minWidth: 310, idealWidth: 350, maxWidth: 440, maxHeight: .infinity)
          }
        }
        .frame(width: detail.size.width, height: detail.size.height)
      }
    }
    .navigationSplitViewStyle(.balanced)
    .navigationTitle("Yueqin")
    .toolbar {
      if workspace.kind == .plan {
        ToolbarItem(placement: .navigation) {
          Button("Generation", systemImage: "sidebar.left") {
            workspace.showPlanningControls.toggle()
          }
          .help("Show or hide generation settings and activity")
          .accessibilityIdentifier("score.generation_controls")
        }
      }
      ToolbarItemGroup(placement: .primaryAction) {
        if workspace.runner.isRunning {
          Button("Cancel", systemImage: "stop.fill") {
            workspace.runner.cancel()
          }
          .accessibilityIdentifier("workflow.cancel")
          .help("Request cancellation at the next engine checkpoint (Command–Period)")
        }
        Button(
          workspace.kind.actionTitle,
          systemImage: workspace.kind == .doctor ? "stethoscope" : "play.fill"
        ) {
          workspace.runActive()
        }
        .buttonStyle(.borderedProminent)
        .labelStyle(.titleAndIcon)
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(workspace.runner.isRunning)
        .accessibilityIdentifier("workflow.run")
        .help("Run this workflow (Command–Return)")
      }
    }
    .frame(minWidth: 1020, minHeight: 680)
    .onChange(of: workspace.settings.snapshot) { _, _ in workspace.settings.save() }
    .onChange(of: workspace.draft.snapshot) { _, _ in workspace.draft.save() }
    .onChange(of: workspace.selection) { _, _ in workspace.inputError = nil }
    .onChange(of: workspace.runner.completedRunID) { _, _ in
      if workspace.runner.lastSubmission?.kind == .plan,
        workspace.runner.planScoreStatus == .ready,
        let score = workspace.runner.plannedScore
      {
        Task { await workspace.receivePlannedScore(score) }
      } else if workspace.runner.lastSubmission?.kind == .prepare {
        if let model = workspace.runner.preparedModelPath {
          workspace.settings.model = model
          workspace.settings.convertedDirectory = model
        }
        if let vae = workspace.runner.preparedVAEPath {
          workspace.settings.vae = vae
        }
      }
    }
  }

  private var workflowForm: some View {
    WorkflowForm(kind: workspace.kind, draft: workspace.draft, settings: workspace.settings)
      .disabled(workspace.runner.isRunning)
  }

  private var results: some View {
    RunResultsView(controller: workspace.runner) { kind, url in
      workspace.useOutput(url, for: kind)
    }
  }

  private var planningScore: some View {
    VStack(alignment: .leading, spacing: 12) {
      scoreDocumentHeader
      if let message = workspace.scoreDocument.errorMessage {
        HStack(alignment: .top) {
          Label(message, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
            .textSelection(.enabled)
          Spacer(minLength: 0)
          Button {
            workspace.scoreDocument.errorMessage = nil
          } label: {
            Image(systemName: "xmark")
          }
          .buttonStyle(.plain)
          .accessibilityLabel("Dismiss score error")
        }
        .font(.callout)
        .accessibilityIdentifier("score.document_error")
      }
      if let pending = workspace.pendingPlannedScore {
        HStack(alignment: .top) {
          Text("A new plan is ready. Your current score and edits have been kept.")
            .foregroundStyle(.secondary)
          Button("Open new plan") { Task { await workspace.openPlannedScore(pending) } }
        }
        .font(.callout)
        .accessibilityIdentifier("score.pending_plan")
      }
      switch workspace.runner.planScoreStatus {
      case .running:
        HStack(spacing: 10) {
          ProgressView().controlSize(.small)
          Text(
            workspace.scoreDocument.hasDocument
              ? "Planning a new score · keep editing while it runs"
              : "Planning your score…")
        }
        .accessibilityIdentifier("plan.score_progress")
      case .failed(let message):
        VStack(alignment: .leading, spacing: 4) {
          Label(message, systemImage: "exclamationmark.triangle")
            .foregroundStyle(.orange)
            .textSelection(.enabled)
          if workspace.scoreDocument.hasDocument {
            Text("Your score and edits are still shown below.").foregroundStyle(.secondary)
          }
        }
        .font(.callout)
        .accessibilityIdentifier("plan.score_error")
      case .empty, .ready:
        EmptyView()
      }

      if workspace.scoreDocument.hasDocument {
        if workspace.scoreDocument.truncated {
          Label(
            "This score reached the token limit and may be incomplete.",
            systemImage: "exclamationmark.triangle"
          )
          .font(.callout)
          .foregroundStyle(.orange)
          .accessibilityIdentifier("plan.score_truncated")
        }
        ABCScoreView(abc: workspace.scoreDocument.abc, session: workspace.scoreSession)
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      } else {
        ContentUnavailableView {
          Label("Your score", systemImage: "music.note.list")
        } description: {
          Text(
            "Generate a score, open an ABC file, or start a new score. Editing works without loading a model."
          )
        } actions: {
          HStack {
            Button("New score") { Task { await workspace.newScore() } }
              .accessibilityIdentifier("score.new")
            Button("Open ABC…") { Task { await workspace.openScore() } }
              .accessibilityIdentifier("score.open")
          }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
        .overlay(
          RoundedRectangle(cornerRadius: 6)
            .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
        .accessibilityIdentifier("plan.score_empty")
      }
    }
    .padding(16)
  }

  private var scoreDocumentHeader: some View {
    HStack(spacing: 10) {
      VStack(alignment: .leading, spacing: 2) {
        Text(
          workspace.scoreDocument.hasDocument ? workspace.scoreDocument.title : "Score workspace"
        )
        .font(.headline)
        .lineLimit(1)
        .truncationMode(.middle)
        Text(workspace.scoreDocument.isDirty ? "Edited · not saved" : "ABC notation")
          .font(.caption)
          .foregroundStyle(workspace.scoreDocument.isDirty ? .primary : .secondary)
      }
      Spacer(minLength: 0)
      Menu {
        Button("New Score") { Task { await workspace.newScore() } }
        Button("Open ABC Score…") { Task { await workspace.openScore() } }
        Button("Save Score As…") { Task { await workspace.scoreDocument.save(as: true) } }
          .disabled(!workspace.scoreDocument.hasDocument)
        Divider()
        Button("ABC Source") { workspace.scoreSession.command("source") }
          .disabled(!workspace.scoreDocument.hasDocument)
        Button("Keyboard Reference") { workspace.scoreSession.command("help") }
          .disabled(!workspace.scoreDocument.hasDocument)
      } label: {
        Image(systemName: "doc.badge.plus")
      }
      .help("New, open and save score")
      .accessibilityLabel("Score file actions")
      .accessibilityIdentifier("score.file_actions")
      Button {
        Task { await workspace.scoreDocument.save() }
      } label: {
        Image(systemName: "square.and.arrow.down")
      }
      .disabled(!workspace.scoreDocument.hasDocument)
      .help("Save edited ABC (Command–S)")
      .accessibilityLabel("Save score")
      .accessibilityIdentifier("score.save")
      Button("Use in song") { Task { await workspace.useEditorScore() } }
        .disabled(workspace.runner.isRunning || !workspace.scoreSession.canUseScore)
        .help(
          "Copy this score into Generate song with the current planning request; review settings before generating audio"
        )
        .accessibilityIdentifier("score.use_in_song")
    }
    .controlSize(.small)
  }

  private func workflowSection(_ title: String, _ kinds: [WorkflowKind]) -> some View {
    Section(title) {
      ForEach(kinds) { kind in
        NavigationLink(value: kind) {
          Label(kind.title, systemImage: kind.symbol)
        }
        .accessibilityIdentifier("workflow.\(kind.rawValue)")
      }
    }
  }
}
