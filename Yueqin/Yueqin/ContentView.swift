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
      guard workspace.runner.lastSubmission?.kind == .prepare else { return }
      if let model = workspace.runner.preparedModelPath {
        workspace.settings.model = model
        workspace.settings.convertedDirectory = model
      }
      if let vae = workspace.runner.preparedVAEPath {
        workspace.settings.vae = vae
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
      switch workspace.runner.planScoreStatus {
      case .running:
        HStack(spacing: 10) {
          ProgressView().controlSize(.small)
          Text(
            workspace.runner.plannedScore == nil
              ? "Planning your score…"
              : "Planning a new score · showing the previous score")
        }
        .accessibilityIdentifier("plan.score_progress")
      case .failed(let message):
        VStack(alignment: .leading, spacing: 4) {
          Label(message, systemImage: "exclamationmark.triangle")
            .foregroundStyle(.orange)
            .textSelection(.enabled)
          if workspace.runner.plannedScore != nil {
            Text("The previous score is still shown below.").foregroundStyle(.secondary)
          }
        }
        .font(.callout)
        .accessibilityIdentifier("plan.score_error")
      case .empty, .ready:
        EmptyView()
      }

      if let score = workspace.runner.plannedScore {
        if score.truncated {
          Label(
            "This score reached the token limit and may be incomplete.",
            systemImage: "exclamationmark.triangle"
          )
          .font(.callout)
          .foregroundStyle(.orange)
          .accessibilityIdentifier("plan.score_truncated")
        }
        ABCScoreView(abc: score.abc)
          .frame(maxWidth: .infinity, maxHeight: .infinity)
      } else {
        ContentUnavailableView {
          Label("Your score", systemImage: "music.note.list")
        } description: {
          Text("Set the style and lyrics, then choose Plan score. The notation will appear here.")
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
