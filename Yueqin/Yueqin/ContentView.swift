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

          WorkflowForm(kind: workspace.kind, draft: workspace.draft, settings: workspace.settings)
            .disabled(workspace.runner.isRunning)
        }
        .frame(minWidth: 440, maxWidth: .infinity, maxHeight: .infinity)

        RunResultsView(controller: workspace.runner) { kind, url in
          workspace.useOutput(url, for: kind)
        }
        .frame(minWidth: 310, idealWidth: 350, maxWidth: 440, maxHeight: .infinity)
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
