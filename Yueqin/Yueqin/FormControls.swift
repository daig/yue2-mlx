import AppKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct PathField: View {
  enum Selection {
    case file, directory, outputDirectory, report
  }

  let title: String
  @Binding var path: String
  let selection: Selection
  let identifier: String
  var prompt = "Absolute path or ~/…"
  var guidance = ""

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(alignment: .firstTextBaseline) {
        TextField(title, text: $path, prompt: Text(prompt))
          .accessibilityIdentifier(identifier)
          .help(
            guidance.isEmpty
              ? "Enter an absolute path, or choose one. Paths beginning with ~/ are supported."
              : guidance)
        Button("Choose…", action: choose)
          .accessibilityLabel("Choose \(title)")
          .accessibilityIdentifier(identifier + ".choose")
      }
      if !guidance.isEmpty {
        Text(guidance).font(.caption).foregroundStyle(.secondary)
      }
    }
  }

  private func choose() {
    switch selection {
    case .file, .directory:
      let panel = NSOpenPanel()
      panel.title = "Choose \(title)"
      panel.canChooseFiles = selection == .file
      panel.canChooseDirectories = selection == .directory
      panel.allowsMultipleSelection = false
      panel.canCreateDirectories = selection == .directory
      seed(panel)
      if panel.runModal() == .OK, let url = panel.url {
        path = url.path
      }
    case .outputDirectory, .report:
      let panel = NSSavePanel()
      panel.title = "Choose \(title)"
      panel.prompt = "Choose"
      panel.canCreateDirectories = true
      panel.nameFieldLabel = selection == .report ? "Report name:" : "Output folder name:"
      panel.message =
        selection == .report
        ? "The workflow will write the diagnostic report at this location."
        : "Choose a new or empty output folder. The workflow creates it when needed."
      if selection == .report { panel.allowedContentTypes = [.json] }
      seed(panel)
      if panel.runModal() == .OK, let url = panel.url {
        path = url.path
      }
    }
  }

  private func seed(_ panel: NSSavePanel) {
    guard !path.isEmpty else { return }
    let expanded = (path as NSString).expandingTildeInPath
    guard expanded.hasPrefix("/") else { return }
    let url = URL(fileURLWithPath: expanded)
    panel.directoryURL = url.deletingLastPathComponent()
    panel.nameFieldStringValue = url.lastPathComponent
  }
}

@MainActor
struct MultilineField: View {
  let title: String
  @Binding var text: String
  let identifier: String
  var guidance = ""
  var monospaced = false
  var height: CGFloat = 130

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      Text(title).font(.headline)
      TextEditor(text: $text)
        .font(monospaced ? .system(.body, design: .monospaced) : .body)
        .frame(minHeight: height, idealHeight: height)
        .scrollContentBackground(.visible)
        .overlay(
          RoundedRectangle(cornerRadius: 5).stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        )
        .accessibilityLabel(title)
        .accessibilityIdentifier(identifier)
        .help(guidance)
      if !guidance.isEmpty {
        Text(guidance).font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}

@MainActor
struct SamplingFields: View {
  let title: String
  @Binding var sampling: SamplingDraft
  let prefix: String
  let isABC: Bool

  var body: some View {
    DisclosureGroup(title) {
      Text(
        "Leave fields blank to use generation_config or engine defaults. Only named fields override this stage."
      )
      .font(.caption).foregroundStyle(.secondary)
      field(
        "Temperature", value: $sampling.temperature, key: "temperature",
        defaultValue: isABC ? "0.7" : "1.0",
        help: "Temperature from 0 through 5; 0 selects greedy decoding.")
      field(
        "Top p", value: $sampling.topP, key: "top_p", defaultValue: isABC ? "0.9" : "0.95",
        help: "Nucleus sampling probability, greater than 0 and at most 1.")
      field(
        "Top k", value: $sampling.topK, key: "top_k", defaultValue: isABC ? "30" : "100",
        help: "Positive integer candidate count, at least 1.")
      field(
        "Repetition penalty", value: $sampling.repetitionPenalty, key: "repetition_penalty",
        defaultValue: isABC ? "1.005" : "1.2",
        help: "Positive repetition penalty; 1 leaves logits unchanged.")
      field(
        "Penalty window", value: $sampling.penaltyWindow, key: "penalty_window",
        defaultValue: isABC ? "100" : "50",
        help: "Integer from 1 through 100: recent tokens considered for repetition penalties.")
      field(
        "Minimum tokens", value: $sampling.minTokens, key: "min_tokens",
        defaultValue: isABC ? "32" : "200",
        help: "Minimum token budget before an end token may be sampled.")
      field(
        "Maximum tokens", value: $sampling.maxTokens, key: "max_tokens",
        defaultValue: isABC ? "4096" : "9000",
        help:
          "Maximum token budget, not song duration. Exhaustion is reported as truncation; context overflow is an error."
      )
    }
  }

  private func field(
    _ label: String, value: Binding<String>, key: String, defaultValue: String, help: String
  ) -> some View {
    TextField(label, text: value, prompt: Text("Engine default: \(defaultValue)"))
      .accessibilityIdentifier(prefix + "." + key)
      .help(help)
  }
}
