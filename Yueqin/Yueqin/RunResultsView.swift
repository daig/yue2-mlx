import AVFoundation
import AppKit
import Observation
import SwiftUI
import UniformTypeIdentifiers

@MainActor @Observable private final class RecordingPlayer {
  private var player: AVAudioPlayer?
  private(set) var url: URL?
  private(set) var error: String?
  private(set) var exporting = false
  var duration: Double { player?.duration ?? 0 }
  var position: Double { player?.currentTime ?? 0 }
  var actuallyPlaying: Bool { player?.isPlaying == true }

  func stop() {
    player?.stop()
    player = nil
    url = nil
    error = nil
  }
  func toggle(_ file: URL) {
    do {
      if url != file {
        stop()
        player = try AVAudioPlayer(contentsOf: file)
        url = file
      }
      guard let player else { return }
      if player.isPlaying {
        player.pause()
      } else {
        guard player.play() else {
          error = "The audio device could not start playback."
          return
        }
        error = nil
      }
    } catch { self.error = "Cannot play this FLAC: \(error.localizedDescription)" }
  }
  func seek(_ value: Double) { player?.currentTime = min(max(0, value), duration) }
  func export(_ file: URL) {
    let panel = NSSavePanel()
    panel.title = "Export original FLAC"
    panel.nameFieldStringValue = file.lastPathComponent
    panel.allowedContentTypes = [UTType(filenameExtension: "flac") ?? .audio]
    panel.canCreateDirectories = true
    panel.begin { [weak self] response in
      guard response == .OK, let target = panel.url, let self else { return }
      guard target.standardizedFileURL != file.standardizedFileURL else {
        self.error = "Choose a destination other than the original recording."
        return
      }
      self.exporting = true
      DispatchQueue.global(qos: .userInitiated).async {
        let failure: String?
        do {
          // Stage beside the destination so replacing an existing export
          // is atomic and never destroys it if copying fails.
          let temporary = target.deletingLastPathComponent().appendingPathComponent(
            ".yueqin-\(UUID().uuidString).flac")
          defer { try? FileManager.default.removeItem(at: temporary) }
          try FileManager.default.copyItem(at: file, to: temporary)
          if FileManager.default.fileExists(atPath: target.path) {
            _ = try FileManager.default.replaceItemAt(target, withItemAt: temporary)
          } else {
            try FileManager.default.moveItem(at: temporary, to: target)
          }
          failure = nil
        } catch { failure = "Export failed: \(error.localizedDescription)" }
        DispatchQueue.main.async { [weak self] in
          self?.exporting = false
          self?.error = failure
          if failure == nil { NSWorkspace.shared.activateFileViewerSelecting([target]) }
        }
      }
    }
  }
}

@MainActor struct RunResultsView: View {
  let controller: RunController
  let onUseOutput: (WorkflowKind, URL) -> Void
  @State private var player = RecordingPlayer()
  @State private var advancedOutputExpanded = false

  init(controller: RunController, onUseOutput: @escaping (WorkflowKind, URL) -> Void) {
    self.controller = controller
    self.onUseOutput = onUseOutput
  }

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        Text("Activity & Results").font(.title2).fontWeight(.semibold)
        if controller.lastSubmission == nil {
          ContentUnavailableView(
            "Ready to run", systemImage: "waveform",
            description: Text(
              "Run a workflow to inspect native progress, diagnostics and saved artifacts. Recordings can be played and exported here."
            ))
        } else {
          activity
          if let error = controller.errorMessage {
            Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
              .textSelection(.enabled)
              .accessibilityIdentifier("result.error")
          }
          if !controller.isRunning { resultDetails }
          if !controller.events.isEmpty {
            DisclosureGroup("Stages & notices (\(controller.events.count))") {
              LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(controller.events) { event in eventRow(event.value) }
              }.padding(.top, 6)
            }
          }
          advancedOutput
        }
      }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
    }
    .frame(minWidth: 320, idealWidth: 360)
    .accessibilityIdentifier("result.inspector")
    .onChange(of: controller.isRunning) { _, running in if running { player.stop() } }
    .onChange(of: controller.startedAt) { _, _ in advancedOutputExpanded = false }
    .onChange(of: controller.outputURL) { _, _ in player.stop() }
    .onDisappear { player.stop() }
  }

  private var activity: some View {
    VStack(alignment: .leading, spacing: 8) {
      if let submission = controller.lastSubmission {
        Text(submission.kind.title).font(.callout).foregroundStyle(.secondary)
      }
      Text(controller.statusTitle).font(.headline).accessibilityIdentifier("result.status")
      if let start = controller.startedAt {
        TimelineView(.periodic(from: .now, by: 1)) { context in
          let elapsed = (controller.endedAt ?? context.date).timeIntervalSince(start)
          Text("Elapsed: \(max(0, elapsed).formatted(.number.precision(.fractionLength(0)))) s")
            .monospacedDigit().foregroundStyle(.secondary)
        }
      }
      if controller.isRunning {
        if let complete = controller.progress["completed"].number,
          let total = controller.progress["total"].number, total > 0
        {
          ProgressView(value: min(complete, total), total: total)
        } else {
          ProgressView().controlSize(.small)
        }
        if let count = controller.progress["completed"].number {
          Text(
            "\(count.formatted()) \(controller.progress["unit"].text ?? "units") · stage \((controller.progress["elapsed_seconds"].number ?? 0).formatted(.number.precision(.fractionLength(1)))) s"
          )
          .font(.caption).monospacedDigit()
        }
        Button(
          controller.cancellationRequested ? "Cancellation requested" : "Cancel", role: .cancel
        ) { controller.cancel() }
        .disabled(controller.cancellationRequested).accessibilityIdentifier("result.cancel")
      }
    }
  }

  @ViewBuilder private var resultDetails: some View {
    let result = controller.result
    if result["truncated"].flag == true
      || result["truncated"].fields.values.contains(where: { $0.flag == true })
    {
      Label(
        "Generation reached a token limit. Completion does not mean the generated sequence reached its natural end.",
        systemImage: "exclamationmark.triangle"
      )
      .foregroundStyle(.orange).accessibilityIdentifier("result.truncated")
      Text("Truncation: \(result["truncated"].summary)").font(.caption)
    }
    if controller.lastSubmission?.kind == .doctor, controller.resultData != nil {
      Label(
        result["ready"].flag == true ? "Runtime ready" : "Runtime not ready",
        systemImage: result["ready"].flag == true ? "checkmark.circle" : "exclamationmark.triangle"
      )
      .accessibilityIdentifier("result.doctor_ready")
      if result["hashes_verified"].flag == true {
        Label("Model hashes verified", systemImage: "checkmark.shield")
      }
      Text("Readiness is not a quality or performance certification.")
        .font(.caption).foregroundStyle(.secondary)
      ForEach(result["errors"].fields.keys.sorted(), id: \.self) { key in
        Text("\(key): \(result["errors"][key].summary)")
          .font(.callout).foregroundStyle(.red).textSelection(.enabled)
      }
      ForEach(["versions", "runtime", "weights"], id: \.self) { key in
        if !result[key].fields.isEmpty {
          DisclosureGroup(key.capitalized) {
            Text(result[key].summary).font(.system(.caption, design: .monospaced))
              .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
          }
        }
      }
    }
    if controller.preparedModelPath != nil {
      Label("Generator prepared", systemImage: "checkmark.circle")
    }
    if controller.preparedVAEPath != nil {
      Label("VAE available", systemImage: "checkmark.circle")
    }
    if controller.lastSubmission?.kind == .batch, !result["results"].rows.isEmpty {
      Text(
        "Batch: \(result["results"].rows.count) recorded of \(result["expected"].summary); \(result["failed"].summary) failed"
      ).font(.headline)
      LazyVStack(alignment: .leading, spacing: 10) {
        ForEach(Array(result["results"].rows.enumerated()), id: \.offset) { _, row in
          VStack(alignment: .leading, spacing: 4) {
            Text("Line \(row["line"].summary) · \(row["id"].summary)").fontWeight(.medium)
            Text(row["status"].summary).foregroundStyle(
              row["status"].text == "failed" ? Color.red : Color.secondary)
            if let reason = row["reason"].text {
              Text("\(row["type"].summary): \(reason)").textSelection(.enabled)
            }
            if row["resumed"].flag == true {
              Text("Resumed existing recording").foregroundStyle(.secondary)
            }
          }.font(.caption)
        }
      }.accessibilityIdentifier("result.batch_rows")
    }
    if let seconds = result["audio_seconds"].number {
      Text(
        "Audio: \(seconds.formatted(.number.precision(.fractionLength(2)))) s · \(result["sample_rate"].summary) Hz"
      ).font(.caption)
    }
    if controller.succeeded, let output = controller.outputURL {
      if controller.lastSubmission?.kind == .plan {
        Button("Render saved plan", systemImage: "waveform") { onUseOutput(.renderPlan, output) }
          .help(
            "Renders the original saved plan, not editor changes. Use in song submits the current edited score."
          )
          .accessibilityIdentifier("result.render_plan")
      } else if let kind = controller.lastSubmission?.kind,
        [WorkflowKind.generate, .renderPlan, .replay].contains(kind)
      {
        Button("Replay recording", systemImage: "arrow.clockwise") { onUseOutput(.replay, output) }
          .accessibilityIdentifier("result.replay")
      }
    }
    if let error = controller.artifactError {
      Text("Some artifacts could not be listed: \(error)").foregroundStyle(.orange)
    }
    let recordings = controller.artifacts.filter { $0.pathExtension.lowercased() == "flac" }
    if !recordings.isEmpty {
      Divider()
      Text(controller.succeeded ? "Recordings" : "Available recordings").font(.headline)
      ForEach(Array(recordings.enumerated()), id: \.element) { index, audio in
        Text(recordings.count == 1 ? "Recording" : "Recording \(index + 1)").font(.subheadline)
        audioControls(audio)
      }
      if let error = player.error {
        Text(error).foregroundStyle(.red).textSelection(.enabled).accessibilityIdentifier(
          "playback.error")
      }
    }
  }

  private var advancedOutput: some View {
    DisclosureGroup("Advanced output", isExpanded: $advancedOutputExpanded) {
      VStack(alignment: .leading, spacing: 14) {
        if !controller.isRunning {
          if let output = controller.outputURL,
            controller.succeeded || !controller.artifacts.isEmpty
          {
            pathDetail("Output destination", output.path)
          }
          if let model = controller.preparedModelPath { pathDetail("Prepared generator", model) }
          if let vae = controller.preparedVAEPath { pathDetail("VAE", vae) }
          if !controller.artifacts.isEmpty {
            Text(controller.succeeded ? "Saved artifacts" : "Files at the destination")
              .font(.headline)
            DisclosureGroup("Files (\(controller.artifacts.count))") {
              LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(controller.artifacts, id: \.self) { file in
                  Button {
                    NSWorkspace.shared.activateFileViewerSelecting([file])
                  } label: {
                    Label(relativePath(file), systemImage: "doc")
                      .lineLimit(2).frame(maxWidth: .infinity, alignment: .leading)
                  }.buttonStyle(.link).help(file.path).accessibilityIdentifier(
                    "result.artifact.\(file.lastPathComponent)")
                }
              }.padding(.top, 6)
            }
          }
        }
        if let data = controller.resultData {
          jsonDisclosure("Exact result JSON", data: data, identifier: "result.json")
        }
        if let data = controller.partialBatchData {
          Text(
            "Batch receipt at the destination. It may include earlier attempts; this operation did not return a completed batch result."
          ).foregroundStyle(.secondary)
          jsonDisclosure(
            "Saved batch receipt JSON", data: data, identifier: "result.partial_batch")
        }
        if let submission = controller.lastSubmission {
          jsonDisclosure(
            "Exact submitted options", data: submission.options, identifier: "result.options")
        }
      }.padding(.top, 6)
    }.accessibilityIdentifier("result.advanced_output")
  }
  private func audioControls(_ audio: URL) -> some View {
    VStack(alignment: .leading, spacing: 8) {
      TimelineView(.periodic(from: .now, by: 0.25)) { _ in
        HStack {
          Button {
            player.toggle(audio)
          } label: {
            Label(
              player.url == audio && player.actuallyPlaying ? "Pause" : "Play",
              systemImage: player.url == audio && player.actuallyPlaying
                ? "pause.fill" : "play.fill")
          }.accessibilityIdentifier("playback.toggle")
          Button("Export FLAC…") { player.export(audio) }.disabled(player.exporting)
            .accessibilityIdentifier("playback.export")
        }
      }
      if player.url == audio {
        TimelineView(.periodic(from: .now, by: 0.25)) { _ in
          VStack(alignment: .leading) {
            Slider(
              value: Binding(get: { player.position }, set: { player.seek($0) }),
              in: 0...max(player.duration, 0.001)
            )
            .accessibilityLabel("Playback position").accessibilityIdentifier("playback.seek")
            Text(
              "\(player.position.formatted(.number.precision(.fractionLength(1)))) / \(player.duration.formatted(.number.precision(.fractionLength(1)))) s\(player.actuallyPlaying ? " · Playing" : " · Stopped or paused")"
            )
            .font(.caption).monospacedDigit()
          }
        }
      }
    }
  }

  private func relativePath(_ file: URL) -> String {
    guard let root = controller.outputURL, file.path.hasPrefix(root.path + "/") else {
      return file.lastPathComponent
    }
    return String(file.path.dropFirst(root.path.count + 1))
  }

  private func pathDetail(_ title: String, _ path: String) -> some View {
    VStack(alignment: .leading) {
      Text(title).font(.headline)
      Button(path) { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
        .buttonStyle(.link).textSelection(.enabled)
        .lineLimit(3).truncationMode(.middle).help(path)
    }
  }

  private func eventRow(_ event: RunJSON) -> some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(event["type"].summary.replacingOccurrences(of: "_", with: " ").capitalized).fontWeight(
        .medium)
      Text(event.summary).textSelection(.enabled)
    }.font(.caption)
      .foregroundStyle(
        ["warning", "failure"].contains(event["type"].text ?? "") ? Color.orange : Color.secondary)
  }

  private func jsonDisclosure(_ title: String, data: Data, identifier: String) -> some View {
    DisclosureGroup(title) {
      ScrollView([.horizontal, .vertical]) {
        Text(String(decoding: data, as: UTF8.self))
          .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
          .frame(maxWidth: .infinity, alignment: .leading)
      }.frame(height: 240)
    }.accessibilityIdentifier(identifier)
  }
}
