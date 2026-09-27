import AppKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct WorkflowForm: View {
  let kind: WorkflowKind
  @Bindable var draft: WorkflowDraft
  @Bindable var settings: EngineSettings
  let usesEditorScore: Bool
  let editorScoreABC: (() -> String?)?
  @State private var exportError: String?

  init(
    kind: WorkflowKind, draft: WorkflowDraft, settings: EngineSettings,
    usesEditorScore: Bool = false, editorScoreABC: (() -> String?)? = nil
  ) {
    self.kind = kind
    self.draft = draft
    self.settings = settings
    self.usesEditorScore = usesEditorScore
    self.editorScoreABC = editorScoreABC
  }

  var body: some View {
    Form {
      switch kind {
      case .generate, .plan:
        requestSections
      case .renderPlan:
        Section("Saved plan") {
          PathField(
            title: "Plan folder", path: $draft.inputPath, selection: .directory,
            identifier: "workflow.input",
            guidance:
              "Choose the intact folder containing plan.json and plan_manifest.json. Saved ABC token IDs and generation settings are restored exactly."
          )
          Text(
            "To change the music, open the score in Scores, edit it, then choose Create song. Keep integrity-checked saved plan artifacts unchanged."
          )
          .font(.caption).foregroundStyle(.secondary)
        }
      case .replay:
        Section("Saved recording") {
          PathField(
            title: "Artifact folder", path: $draft.inputPath, selection: .directory,
            identifier: "workflow.input",
            guidance:
              "Choose an intact saved-song folder. Replay retains the saved request, generation settings and model identities."
          )
          Picker("Restart stage", selection: $draft.replayStage) {
            Text("Decode retained latents").tag("decode")
            Text("Synthesize retained semantic tokens").tag("synthesize")
          }
          .accessibilityIdentifier("workflow.stage")
          .help(
            "Decode runs only the VAE. Synthesize reruns the acoustic solver and decoder using retained semantic tokens and original noise, or saved-seed noise when older artifacts lack it."
          )
        }
      case .batch:
        Section("Serial batch") {
          PathField(
            title: "Requests JSONL", path: $draft.inputPath, selection: .file,
            identifier: "workflow.input",
            guidance:
              "One JSON request per nonempty line, each with a unique filename-safe id. Relative lyric and ABC paths resolve beside this file. Rows run serially; individual failures remain visible in the batch result."
          )
          Picker("Score mode override", selection: $draft.batchMode) {
            Text("Use each request’s mode").tag("")
            Text("Full · melody and chords").tag("full")
            Text("Melody only").tag("melody")
            Text("Off · no score").tag("off")
          }
          .accessibilityIdentifier("batch.cot")
          .help(
            "An explicit mode replaces cot for every batch row. Other request and sampling fields remain in the JSONL file."
          )
        }
      case .prepare:
        Section("Model preparation") {
          PathField(
            title: "Source", path: $draft.sourcePath, selection: .directory,
            identifier: "prepare.source", prompt: "Use downloaded pinned generator",
            guidance:
              "Optional absolute local source folder. Leave blank to convert the downloaded pinned generator. Native preparation also resolves the pinned VAE through its cache."
          )
          PathField(
            title: "Download cache", path: $draft.cachePath, selection: .directory,
            identifier: "prepare.cache", prompt: "Default native cache",
            guidance: "Optional cache directory. Offline mode uses only locally available files.")
          Text(
            "Preparation downloads or converts the real model assets. Returned generator and VAE paths become available for subsequent workflows."
          )
          .font(.caption).foregroundStyle(.secondary)
        }
      case .doctor:
        Section("Diagnostics") {
          Toggle("Verify local model and VAE hashes", isOn: $draft.verifyHashes)
            .accessibilityIdentifier("doctor.verify_hashes")
            .help(
              "Requires local converted-model and VAE folders in engine settings. Diagnostics never downloads models."
            )
          Text(
            "Checks native dependencies, OS and architecture, Metal backends and unsafe environment settings. Readiness is not a quality or performance certification."
          )
          .font(.caption).foregroundStyle(.secondary)
        }
      }
      outputSection
      engineSection
    }
    .formStyle(.grouped)
    .alert(
      "Could not export request",
      isPresented: Binding(get: { exportError != nil }, set: { if !$0 { exportError = nil } })
    ) {
      Button("OK", role: .cancel) { exportError = nil }
    } message: {
      Text(exportError ?? "")
    }
  }

  @ViewBuilder
  private var requestSections: some View {
    Section("Shared song brief") {
      Text("Style and lyrics are shared between Scores and song setup.")
        .font(.caption).foregroundStyle(.secondary)
      Picker("Request source", selection: $draft.request.source) {
        Text("Compose").tag("compose")
        Text("JSON file").tag("file")
      }
      .pickerStyle(.segmented)
      .accessibilityIdentifier("request.source")
      .help(
        "Compose a shared brief or use a JSON file with explicit overrides. Score generation and an attached editor score always replace the file’s score and conditioning."
      )
      if draft.request.source == "file" {
        PathField(
          title: "Request JSON", path: $draft.request.filePath, selection: .file,
          identifier: "request.file",
          guidance:
            "Submitted by path, never imported or rewritten. Relative lyrics_path and abc_path resolve beside the request file. Edit nested sampling and generation_config in that file."
        )
      } else {
        styleField
        lyricsFields
      }
    }
    if draft.request.source == "file" {
      Section("Explicit file overrides") {
        Text(
          "Enabled overrides replace file values. Shared style and lyrics apply in both workflows. Score generation and an editor attachment always replace the score and mode."
        )
        .font(.caption).foregroundStyle(.secondary)
        override(
          "Override style", enabled: $draft.request.overrideStyle,
          identifier: "request.override.style"
        ) { styleField }
        override(
          "Override lyrics", enabled: $draft.request.overrideLyrics,
          identifier: "request.override.lyrics"
        ) { lyricsFields }
        override(
          "Override identifier", enabled: $draft.request.overrideID,
          identifier: "request.override.id"
        ) { idField }
        override(
          "Override seed", enabled: $draft.request.overrideSeed, identifier: "request.override.seed"
        ) { seedField }
        if kind == .plan || usesEditorScore {
          modeField
        } else {
          override(
            "Override score mode", enabled: $draft.request.overrideCot,
            identifier: "request.override.cot"
          ) { modeField }
        }
        override(
          "Override CFG scale", enabled: $draft.request.overrideCFG,
          identifier: "request.override.cfg_scale"
        ) { cfgField }
        if kind != .plan && !usesEditorScore {
          override(
            "Override ABC score", enabled: $draft.request.overrideABC,
            identifier: "request.override.abc"
          ) {
            Picker("ABC override source", selection: $draft.request.scoreSource) {
              Text("ABC file").tag("file")
              Text("ABC text").tag("text")
            }
            .accessibilityIdentifier("request.abc_override_source")
            if draft.request.scoreSource == "text" {
              MultilineField(
                title: "ABC score", text: $draft.request.abc, identifier: "request.abc",
                guidance:
                  "Exact score snapshot, including editor changes. Use full or melody mode.",
                monospaced: true)
            } else {
              PathField(
                title: "ABC score", path: $draft.request.abcPath, selection: .file,
                identifier: "request.abc",
                guidance: "Supplied UTF-8 ABC is preserved exactly. Incompatible with cot = off.")
            }
          }
        }
      }
    } else {
      Section("Score") {
        modeField
        if kind != .plan && !usesEditorScore {
          Picker("Score source", selection: $draft.request.scoreSource) {
            Text("Generate").tag("generate")
            Text("ABC text").tag("text")
            Text("ABC file").tag("file")
          }
          .accessibilityIdentifier("request.score_source")
          .help(
            "Generate ABC, or supply exact UTF-8 ABC. Supplied ABC is invalid when score mode is off."
          )
          if draft.request.scoreSource == "text" {
            MultilineField(
              title: "ABC score", text: $draft.request.abc, identifier: "request.abc",
              guidance: "Preserved as entered; choose full or melody to use a supplied score.",
              monospaced: true)
          } else if draft.request.scoreSource == "file" {
            PathField(
              title: "ABC score", path: $draft.request.abcPath, selection: .file,
              identifier: "request.abc",
              guidance: "The UTF-8 file is read by the engine without rewriting its score.")
          }
        }
      }
      Section("Request identity") {
        idField
        seedField
      }
      Section {
        DisclosureGroup("Advanced generation") {
          cfgField
          SamplingFields(
            title: "ABC sampling", sampling: $draft.request.abcSampling, prefix: "request.abc",
            isABC: true)
          SamplingFields(
            title: "Semantic sampling", sampling: $draft.request.semanticSampling,
            prefix: "request.semantic", isABC: false)
          TextField(
            "Solver steps", text: $draft.request.odeSteps, prompt: Text("Engine default: 32")
          )
          .accessibilityIdentifier("request.ode_steps")
          .help(
            "Positive integer ode_steps. Midpoint uses two velocity evaluations per step. Blank preserves raw generation_config or engine defaults."
          )
          DisclosureGroup("Raw generation_config JSON") {
            MultilineField(
              title: "Generation configuration", text: $draft.request.generationJSON,
              identifier: "request.generation_config",
              guidance:
                "Optional JSON object; preserves all native fields including version, context and ode_method. Explicit solver steps override ode_steps; named stage sampling fields take precedence for their stage. The native protocol fixes context to 24576 and ode_method to midpoint.",
              monospaced: true, height: 150)
          }
        }
      }
      Section {
        Button("Export Request JSON…", action: exportRequest)
          .accessibilityIdentifier("request.export")
          .help(
            "Save the effective request as JSON, including the attached score snapshot when present. File-sourced lyrics and scores remain absolute path references; this does not run the workflow."
          )
      }
    }
  }

  private var styleField: some View {
    TextField(
      "Style", text: $draft.request.style,
      prompt: Text("Language, genre, instrumentation, vocal style…"), axis: .vertical
    )
    .lineLimit(2...4)
    .accessibilityIdentifier("request.style")
    .help(
      "Raw style/tags text passed to the native request, without generated presets or transformations."
    )
  }

  @ViewBuilder
  private var lyricsFields: some View {
    Picker("Lyrics source", selection: $draft.request.lyricsSource) {
      Text("Text").tag("text")
      Text("UTF-8 file").tag("file")
    }
    .accessibilityIdentifier("request.lyrics_source")
    if draft.request.lyricsSource == "file" {
      PathField(
        title: "Lyrics file", path: $draft.request.lyricsPath, selection: .file,
        identifier: "request.lyrics",
        guidance:
          "Choose a UTF-8 lyrics file. Explicit file input replaces the original request’s lyrics.")
    } else {
      MultilineField(
        title: "Lyrics", text: $draft.request.lyrics, identifier: "request.lyrics",
        guidance: "Enter the actual lyrics, including section labels such as [Verse] and [Chorus].")
    }
  }

  private var idField: some View {
    TextField("Identifier", text: $draft.request.id, prompt: Text("song"))
      .accessibilityIdentifier("request.id")
      .help("Filename-safe request identifier; do not include path separators.")
  }

  private var seedField: some View {
    TextField("Seed", text: $draft.request.seed, prompt: Text("Engine default"))
      .accessibilityIdentifier("request.seed")
      .help(
        "Exact integer from 0 through 9223372036854775807. Leave blank to retain the native default; no floating-point conversion is used."
      )
  }

  private var modeField: some View {
    Picker(
      kind == .plan || usesEditorScore ? "Score conditioning" : "Score mode (cot)",
      selection: Binding(
        get: {
          (kind == .plan || usesEditorScore) && draft.request.cot == "off"
            ? "full" : draft.request.cot
        },
        set: { draft.request.cot = $0 }
      )
    ) {
      Text("Full · melody and chords").tag("full")
      Text("Melody only").tag("melody")
      if kind != .plan && !usesEditorScore {
        Text("Off · no score").tag("off")
      }
    }
    .accessibilityIdentifier("request.cot")
    .help(
      kind == .plan || usesEditorScore
        ? "Full conditions on melody and chords. Melody leaves accompaniment free."
        : "Full plans melody and chords. Melody leaves accompaniment free. Off generates semantic music directly and rejects supplied ABC."
    )
  }

  private var cfgField: some View {
    TextField(
      "CFG scale", text: $draft.request.cfgScale, prompt: Text("Full/melody: 1.0 · off: 1.01")
    )
    .accessibilityIdentifier("request.cfg_scale")
    .help(
      "Semantic guidance scale in [0, 20]. ABC planning has no CFG. Blank preserves the native mode-specific default."
    )
  }

  private func override<Content: View>(
    _ title: String, enabled: Binding<Bool>, identifier: String, @ViewBuilder content: () -> Content
  ) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      Toggle(title, isOn: enabled).accessibilityIdentifier(identifier)
      if enabled.wrappedValue { content() }
    }
  }

  private var outputSection: some View {
    Section(kind == .doctor ? "Report" : "Output") {
      if kind == .doctor {
        PathField(
          title: "Report JSON", path: $draft.outputPath, selection: .report,
          identifier: "workflow.output",
          prompt: "Optional · results always appear in the inspector",
          guidance:
            "Optional report file. Choose a writable location; an existing report may be replaced.")
      } else {
        PathField(
          title: kind == .prepare ? "Converted model folder" : "Output folder",
          path: $draft.outputPath, selection: draft.resume ? .directory : .outputDirectory,
          identifier: "workflow.output",
          prompt: outputPrompt,
          guidance: kind == .prepare
            ? "Leave blank to use the converted-model directory in engine settings."
            : "Use a new or empty directory. Existing recordings are never silently mixed or overwritten."
        )
      }
      if kind == .generate || kind == .batch {
        Toggle("Resume matching output", isOn: $draft.resume)
          .accessibilityIdentifier("workflow.resume")
          .help(
            "Reuse a matching completed result or retry an interrupted/failed run. Request, configuration and model identities must match; changed inputs require a new output folder."
          )
      }
    }
  }

  private var outputPrompt: String {
    switch kind {
    case .prepare: "Use converted-model directory"
    default: "New folder under output root"
    }
  }

  private var engineSection: some View {
    Section {
      DisclosureGroup("Engine settings") {
        if kind != .prepare {
          PathField(
            title: "Model", path: $settings.model, selection: .directory,
            identifier: "engine.model", prompt: "Repository ID or absolute folder",
            guidance:
              "Converted model folder, portable model folder, or native-supported repository ID.")
        }
        if kind != .prepare {
          PathField(
            title: "VAE", path: $settings.vae, selection: .directory, identifier: "engine.vae",
            prompt: "Repository ID or absolute folder",
            guidance:
              "Explicit decoder identity. Offline and hash verification require local assets.")
        }
        PathField(
          title: "Converted-model directory", path: $settings.convertedDirectory,
          selection: .directory, identifier: "engine.converted_dir",
          guidance: "Native conversion location; also the default output for Prepare models.")
        Picker("Precision", selection: $settings.precision) {
          Text("BF16").tag("bf16")
          Text("8-bit · experimental").tag("8bit")
          Text("4-bit · experimental").tag("4bit")
        }
        .accessibilityIdentifier("engine.precision")
        .help(
          "BF16 is the MVP default. Quantized variants have not completed matched listening assessment."
        )
        if kind != .doctor {
          Toggle("Offline · local assets only", isOn: $settings.offline)
            .accessibilityIdentifier("engine.offline")
            .help("Do not download models; resolution must succeed from local files and cache.")
        }
        if kind != .prepare && kind != .doctor {
          Toggle("Require AC power", isOn: $settings.requireAC)
            .accessibilityIdentifier("engine.require_ac")
            .help(
              "Reject a run that starts without AC power or loses AC power. No application memory cap is imposed."
            )
          TextField(
            "VAE core frames", text: $settings.vaeCoreFrames, prompt: Text("Engine default: 256")
          )
          .accessibilityIdentifier("engine.vae_core_frames")
          .help(
            "Positive integer decoder tile core size; the native halo and exact crop remain unchanged."
          )
          PathField(
            title: "Default output root", path: $settings.outputRoot, selection: .directory,
            identifier: "engine.output_root",
            guidance:
              "Used for generated default output paths. An explicit workflow output takes precedence."
          )
        }
        Text(
          "Settings are shared across workflows and saved locally. Filesystem paths must be absolute; ~/ is expanded. Engine errors appear in the activity inspector with their native details."
        )
        .font(.caption).foregroundStyle(.secondary)
      }
    }
  }

  private func exportRequest() {
    do {
      let scoreABC = usesEditorScore ? editorScoreABC?() : nil
      guard !usesEditorScore || scoreABC != nil else {
        exportError =
          "The attached editor score is unavailable. Return to Scores and attach it again."
        return
      }
      let data = try draft.requestData(scoreABC: scoreABC)
      let panel = NSSavePanel()
      panel.title = "Export Request JSON"
      panel.allowedContentTypes = [.json]
      panel.canCreateDirectories = true
      panel.nameFieldStringValue = "request.json"
      guard panel.runModal() == .OK, let url = panel.url else { return }
      do {
        try data.write(to: url, options: .atomic)
      } catch {
        exportError =
          "Could not write \(url.path). Choose a writable location and try again.\n\n\(error.localizedDescription)"
      }
    } catch {
      exportError = "Correct the request fields before exporting.\n\n\(error.localizedDescription)"
    }
  }
}
