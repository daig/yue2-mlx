import Foundation
import LyraCore
import Observation

enum WorkflowKind: String, CaseIterable, Identifiable, Sendable {
  case generate, plan, renderPlan, replay, batch, prepare, doctor

  var id: String { rawValue }
  var title: String {
    switch self {
    case .generate: "Generate song"
    case .plan: "Plan score"
    case .renderPlan: "Render plan"
    case .replay: "Replay artifacts"
    case .batch: "Batch"
    case .prepare: "Prepare models"
    case .doctor: "Diagnostics"
    }
  }
  var subtitle: String {
    switch self {
    case .generate: "Generate a song from style, lyrics and an optional score."
    case .plan: "Generate and review the score before rendering audio."
    case .renderPlan: "Render audio from a saved plan directory."
    case .replay: "Decode or synthesize a saved song's artifacts."
    case .batch: "Run requests from a JSONL file serially."
    case .prepare: "Download pinned models and convert generator weights."
    case .doctor: "Check model readiness without downloading."
    }
  }
  var actionTitle: String {
    switch self {
    case .generate: "Generate"
    case .plan: "Plan score"
    case .renderPlan: "Render"
    case .replay: "Replay"
    case .batch: "Run batch"
    case .prepare: "Prepare"
    case .doctor: "Check readiness"
    }
  }
  var symbol: String {
    switch self {
    case .generate: "music.note"
    case .plan: "music.quarternote.3"
    case .renderPlan: "waveform"
    case .replay: "arrow.clockwise"
    case .batch: "list.bullet.rectangle"
    case .prepare: "arrow.down.circle"
    case .doctor: "stethoscope"
    }
  }
  var operation: LyraCore.Workflow {
    switch self {
    case .generate: .generate
    case .plan: .plan
    case .renderPlan: .renderPlan
    case .replay: .replay
    case .batch: .batch
    case .prepare: .prepare
    case .doctor: .doctor
    }
  }
}

struct SamplingDraft: Codable, Equatable, Sendable {
  var temperature = ""
  var topP = ""
  var topK = ""
  var repetitionPenalty = ""
  var penaltyWindow = ""
  var minTokens = ""
  var maxTokens = ""

  fileprivate func object(stage: String) throws -> JSONObject {
    var result = JSONObject()
    for (key, value) in [
      ("temperature", temperature), ("top_p", topP),
      ("repetition_penalty", repetitionPenalty),
    ] where !value.isBlank {
      result.fields[key] = try number(value, label: "\(stage).\(key)")
    }
    for (key, value) in [
      ("top_k", topK), ("penalty_window", penaltyWindow),
      ("min_tokens", minTokens), ("max_tokens", maxTokens),
    ] where !value.isBlank {
      result.fields[key] = try integer(value, label: "\(stage).\(key)")
    }
    return result
  }
}

struct RequestDraft: Codable, Equatable, Sendable {
  var source = "compose"
  var filePath = ""
  var id = "song"
  var style = ""
  var lyrics = ""
  var cot = "full"
  var seed = ""
  var cfgScale = ""
  var lyricsSource = "text"
  var lyricsPath = ""
  var scoreSource = "generate"
  var abc = ""
  var abcPath = ""
  var odeSteps = ""
  var generationJSON = ""
  var abcSampling = SamplingDraft()
  var semanticSampling = SamplingDraft()
  var overrideID = false
  var overrideStyle = false
  var overrideLyrics = false
  var overrideCot = false
  var overrideSeed = false
  var overrideCFG = false
  var overrideABC = false

  fileprivate func composedObject() throws -> JSONObject {
    var result = JSONObject()
    try result.set("id", id)
    try result.set("style", style)
    try result.set("cot", cot)
    if lyricsSource == "file" {
      try result.set("lyrics_path", absolutePath(lyricsPath, label: "Lyrics file"))
    } else {
      try result.set("lyrics", lyrics)
    }
    switch scoreSource {
    case "file": try result.set("abc_path", absolutePath(abcPath, label: "ABC file"))
    case "text": try result.set("abc", abc)
    default: break
    }
    if !seed.isBlank { result.fields["seed"] = try integer(seed, label: "Seed") }
    if !cfgScale.isBlank { result.fields["cfg_scale"] = try number(cfgScale, label: "CFG scale") }
    var generation = generationJSON.isBlank ? JSONObject() : try JSONObject(raw: generationJSON)
    if !odeSteps.isBlank {
      generation.fields["ode_steps"] = try integer(odeSteps, label: "ODE steps")
    }
    if !generation.fields.isEmpty || !generationJSON.isBlank {
      result.fields["generation_config"] = try generation.text()
    }
    let abcOptions = try abcSampling.object(stage: "abc_sampling")
    let semanticOptions = try semanticSampling.object(stage: "semantic_sampling")
    if !abcOptions.fields.isEmpty { result.fields["abc_sampling"] = try abcOptions.text() }
    if !semanticOptions.fields.isEmpty {
      result.fields["semantic_sampling"] = try semanticOptions.text()
    }
    return result
  }

  fileprivate func addFileRequest(to options: inout JSONObject) throws {
    try options.set("request_file", absolutePath(filePath, label: "Request file"))
    var overrides = JSONObject()
    if overrideID { try overrides.set("id", id) }
    if overrideStyle { try overrides.set("style", style) }
    if overrideCot { try overrides.set("cot", cot) }
    if overrideSeed { overrides.fields["seed"] = try integer(seed, label: "Seed override") }
    if overrideCFG {
      overrides.fields["cfg_scale"] = try number(cfgScale, label: "CFG scale override")
    }
    if overrideLyrics {
      if lyricsSource == "file" {
        try options.set("lyrics_file", absolutePath(lyricsPath, label: "Lyrics override file"))
      } else {
        try overrides.set("lyrics", lyrics)
      }
    }
    if overrideABC {
      try options.set("abc_file", absolutePath(abcPath, label: "ABC override file"))
    }
    if !overrides.fields.isEmpty { options.fields["overrides"] = try overrides.text() }
  }
}

@MainActor @Observable final class EngineSettings {
  struct Snapshot: Codable, Equatable {
    var model = "m-a-p/YuE2-3B"
    var vae = "m-a-p/YuE2-Vae"
    var convertedDirectory = FileManager.default.urls(
      for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("Yueqin/models/converted", isDirectory: true).path
    var precision = "bf16"
    var vaeCoreFrames = ""
    var outputRoot = FileManager.default.urls(for: .musicDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("Yueqin", isDirectory: true).path
    var offline = false
    var requireAC = false
  }
  private var state: Snapshot
  var snapshot: Snapshot { state }
  var model: String {
    get { state.model }
    set { state.model = newValue }
  }
  var vae: String {
    get { state.vae }
    set { state.vae = newValue }
  }
  var convertedDirectory: String {
    get { state.convertedDirectory }
    set { state.convertedDirectory = newValue }
  }
  var precision: String {
    get { state.precision }
    set { state.precision = newValue }
  }
  var vaeCoreFrames: String {
    get { state.vaeCoreFrames }
    set { state.vaeCoreFrames = newValue }
  }
  var outputRoot: String {
    get { state.outputRoot }
    set { state.outputRoot = newValue }
  }
  var offline: Bool {
    get { state.offline }
    set { state.offline = newValue }
  }
  var requireAC: Bool {
    get { state.requireAC }
    set { state.requireAC = newValue }
  }

  init() { state = restored(Snapshot.self, key: "Yueqin.engineSettings.v1") ?? Snapshot() }
  func save() { persist(state, key: "Yueqin.engineSettings.v1") }
}

@MainActor @Observable final class WorkflowDraft {
  struct Snapshot: Codable, Equatable {
    var request = RequestDraft()
    var inputPath = ""
    var outputPath = ""
    var resume = false
    var replayStage = "decode"
    var verifyHashes = false
    var sourcePath = ""
    var cachePath = ""
    var batchMode = ""
  }
  let kind: WorkflowKind
  private var state: Snapshot
  var snapshot: Snapshot { state }
  var request: RequestDraft {
    get { state.request }
    set { state.request = newValue }
  }
  var inputPath: String {
    get { state.inputPath }
    set { state.inputPath = newValue }
  }
  var outputPath: String {
    get { state.outputPath }
    set { state.outputPath = newValue }
  }
  var resume: Bool {
    get { state.resume }
    set { state.resume = newValue }
  }
  var replayStage: String {
    get { state.replayStage }
    set { state.replayStage = newValue }
  }
  var verifyHashes: Bool {
    get { state.verifyHashes }
    set { state.verifyHashes = newValue }
  }
  var sourcePath: String {
    get { state.sourcePath }
    set { state.sourcePath = newValue }
  }
  var cachePath: String {
    get { state.cachePath }
    set { state.cachePath = newValue }
  }
  var batchMode: String {
    get { state.batchMode }
    set { state.batchMode = newValue }
  }

  init(kind: WorkflowKind) {
    self.kind = kind
    state = restored(Snapshot.self, key: "Yueqin.draft.\(kind.rawValue).v1") ?? Snapshot()
  }
  func save() { persist(state, key: "Yueqin.draft.\(kind.rawValue).v1") }

  func requestData() throws -> Data { try request.composedObject().data() }

  func makeSubmission(settings: EngineSettings) throws -> RunSubmission {
    var options = JSONObject()
    try options.set("precision", settings.precision)
    let output: URL?
    if !outputPath.isBlank {
      output = URL(fileURLWithPath: try absolutePath(outputPath, label: "Output destination"))
    } else if resume && (kind == .generate || kind == .batch) {
      throw DraftError(
        "Resume requires an explicit output directory. Select the existing run to resume.")
    } else if kind == .doctor {
      output = nil
    } else if kind == .prepare {
      output = URL(
        fileURLWithPath: try absolutePath(
          settings.convertedDirectory, label: "Converted model directory"))
    } else {
      let root = URL(fileURLWithPath: try absolutePath(settings.outputRoot, label: "Output root"))
      let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(
        of: ":", with: "-")
      output = root.appendingPathComponent(
        "\(kind.rawValue)-\(stamp)-\(UUID().uuidString)", isDirectory: true)
    }
    if let output { try options.set("output", output.path) }
    if kind == .prepare {
      options.fields["offline"] = settings.offline ? "true" : "false"
      if !sourcePath.isBlank {
        try options.set("source", absolutePath(sourcePath, label: "Source model directory"))
      }
      if !cachePath.isBlank {
        try options.set("cache_dir", absolutePath(cachePath, label: "Download cache directory"))
      }
    } else {
      if !settings.model.isBlank { try options.set("model", modelReference(settings.model)) }
      if !settings.vae.isBlank { try options.set("vae", modelReference(settings.vae)) }
      if !settings.convertedDirectory.isBlank {
        try options.set(
          "converted_dir",
          absolutePath(settings.convertedDirectory, label: "Converted model directory"))
      }
      if kind == .doctor {
        options.fields["verify_hashes"] = verifyHashes ? "true" : "false"
      } else {
        options.fields["offline"] = settings.offline ? "true" : "false"
        options.fields["require_ac"] = settings.requireAC ? "true" : "false"
        if !settings.vaeCoreFrames.isBlank {
          options.fields["vae_core_frames"] = try integer(
            settings.vaeCoreFrames, label: "VAE core frames")
        }
      }
    }
    switch kind {
    case .generate, .plan:
      if request.source == "file" {
        try request.addFileRequest(to: &options)
      } else {
        options.fields["request"] = try request.composedObject().text()
      }
      if kind == .generate { options.fields["resume"] = resume ? "true" : "false" }
    case .renderPlan, .replay, .batch:
      try options.set("input", absolutePath(inputPath, label: "Input"))
      if kind == .replay { try options.set("stage", replayStage) }
      if kind == .batch {
        options.fields["resume"] = resume ? "true" : "false"
        if !batchMode.isBlank {
          var overrides = JSONObject()
          try overrides.set("cot", batchMode)
          options.fields["overrides"] = try overrides.text()
        }
      }
    case .prepare, .doctor: break
    }
    return RunSubmission(kind: kind, options: try options.data(), outputURL: output)
  }
}

struct RunSubmission: Sendable {
  let kind: WorkflowKind
  let options: Data
  let outputURL: URL?
}

private struct DraftError: LocalizedError {
  let message: String
  init(_ message: String) { self.message = message }
  var errorDescription: String? { message }
}

extension String {
  fileprivate var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
  fileprivate var isBlank: Bool { trimmed.isEmpty }
}

private func integer(_ text: String, label: String) throws -> String {
  guard let value = Int64(text.trimmed) else {
    throw DraftError("\(label) must be a whole number representable as a signed 64-bit integer.")
  }
  return String(value)
}

private func number(_ text: String, label: String) throws -> String {
  guard let value = Double(text.trimmed), value.isFinite else {
    throw DraftError("\(label) must be a finite number, for example 1.0.")
  }
  return String(value)
}

private func absolutePath(_ text: String, label: String) throws -> String {
  guard !text.isBlank else {
    throw DraftError("\(label) is required. Choose a file or directory, or enter an absolute path.")
  }
  guard !text.contains("\0") else { throw DraftError("\(label) contains a null character.") }
  let expanded = (text as NSString).expandingTildeInPath
  guard (expanded as NSString).isAbsolutePath else {
    throw DraftError("\(label) must be an absolute path (or begin with ~/).")
  }
  return expanded
}

private func modelReference(_ text: String) throws -> String {
  if text.hasPrefix("/") || text.hasPrefix("~") || text.hasPrefix(".") {
    return try absolutePath(text, label: "Model location")
  }
  guard !text.contains("\0") else { throw DraftError("Model location contains a null character.") }
  return text
}

private func restored<T: Decodable>(_ type: T.Type, key: String) -> T? {
  guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
  return try? JSONDecoder().decode(type, from: data)
}

private func persist<T: Encodable>(_ value: T, key: String) {
  // Draft snapshots contain only strings and booleans, so encoding cannot fail.
  guard let data = try? JSONEncoder().encode(value) else { return }
  UserDefaults.standard.set(data, forKey: key)
}

/// Stores JSON member values as raw JSON so arbitrary generation-config numbers
/// and nested values never round-trip through floating point or Foundation bridging.
private struct JSONObject {
  var fields: [String: String] = [:]
  init() {}

  init(raw: String) throws {
    let data = Data(raw.utf8)
    do {
      guard
        try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) is [String: Any]
      else {
        throw DraftError(
          "generation_config must be a JSON object, for example {\"ode_steps\": 32}.")
      }
    } catch let error as DraftError {
      throw error
    } catch {
      throw DraftError("Invalid generation_config JSON: \(error.localizedDescription)")
    }
    let characters = Array(raw.trimmed)
    var start = 1
    var depth = 0
    var quoted = false
    var escaped = false
    var colon: Int?
    for index in 1..<(characters.count - 1) {
      let character = characters[index]
      if quoted {
        if escaped {
          escaped = false
        } else if character == "\\" {
          escaped = true
        } else if character == "\"" {
          quoted = false
        }
        continue
      }
      switch character {
      case "\"": quoted = true
      case "{", "[": depth += 1
      case "}", "]": depth -= 1
      case ":" where depth == 0: if colon == nil { colon = index }
      case "," where depth == 0:
        try addMember(characters, start: start, colon: colon, end: index)
        start = index + 1
        colon = nil
      default: break
      }
    }
    if let colon {
      try addMember(characters, start: start, colon: colon, end: characters.count - 1)
    }
  }

  private mutating func addMember(_ characters: [Character], start: Int, colon: Int?, end: Int)
    throws
  {
    guard let colon else { throw DraftError("Invalid generation_config JSON member.") }
    let key = try JSONDecoder().decode(
      String.self, from: Data(String(characters[start..<colon]).utf8))
    fields[key] = String(characters[(colon + 1)..<end]).trimmed
  }

  mutating func set(_ key: String, _ value: String) throws {
    fields[key] = String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
  }
  func text() throws -> String {
    let encoder = JSONEncoder()
    return try "{"
      + fields.keys.sorted().map { key in
        String(decoding: try encoder.encode(key), as: UTF8.self) + ":" + fields[key]!
      }.joined(separator: ",") + "}"
  }
  func data() throws -> Data { Data(try text().utf8) }
}
