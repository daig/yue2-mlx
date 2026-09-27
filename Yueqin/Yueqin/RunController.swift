import Foundation
import LyraCore
import Observation

indirect enum RunJSON: Decodable, Sendable {
  case object([String: RunJSON])
  case array([RunJSON])
  case string(String)
  case number(Double)
  case bool(Bool)
  case null
  init(from decoder: any Decoder) throws {
    let c = try decoder.singleValueContainer()
    if c.decodeNil() {
      self = .null
    } else if let v = try? c.decode(Bool.self) {
      self = .bool(v)
    } else if let v = try? c.decode(String.self) {
      self = .string(v)
    } else if let v = try? c.decode([String: RunJSON].self) {
      self = .object(v)
    } else if let v = try? c.decode([RunJSON].self) {
      self = .array(v)
    } else {
      self = .number(try c.decode(Double.self))
    }
  }
  subscript(_ key: String) -> RunJSON {
    if case .object(let v) = self { return v[key] ?? .null }
    return .null
  }
  var text: String? {
    if case .string(let v) = self { return v }
    return nil
  }
  var number: Double? {
    if case .number(let v) = self { return v }
    return nil
  }
  var flag: Bool? {
    if case .bool(let v) = self { return v }
    return nil
  }
  var rows: [RunJSON] {
    if case .array(let v) = self { return v }
    return []
  }
  var fields: [String: RunJSON] {
    if case .object(let v) = self { return v }
    return [:]
  }
  var summary: String {
    switch self {
    case .string(let v): v
    case .number(let v): v.formatted()
    case .bool(let v): v ? "Yes" : "No"
    case .null: "—"
    case .array(let v): v.map(\.summary).joined(separator: ", ")
    case .object(let v): v.keys.sorted().map { "\($0): \(v[$0]!.summary)" }.joined(separator: "\n")
    }
  }
}

// The lock serializes cancellation with engine attachment and detachment. The
// workflow_started callback reapplies intent after the native flag is reset.
private final class RunCancellation: @unchecked Sendable {
  private let lock = NSLock()
  private var requested = false
  private var engine: Engine?
  func attach(_ engine: Engine) {
    lock.lock()
    defer { lock.unlock() }
    self.engine = engine
  }
  func detach() {
    lock.lock()
    defer { lock.unlock() }
    engine = nil
  }
  func request() {
    lock.lock()
    defer { lock.unlock() }
    requested = true
    engine?.cancel()
  }
  func didStart() {
    lock.lock()
    defer { lock.unlock() }
    if requested { engine?.cancel() }
  }
}

struct RunEvent: Identifiable, Sendable {
  let id = UUID()
  let value: RunJSON
}

struct PlannedScore: Sendable {
  let abc: String
  let outputURL: URL
  let truncated: Bool
}

enum PlannedScoreStatus: Equatable, Sendable {
  case empty
  case running
  case ready
  case failed(String)
}

@MainActor @Observable final class RunController {
  private(set) var isRunning = false
  private(set) var statusTitle = "Ready"
  private(set) var completedRunID: UUID?
  private(set) var preparedModelPath: String?
  private(set) var preparedVAEPath: String?
  private(set) var lastSubmission: RunSubmission?
  private(set) var cancellationRequested = false
  private(set) var startedAt: Date?
  private(set) var endedAt: Date?
  private(set) var progress: RunJSON = .null
  private(set) var events: [RunEvent] = []
  private(set) var result: RunJSON = .null
  private(set) var resultData: Data?
  private(set) var partialBatchData: Data?
  private(set) var errorMessage: String?
  private(set) var succeeded = false
  private(set) var artifacts: [URL] = []
  private(set) var outputURL: URL?
  private(set) var artifactError: String?
  private(set) var plannedScore: PlannedScore?
  private(set) var planScoreStatus: PlannedScoreStatus = .empty
  var onIdle: (() -> Void)?
  @ObservationIgnored private let queue = DispatchQueue(
    label: "Yueqin.native-workflow", qos: .userInitiated)
  @ObservationIgnored private var cancellation: RunCancellation?
  private var activeID: UUID?

  init() {}

  func start(_ submission: RunSubmission) {
    guard !isRunning else { return }
    let id = UUID()
    let token = RunCancellation()
    activeID = id
    cancellation = token
    lastSubmission = submission
    isRunning = true
    cancellationRequested = false
    statusTitle = "Starting \(submission.kind.title.lowercased())…"
    startedAt = Date()
    endedAt = nil
    preparedModelPath = nil
    preparedVAEPath = nil
    progress = .null
    events = []
    result = .null
    resultData = nil
    partialBatchData = nil
    errorMessage = nil
    artifactError = nil
    succeeded = false
    artifacts = []
    outputURL = nil
    if submission.kind == .plan { planScoreStatus = .running }
    queue.async { [self] in
      var outcome: WorkflowResult?
      var failure: String?
      var cancelled = false
      do {
        let engine = try Engine { [weak self, token] data in
          guard let event = try? JSONDecoder().decode(RunJSON.self, from: data) else { return }
          if event["type"].text == "workflow_started" { token.didStart() }
          DispatchQueue.main.async { [weak self] in self?.receive(event, id: id) }
        }
        token.attach(engine)
        defer { token.detach() }
        outcome = try engine.execute(submission.kind.operation, options: submission.options)
      } catch let error as EngineError {
        cancelled = error.type == "InterruptedError"
        failure = "\(error.type) (status \(error.status)): \(error.message)"
      } catch {
        failure = error.localizedDescription
      }
      DispatchQueue.main.async { [self] in
        guard activeID == id, isRunning else { return }
        progress = .null
        if !cancellationRequested { statusTitle = "Checking output…" }
      }
      let parsed =
        outcome.flatMap { try? JSONDecoder().decode(RunJSON.self, from: $0.json) } ?? .null
      let output = parsed["output"].text.map { URL(fileURLWithPath: $0) } ?? submission.outputURL
      var score: PlannedScore?
      var scoreFailure: String?
      if submission.kind == .plan {
        if outcome?.succeeded == true, let output {
          let scoreURL = output.appendingPathComponent("score.abc")
          do {
            let abc: String
            do {
              let data = try Data(contentsOf: scoreURL)
              guard let text = String(data: data, encoding: .utf8) else {
                throw CocoaError(.fileReadInapplicableStringEncoding)
              }
              abc = text
            } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
              let metadata = try JSONDecoder().decode(
                RunJSON.self,
                from: Data(contentsOf: output.appendingPathComponent("plan.json")))
              guard metadata["request"]["cot"].text == "off",
                case .some(.null) = metadata.fields["abc"]
              else { throw error }
              abc = ""
            }
            score = PlannedScore(
              abc: abc, outputURL: output, truncated: parsed["truncated"].flag == true)
          } catch {
            scoreFailure =
              "The plan completed, but its saved score could not be loaded at \(scoreURL.path): \(error.localizedDescription)"
          }
        } else {
          scoreFailure =
            cancelled
            ? "Planning was cancelled."
            : failure
              ?? (outcome?.succeeded == true
                ? "The plan completed, but no output location was returned."
                : "Planning failed.")
        }
      }
      var files: [URL] = []
      var listingError: String?
      var partial: Data?
      if let output {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: output.path, isDirectory: &isDirectory) {
          if isDirectory.boolValue {
            if let enumerator = FileManager.default.enumerator(
              at: output, includingPropertiesForKeys: [.isRegularFileKey],
              errorHandler: { _, error in
                listingError = error.localizedDescription
                return true
              })
            {
              for case let file as URL in enumerator {
                if (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                  files.append(file)
                }
              }
            }
            if submission.kind == .batch, outcome == nil {
              partial = try? Data(contentsOf: output.appendingPathComponent("batch.json"))
            }
          } else {
            files = [output]
          }
        }
      }
      let partialResult = partial.flatMap { try? JSONDecoder().decode(RunJSON.self, from: $0) }
      let finalOutcome = outcome
      let finalFailure = failure
      let finalCancelled = cancelled
      let finalFiles = files.sorted { $0.path < $1.path }
      let finalListingError = listingError
      let finalPartial = partial
      let finalScore = score
      let finalScoreFailure = scoreFailure
      DispatchQueue.main.async { [self] in
        guard activeID == id else { return }
        resultData = finalOutcome?.json
        partialBatchData = finalPartial
        result = partialResult ?? parsed
        succeeded = finalOutcome?.succeeded == true
        errorMessage = finalFailure
        statusTitle =
          finalCancelled
          ? "Cancelled"
          : succeeded
            ? "Completed"
            : submission.kind == .doctor && finalOutcome != nil
              ? "Not ready"
              : submission.kind == .batch && finalOutcome != nil
                ? "Batch finished with failures" : "Failed"
        artifacts = finalFiles
        artifactError = finalListingError
        outputURL = output
        if submission.kind == .plan {
          if let finalScore {
            plannedScore = finalScore
            planScoreStatus = .ready
          } else {
            planScoreStatus = .failed(finalScoreFailure ?? "The saved score could not be loaded.")
          }
        }
        if succeeded && submission.kind == .prepare {
          preparedModelPath = parsed["model"].text
          preparedVAEPath = parsed["vae"].text
        }
        endedAt = Date()
        isRunning = false
        cancellation = nil
        completedRunID = id
        let callback = onIdle
        onIdle = nil
        callback?()
      }
    }
  }

  func cancel() {
    guard isRunning else { return }
    cancellationRequested = true
    statusTitle = "Cancellation requested — waiting for native work to stop"
    cancellation?.request()
  }

  private func receive(_ event: RunJSON, id: UUID) {
    guard activeID == id, isRunning else { return }
    let type = event["type"].text ?? "event"
    if type == "progress" || type == "stage_started" || type == "stage_completed" {
      progress = event
      if !cancellationRequested { statusTitle = event["stage"].text ?? "Running" }
    }
    if type != "progress" { events.append(RunEvent(value: event)) }
  }
}
