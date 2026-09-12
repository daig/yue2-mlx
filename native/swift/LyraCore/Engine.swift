import CLyraCore
import CSndFile
import Darwin
import Foundation

/// Raw engine workflows. Each case corresponds to the same operation used by
/// the terminal frontend, not a command to launch as a subprocess.
public enum Workflow: Sendable, CaseIterable {
  case prepare, generate, plan, renderPlan, replay, batch, doctor

  fileprivate var native: lyra_operation {
    switch self {
    case .prepare: LYRA_PREPARE
    case .generate: LYRA_GENERATE
    case .plan: LYRA_PLAN
    case .renderPlan: LYRA_RENDER_PLAN
    case .replay: LYRA_REPLAY
    case .batch: LYRA_BATCH
    case .doctor: LYRA_DOCTOR
    }
  }
}

public struct EngineError: Error, Sendable, LocalizedError {
  public let type: String
  public let message: String
  public let status: Int
  public var errorDescription: String? { "\(type): \(message)" }
}

/// A doctor can be unready or a batch partly fail without throwing away its
/// diagnostic/per-row result. Thrown engine errors are separate from this state.
public struct WorkflowResult: Sendable {
  public let succeeded: Bool
  public let json: Data
}

private final class EventRelay {
  let receive: @Sendable (Data) -> Void
  init(_ receive: @escaping @Sendable (Data) -> Void) { self.receive = receive }
}

private func receiveNativeEvent(_ user: UnsafeMutableRawPointer?, _ text: UnsafePointer<CChar>?) {
  guard let user, let text else { return }
  let relay = Unmanaged<EventRelay>.fromOpaque(user).takeUnretainedValue()
  relay.receive(Data(bytes: text, count: strlen(text)))
}

/// Owns one native execution context. `execute` blocks its caller: invoke it on
/// your worker/task, not the UI actor. Concurrent execution on the same engine
/// is rejected; `cancel` is safe from another thread. Callbacks are synchronous
/// on the executing thread. Dispatch presentation to the UI actor yourself.
///
/// The unchecked Sendable conformance is backed by the native execution mutex,
/// atomic cancellation flag, and immutable, Sendable event callback.
public final class Engine: @unchecked Sendable {
  private let context: OpaquePointer
  private let relay: EventRelay?

  public init(onEvent: (@Sendable (Data) -> Void)? = nil) throws {
    guard String(cString: lyra_core_build_identifier()) == packagedCoreBuildIdentifier else {
      throw EngineError(
        type: "RuntimeError",
        message:
          "Swift and native core builds differ; regenerate package artifacts and rebuild the client.",
        status: 1)
    }
    guard
      let resource = Bundle.module.url(
        forResource: "mlx", withExtension: "metallib", subdirectory: "Resources"
      )
    else {
      throw EngineError(
        type: "FileNotFoundError", message: "Bundled mlx.metallib is missing", status: 1)
    }
    let relay = onEvent.map(EventRelay.init)
    var error: UnsafeMutablePointer<CChar>?
    let context = resource.path.withCString {
      lyra_context_create(
        $0, relay == nil ? nil : receiveNativeEvent,
        relay.map { Unmanaged.passUnretained($0).toOpaque() }, &error)
    }
    defer { lyra_string_free(error) }
    guard let context else { throw Self.decodeError(error, status: 1) }
    self.context = context
    self.relay = relay
  }

  deinit { lyra_context_destroy(context) }

  /// Options are native JSON (including exact integer seeds), not argv. See
  /// docs/usage.md for the operation fields and unchanged request schema.
  public func execute(_ operation: Workflow, options: Data = Data("{}".utf8)) throws
    -> WorkflowResult
  {
    guard let text = String(data: options, encoding: .utf8), !text.utf8.contains(0) else {
      throw EngineError(
        type: "ValueError", message: "Options must be UTF-8 JSON without embedded NUL bytes",
        status: 2)
    }
    var result: UnsafeMutablePointer<CChar>?
    var error: UnsafeMutablePointer<CChar>?
    let status = text.withCString {
      lyra_execute(context, operation.native, $0, &result, &error)
    }
    defer {
      lyra_string_free(result)
      lyra_string_free(error)
    }
    guard error == nil, let result else {
      throw Self.decodeError(error, status: Int(status.rawValue))
    }
    return WorkflowResult(
      succeeded: status == LYRA_SUCCESS,
      json: Data(bytes: result, count: strlen(result))
    )
  }

  public func cancel() { lyra_context_cancel(context) }

  private static func decodeError(_ text: UnsafePointer<CChar>?, status: Int) -> EngineError {
    struct NativeError: Decodable {
      let type: String
      let reason: String
    }
    if let text,
      let error = try? JSONDecoder().decode(
        NativeError.self, from: Data(bytes: text, count: strlen(text)))
    {
      return EngineError(type: error.type, message: error.reason, status: status)
    }
    return EngineError(
      type: "RuntimeError", message: "Native execution failed without an available error payload",
      status: status)
  }
}
