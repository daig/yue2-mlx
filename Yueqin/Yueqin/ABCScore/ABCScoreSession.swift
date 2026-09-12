import AppKit
import Observation

struct ABCScoreSnapshot: Sendable {
  let id: String
  let version: Int
  let abc: String
  let compatible: Bool
  let editable: Bool
}

@MainActor @Observable final class ABCScoreSession {
  let document: ScoreDocument
  private(set) var compatible = false
  private(set) var editable = false
  private(set) var validatedID = ""
  private(set) var validatedVersion = -1
  @ObservationIgnored private var owner: UUID?
  @ObservationIgnored private var commandHandler:
    ((String, String) async throws -> ABCScoreSnapshot)?
  @ObservationIgnored private var snapshotHandler: (() async throws -> ABCScoreSnapshot)?

  var canUseScore: Bool {
    document.hasDocument && !document.abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && validatedID == document.id && validatedVersion == document.revision && compatible
  }

  init(document: ScoreDocument) {
    self.document = document
    document.flush = { [weak self] in try await self?.flush() }
  }

  func attach(
    owner: UUID,
    command: @escaping (String, String) async throws -> ABCScoreSnapshot,
    snapshot: @escaping () async throws -> ABCScoreSnapshot
  ) {
    self.owner = owner
    commandHandler = command
    snapshotHandler = snapshot
  }

  func detach(owner: UUID) {
    guard self.owner == owner else { return }
    self.owner = nil
    commandHandler = nil
    snapshotHandler = nil
  }

  func receiveStatus(id: String, version: Int, compatible: Bool, editable: Bool) {
    guard id == document.id, version == document.revision else { return }
    validatedID = id
    validatedVersion = version
    self.compatible = compatible
    self.editable = editable
  }

  func flush() async throws {
    guard let snapshotHandler else { return }
    let snapshot = try await snapshotHandler()
    try accept(snapshot)
  }

  func command(_ action: String, value: String = "") {
    Task { @MainActor in
      do {
        guard let commandHandler else { return }
        let snapshot = try await commandHandler(action, value)
        try accept(snapshot)
      } catch {
        document.errorMessage = error.localizedDescription
      }
    }
  }

  func receiveNative(_ action: String) {
    switch action {
    case "undo": document.undo()
    case "redo": document.redo()
    case "save": Task { await document.save() }
    case "paste":
      if let text = NSPasteboard.general.string(forType: .string) {
        command("paste", value: text)
      }
    default: break
    }
  }

  func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
  }

  private func accept(_ snapshot: ABCScoreSnapshot) throws {
    // A disposed view can finish flushing after another document was opened.
    guard snapshot.id == document.id else { return }
    // An undo, redo or later edit may have advanced native state during the call.
    guard snapshot.version >= document.revision else { return }
    guard document.acceptSnapshot(id: snapshot.id, version: snapshot.version, abc: snapshot.abc)
    else {
      throw ScoreSessionError("The score changed while synchronizing. Try the action again.")
    }
    receiveStatus(
      id: snapshot.id, version: snapshot.version,
      compatible: snapshot.compatible, editable: snapshot.editable)
  }
}

private struct ScoreSessionError: LocalizedError {
  let message: String
  init(_ message: String) { self.message = message }
  var errorDescription: String? { message }
}
