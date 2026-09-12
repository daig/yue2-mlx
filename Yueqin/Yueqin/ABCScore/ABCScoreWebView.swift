import SwiftUI
import WebKit

struct ABCScoreRequest: Equatable {
  let abc: String
  let theme: String
  let zoom: Double
  var editor: ABCScoreEditorRequest?
}

struct ABCScoreEditorRequest: Equatable {
  let id: String
  let version: Int
  let canUndo: Bool
  let canRedo: Bool
  let selectionStart: Int?

  @MainActor
  init(document: ScoreDocument) {
    id = document.id
    version = document.revision
    canUndo = document.canUndo
    canRedo = document.canRedo
    selectionStart = document.selectionStart
  }

  var payload: [String: Any] {
    [
      "id": id, "version": version, "canUndo": canUndo, "canRedo": canRedo,
      "selectionStart": selectionStart.map { $0 as Any } ?? NSNull(),
    ]
  }
}

struct ABCScoreStatus {
  enum State {
    case loading, rendered, empty, failed
  }

  var state: State
  var abc: String = ""
  var warnings: [String] = []
  var message: String?
}

private final class ABCScoreBundleMarker: NSObject {}

@MainActor
struct ABCScoreWebView: NSViewRepresentable {
  let request: ABCScoreRequest
  let onStatus: @MainActor (ABCScoreStatus) -> Void
  var session: ABCScoreSession?

  func makeCoordinator() -> Coordinator {
    Coordinator(onStatus: onStatus, session: session)
  }

  func makeNSView(context: Context) -> WKWebView {
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    configuration.userContentController.add(context.coordinator, name: "abcScore")
    let webView = WKWebView(frame: .zero, configuration: configuration)
    webView.navigationDelegate = context.coordinator
    webView.allowsBackForwardNavigationGestures = false
    webView.underPageBackgroundColor = .clear
    context.coordinator.start(webView, request: request)
    return webView
  }

  func updateNSView(_ webView: WKWebView, context: Context) {
    context.coordinator.onStatus = onStatus
    context.coordinator.bind(session)
    context.coordinator.update(request)
  }

  static func dismantleNSView(_ webView: WKWebView, coordinator: Coordinator) {
    coordinator.dispose(webView)
    webView.navigationDelegate = nil
    webView.configuration.userContentController.removeScriptMessageHandler(forName: "abcScore")
  }

  @MainActor
  final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
    var onStatus: (@MainActor (ABCScoreStatus) -> Void)?
    private weak var webView: WKWebView?
    private var request: ABCScoreRequest?
    private var revision = 0
    private var active = true
    private var loaded = false
    private var publication = 0
    private var documentURL: URL?
    private var loadFailure: String?
    private let owner = UUID()
    private var session: ABCScoreSession?

    init(onStatus: @escaping @MainActor (ABCScoreStatus) -> Void, session: ABCScoreSession?) {
      self.onStatus = onStatus
      self.session = session
    }

    func bind(_ session: ABCScoreSession?) {
      if self.session !== session { self.session?.detach(owner: owner) }
      self.session = session
      session?.attach(
        owner: owner,
        command: { [weak self] action, value in
          guard let self else { throw CocoaError(.coderInvalidValue) }
          return try await self.snapshot(action: action, value: value)
        },
        snapshot: { [weak self] in
          guard let self else { throw CocoaError(.coderInvalidValue) }
          return try await self.snapshot()
        })
    }

    func start(_ webView: WKWebView, request: ABCScoreRequest) {
      self.webView = webView
      bind(session)
      update(request)
      guard
        let url = Bundle(for: ABCScoreBundleMarker.self).url(
          forResource: "ABCScoreViewer", withExtension: "html")
      else {
        failLoading("The bundled notation renderer could not be found.")
        return
      }
      documentURL = url.standardizedFileURL
      webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    }

    func update(_ newRequest: ABCScoreRequest) {
      guard active, request != newRequest else { return }
      let sourceChanged = request?.abc != newRequest.abc
      request = newRequest
      revision += 1
      if let loadFailure {
        publish(ABCScoreStatus(state: .failed, abc: newRequest.abc, message: loadFailure))
      } else if loaded {
        if sourceChanged { publish(ABCScoreStatus(state: .loading, abc: newRequest.abc)) }
        render()
      } else {
        publish(ABCScoreStatus(state: .loading, abc: newRequest.abc))
      }
    }

    func dispose(_ webView: WKWebView) {
      active = false
      onStatus = nil
      // Keep the old page alive long enough to capture its last input event.
      // A replacement document rejects this snapshot by epoch.
      Task { @MainActor [self, webView] in
        if let session, loaded {
          do {
            let final = try await snapshot(action: "stop")
            _ = session.document.acceptSnapshot(
              id: final.id, version: final.version, abc: final.abc)
          } catch {
            session.document.errorMessage =
              "Could not retain the latest score edit: \(error.localizedDescription)"
          }
        }
        session?.detach(owner: owner)
        webView.stopLoading()
        self.webView = nil
      }
    }

    private func render(forceEditor: Bool = false) {
      guard active, loaded, let webView, let request else { return }
      let renderingRevision = revision
      var editor = request.editor?.payload
      var abc = request.abc
      if forceEditor, let document = session?.document {
        editor = ABCScoreEditorRequest(document: document).payload
        editor?["force"] = true
        abc = document.abc
      }
      webView.callAsyncJavaScript(
        "window.YueqinABCScore.update({abc, revision, theme, zoom, editor});",
        arguments: [
          "abc": abc, "revision": renderingRevision,
          "theme": request.theme, "zoom": request.zoom,
          "editor": editor.map { $0 as Any } ?? NSNull(),
        ],
        in: nil, in: .page
      ) { [weak self] result in
        guard let self, self.active, self.revision == renderingRevision else { return }
        if case .failure(let error) = result {
          self.publish(
            ABCScoreStatus(
              state: .failed, abc: request.abc,
              message: "The notation renderer could not run: \(error.localizedDescription)"))
        }
      }
    }

    private func snapshot(action: String? = nil, value: String = "") async throws
      -> ABCScoreSnapshot
    {
      guard let webView, loaded else {
        guard let document = session?.document else { throw CocoaError(.coderInvalidValue) }
        return ABCScoreSnapshot(
          id: document.id, version: document.revision, abc: document.abc,
          compatible: false, editable: false)
      }
      return try await withCheckedThrowingContinuation { continuation in
        webView.callAsyncJavaScript(
          action == nil
            ? "return window.YueqinABCScore.snapshot();"
            : "return window.YueqinABCScore.command(action, value);",
          arguments: ["action": action ?? "", "value": value],
          in: nil, in: .page
        ) { result in
          switch result {
          case .success(let value):
            guard let body = value as? [String: Any],
              let id = body["id"] as? String, let version = body["version"] as? Int,
              let abc = body["abc"] as? String
            else {
              continuation.resume(throwing: CocoaError(.coderInvalidValue))
              return
            }
            continuation.resume(
              returning: ABCScoreSnapshot(
                id: id, version: version, abc: abc,
                compatible: body["compatible"] as? Bool == true,
                editable: body["editable"] as? Bool == true))
          case .failure(let error):
            continuation.resume(throwing: error)
          }
        }
      }
    }

    // Defer all observable state changes beyond make/updateNSView. A queued result
    // must still belong to the live view and its latest request when delivered.
    private func publish(_ status: ABCScoreStatus) {
      let publishingRevision = revision
      publication += 1
      let publishingSequence = publication
      Task { @MainActor [weak self] in
        guard let self, self.active, self.revision == publishingRevision,
          self.publication == publishingSequence
        else { return }
        self.onStatus?(status)
      }
    }

    private func failLoading(_ message: String) {
      loaded = false
      loadFailure = message
      publish(ABCScoreStatus(state: .failed, abc: request?.abc ?? "", message: message))
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
      guard active, webView.url?.standardizedFileURL == documentURL else { return }
      loaded = true
      loadFailure = nil
      render()
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
      guard active else { return }
      failLoading("The notation renderer could not load: \(error.localizedDescription)")
    }

    func webView(
      _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
      withError error: Error
    ) {
      guard active else { return }
      failLoading("The notation renderer could not load: \(error.localizedDescription)")
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
      guard active else { return }
      failLoading("The notation renderer stopped unexpectedly. Reopen the score to try again.")
    }

    func webView(
      _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
      decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
    ) {
      let isInitialDocument =
        active && !loaded
        && navigationAction.navigationType == .other
        && navigationAction.targetFrame?.isMainFrame == true
        && navigationAction.request.url?.standardizedFileURL == documentURL
      decisionHandler(isInitialDocument ? .allow : .cancel)
    }

    func userContentController(
      _ userContentController: WKUserContentController, didReceive message: WKScriptMessage
    ) {
      guard active, loaded, message.name == "abcScore", message.frameInfo.isMainFrame,
        message.frameInfo.request.url?.standardizedFileURL == documentURL,
        let body = message.body as? [String: Any]
      else { return }
      switch body["kind"] as? String {
      case "edit":
        guard let document = session?.document,
          let id = body["id"] as? String, let base = body["baseVersion"] as? Int,
          let start = body["start"] as? Int, let end = body["end"] as? Int,
          let text = body["text"] as? String, let label = body["label"] as? String
        else { return }
        if !document.applyEdit(
          id: id, baseVersion: base, start: start, end: end, text: text,
          label: label, selectionStart: body["selectionStart"] as? Int)
        {
          render(forceEditor: true)
        }
      case "native":
        guard body["id"] as? String == session?.document.id,
          let action = body["action"] as? String
        else { return }
        session?.receiveNative(action)
      case "clipboard":
        guard body["id"] as? String == session?.document.id,
          let text = body["text"] as? String
        else { return }
        session?.copy(text)
      case "status", nil:
        guard let messageRevision = body["revision"] as? Int, messageRevision == revision,
          let state = body["state"] as? String, let warnings = body["warnings"] as? [String],
          let request
        else { return }
        let renderState: ABCScoreStatus.State
        switch state {
        case "rendered": renderState = .rendered
        case "empty": renderState = .empty
        case "failed": renderState = .failed
        default: return
        }
        if let editor = body["editor"] as? [String: Any],
          let id = editor["id"] as? String, let version = editor["version"] as? Int
        {
          session?.receiveStatus(
            id: id, version: version, compatible: editor["compatible"] as? Bool == true,
            editable: editor["editable"] as? Bool == true)
        }
        publish(
          ABCScoreStatus(
            state: renderState, abc: request.abc, warnings: warnings,
            message: body["message"] as? String))
      default: break
      }
    }
  }
}
