import SwiftUI
import WebKit

struct ABCScoreRequest: Equatable {
  let abc: String
  let theme: String
  let zoom: Double
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

  func makeCoordinator() -> Coordinator {
    Coordinator(onStatus: onStatus)
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
    context.coordinator.update(request)
  }

  static func dismantleNSView(_ webView: WKWebView, coordinator: Coordinator) {
    coordinator.dispose()
    webView.stopLoading()
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

    init(onStatus: @escaping @MainActor (ABCScoreStatus) -> Void) {
      self.onStatus = onStatus
    }

    func start(_ webView: WKWebView, request: ABCScoreRequest) {
      self.webView = webView
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

    func dispose() {
      active = false
      onStatus = nil
      webView = nil
    }

    private func render() {
      guard active, loaded, let webView, let request else { return }
      let renderingRevision = revision
      webView.callAsyncJavaScript(
        "window.YueqinABCScore.update({abc, revision, theme, zoom});",
        arguments: [
          "abc": request.abc, "revision": renderingRevision,
          "theme": request.theme, "zoom": request.zoom,
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
        let body = message.body as? [String: Any],
        let messageRevision = body["revision"] as? Int, messageRevision == revision,
        let state = body["state"] as? String,
        let warnings = body["warnings"] as? [String],
        let request
      else { return }
      let renderState: ABCScoreStatus.State
      switch state {
      case "rendered": renderState = .rendered
      case "empty": renderState = .empty
      case "failed": renderState = .failed
      default: return
      }
      publish(
        ABCScoreStatus(
          state: renderState, abc: request.abc, warnings: warnings,
          message: body["message"] as? String))
    }
  }
}
