import SwiftUI

/// Offline notation with an optional native document-backed editing session.
@MainActor
struct ABCScoreView: View {
  let abc: String
  var session: ABCScoreSession?

  @Environment(\.colorScheme) private var colorScheme
  @State private var zoom = 1.0
  @State private var status = ABCScoreStatus(state: .loading)

  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 8) {
        Text(session == nil ? "Score" : "Write").font(.headline)
        if let session {
          Button {
            session.document.undo()
          } label: {
            Image(systemName: "arrow.uturn.backward")
          }
          .disabled(!session.document.canUndo)
          .help("Undo score edit (Command–Z while editing notation)")
          .accessibilityIdentifier("score.undo")
          Button {
            session.document.redo()
          } label: {
            Image(systemName: "arrow.uturn.forward")
          }
          .disabled(!session.document.canRedo)
          .help("Redo score edit (Shift–Command–Z while editing notation)")
          .accessibilityIdentifier("score.redo")
        }
        Spacer()
        Button {
          zoom = max(0.5, zoom - 0.25)
        } label: {
          Image(systemName: "minus.magnifyingglass")
        }
        .disabled(zoom <= 0.5)
        .accessibilityLabel("Zoom out")
        .accessibilityIdentifier("abcScore.zoomOut")
        Button("Fit width") { zoom = 1 }
          .accessibilityLabel("Reset score to fit width")
          .accessibilityIdentifier("abcScore.fitWidth")
        Button {
          zoom = min(3, zoom + 0.25)
        } label: {
          Image(systemName: "plus.magnifyingglass")
        }
        .disabled(zoom >= 3)
        .accessibilityLabel("Zoom in")
        .accessibilityIdentifier("abcScore.zoomIn")
        Text(zoom, format: .percent.precision(.fractionLength(0)))
          .monospacedDigit()
          .frame(minWidth: 42, alignment: .trailing)
          .accessibilityLabel("Score zoom")
          .accessibilityValue(zoom.formatted(.percent))
      }
      .controlSize(.small)
      .padding(10)
      Divider()
      ZStack {
        ABCScoreWebView(
          request: ABCScoreRequest(
            abc: abc, theme: colorScheme == .dark ? "dark" : "light", zoom: zoom,
            editor: session.map { ABCScoreEditorRequest(document: $0.document) }),
          onStatus: { status = $0 },
          session: session
        )
        .accessibilityIdentifier("abcScore.notation")
        .opacity(session != nil || visibleState == .rendered ? 1 : 0)
        .allowsHitTesting(session != nil || visibleState == .rendered)
        .accessibilityHidden(session == nil && visibleState != .rendered)

        if session == nil {
          switch visibleState {
          case .loading:
            ProgressView("Rendering score…")
          case .empty:
            ContentUnavailableView(
              "No notation", systemImage: "music.note",
              description: Text("There is no renderable notation in this ABC source."))
          case .failed:
            ContentUnavailableView(
              "Unable to display score", systemImage: "exclamationmark.triangle",
              description: Text(verbatim: status.message ?? "The notation renderer failed."))
          case .rendered:
            EmptyView()
          }
        }
      }
      .frame(minHeight: 180)

      if session == nil, status.abc == abc, !status.warnings.isEmpty {
        Divider()
        DisclosureGroup("Parser warnings (\(status.warnings.count))") {
          ScrollView {
            VStack(alignment: .leading, spacing: 6) {
              ForEach(Array(status.warnings.enumerated()), id: \.offset) { _, warning in
                Text(verbatim: warning)
                  .frame(maxWidth: .infinity, alignment: .leading)
              }
            }
            .font(.caption)
            .textSelection(.enabled)
            .padding(.top, 6)
          }
          .frame(maxHeight: 150)
        }
        .accessibilityIdentifier("abcScore.warnings")
        .padding(10)
      }
    }
    .background(Color(nsColor: .textBackgroundColor))
    .clipShape(RoundedRectangle(cornerRadius: 6))
    .overlay(
      RoundedRectangle(cornerRadius: 6)
        .stroke(Color(nsColor: .separatorColor), lineWidth: 1))
  }

  private var visibleState: ABCScoreStatus.State {
    status.abc == abc ? status.state : .loading
  }
}

#Preview("Inline ABC score") {
  ABCScoreView(
    abc: """
      X:1
      T:Simple melody
      M:4/4
      L:1/4
      K:C
      C D E F | G A G2 | E D C2 |]
      """
  )
  .frame(width: 640, height: 420)
  .padding()
}
