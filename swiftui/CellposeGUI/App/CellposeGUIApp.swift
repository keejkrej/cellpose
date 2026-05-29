import SwiftUI

@main
struct CellposeGUIApp: App {
    @State private var sidecarManager = SidecarProcessManager()
    @State private var viewModel: MainViewModel?

    var body: some Scene {
        WindowGroup {
            Group {
                if let viewModel {
                    ContentView(viewModel: viewModel)
                        .focusedSceneValue(\.mainViewModel, viewModel)
                } else {
                    VStack(spacing: 12) {
                        if !sidecarManager.hasFailed {
                            ProgressView()
                        }
                        Text(sidecarManager.statusMessage)
                            .foregroundStyle(sidecarManager.hasFailed ? .primary : .secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 480)
                        if sidecarManager.hasFailed {
                            Text("Set CELLPOSE_ROOT to your repo path, or run `task run:swiftui` from the repo root.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                                .frame(maxWidth: 480)
                        }
                    }
                    .frame(minWidth: 900, minHeight: 600)
                }
            }
            .task {
                if RuntimeEnvironment.skipSidecar {
                    viewModel = MainViewModel(ml: StubMlEngine())
                    return
                }
                await sidecarManager.startIfNeeded()
                if viewModel == nil, sidecarManager.isReady {
                    let ml = SidecarMlEngine(
                        client: SidecarClient(baseURL: sidecarManager.baseURL)
                    )
                    let model = MainViewModel(ml: ml)
                    await model.refreshModels()
                    viewModel = model
                }
            }
        }
        .commands {
            AppCommands()
        }
        .defaultSize(width: 1280, height: 800)
    }
}
