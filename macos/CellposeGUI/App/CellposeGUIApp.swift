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
                        ProgressView()
                        Text(sidecarManager.statusMessage)
                            .foregroundStyle(.secondary)
                    }
                    .frame(minWidth: 900, minHeight: 600)
                }
            }
            .task {
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
