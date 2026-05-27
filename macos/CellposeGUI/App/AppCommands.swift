import SwiftUI

private struct MainViewModelFocusedKey: FocusedValueKey {
    typealias Value = MainViewModel
}

extension FocusedValues {
    var mainViewModel: MainViewModel? {
        get { self[MainViewModelFocusedKey.self] }
        set { self[MainViewModelFocusedKey.self] = newValue }
    }
}

struct AppCommands: Commands {
    @FocusedValue(\.mainViewModel) private var viewModel

    var body: some Commands {
        CommandGroup(replacing: .newItem) {}
        CommandGroup(replacing: .saveItem) {}

        CommandMenu("File") {
            Button("Load image") {
                Task { await viewModel?.loadImagePanel() }
            }
            .keyboardShortcut("l", modifiers: [.command])

            Button("Load folder") {
                viewModel?.presentLoadFolderPanel()
            }
            .keyboardShortcut("l", modifiers: [.command, .shift])

            Divider()

            Button("Save results") {
                Task { await viewModel?.saveResults() }
            }
            .keyboardShortcut("s", modifiers: [.command])
            .disabled(!(viewModel?.canSaveMasks ?? false))
        }

        CommandMenu("Edit") {
            Button("Undo previous mask/trace") {
                viewModel?.undoAction()
            }
            .keyboardShortcut("z", modifiers: [.command])
            .disabled(true)

            Button("Undo remove mask") {
                viewModel?.undoRemoveAction()
            }
            .keyboardShortcut("y", modifiers: [.command])
            .disabled(true)

            Button("Clear all masks") {
                Task { await viewModel?.clearAllMasks() }
            }
            .keyboardShortcut("0", modifiers: [.command])
            .disabled(!(viewModel?.canSaveMasks ?? false))

            Button("Remove selected cell (Ctrl+Click)") {}
                .disabled(true)

            Button("FYI: Merge cells by Alt+Click") {}
                .disabled(true)
        }

        CommandMenu("Models") {
            Button("Add custom torch model to GUI") {
                Task { await viewModel?.addCustomModel() }
            }

            Button("Remove selected custom model from GUI") {
                Task { await viewModel?.removeSelectedModel() }
            }
            .disabled(viewModel?.isCustomModel != true)

            Button("Train new model with image+masks in folder") {
                viewModel?.showTrainDialog = true
            }
            .keyboardShortcut("t", modifiers: [.command])
            .disabled(!(viewModel?.imageLoaded ?? false))
        }
    }
}
