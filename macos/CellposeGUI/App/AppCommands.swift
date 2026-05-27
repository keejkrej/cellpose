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
            Button("Load image (*.tif, *.png, *.jpg)") {
                Task { await viewModel?.loadImagePanel() }
            }
            .keyboardShortcut("l", modifiers: [.command])

            Button("Load folder with pattern...") {
                viewModel?.presentLoadFolderPanel()
            }
            .keyboardShortcut("l", modifiers: [.command, .shift])

            Toggle("Autoload masks from _masks.tif file", isOn: Binding(
                get: { viewModel?.autoloadMasks ?? false },
                set: { viewModel?.autoloadMasks = $0 }
            ))

            Toggle("Disable autosave _seg.cellpose file", isOn: Binding(
                get: { viewModel?.disableAutosave ?? false },
                set: { viewModel?.disableAutosave = $0 }
            ))

            Button("Load masks (*.tif, *.png, *.jpg)") {
                Task { await viewModel?.loadMasksPanel() }
            }
            .keyboardShortcut("m", modifiers: [.command])
            .disabled(!(viewModel?.imageLoaded ?? false))

            Button("Load processed/labelled image (*_seg.cellpose)") {
                Task { await viewModel?.loadSegPanel() }
            }
            .keyboardShortcut("p", modifiers: [.command])

            Divider()

            Button("Save masks and image (as *_seg.cellpose)") {
                Task { await viewModel?.saveSeg() }
            }
            .keyboardShortcut("s", modifiers: [.command])
            .disabled(!(viewModel?.canSaveMasks ?? false))

            Button("Save masks as PNG/tif") {
                Task { await viewModel?.exportMasks() }
            }
            .keyboardShortcut("n", modifiers: [.command])
            .disabled(!(viewModel?.canSaveMasks ?? false))

            Button("Save outlines as text for imageJ") {
                Task { await viewModel?.exportOutlines() }
            }
            .keyboardShortcut("o", modifiers: [.command])
            .disabled(!(viewModel?.canSaveMasks ?? false))

            Button("Save outlines as .zip archive of ROI files for ImageJ") {
                Task { await viewModel?.exportROIs() }
            }
            .keyboardShortcut("r", modifiers: [.command])
            .disabled(!(viewModel?.canSaveMasks ?? false))

            Button("Save flows and cellprob as tif") {
                Task { await viewModel?.exportFlows() }
            }
            .keyboardShortcut("f", modifiers: [.command])
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
