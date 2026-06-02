import Foundation

struct GuiSyncScope: OptionSet {
    let rawValue: Int

    static let labels = GuiSyncScope(rawValue: 1 << 0)
    static let canvasImage = GuiSyncScope(rawValue: 1 << 1)
    static let canvasMask = GuiSyncScope(rawValue: 1 << 2)
    static let canvasSelection = GuiSyncScope(rawValue: 1 << 3)

    static let selectionChange: GuiSyncScope = [.canvasSelection]
    static let instanceEdit: GuiSyncScope = [.labels, .canvasMask, .canvasSelection]
    static let filterChange: GuiSyncScope = instanceEdit
    static let visibilityEdit: GuiSyncScope = [.labels, .canvasMask]
    static let maskGeometry: GuiSyncScope = [.labels, .canvasMask]
}
