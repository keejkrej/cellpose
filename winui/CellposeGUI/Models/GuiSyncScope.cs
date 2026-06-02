namespace CellposeGUI.Models;

[Flags]
public enum GuiSyncScope
{
    None = 0,
    Labels = 1 << 0,
    VisibilityHeader = 1 << 1,
    CanvasImage = 1 << 2,
    CanvasMask = 1 << 3,
    CanvasSelection = 1 << 4,
    Ellipses = 1 << 5,

    SelectionChange = CanvasSelection,
    InstanceEdit = Labels | CanvasMask | CanvasSelection,
    FilterChange = InstanceEdit,
    VisibilityEdit = Labels | CanvasMask | VisibilityHeader,
    MaskGeometry = Labels | CanvasMask | Ellipses,
}
