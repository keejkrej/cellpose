using CellposeGUI.Models;

namespace CellposeGUI.Services;

internal readonly record struct LabelGrid(int Width, int Height, int[] Labels)
{
    public int PixelCount => Labels.Length;

    public int Index(int x, int y) => y * Width + x;

    public int At(int x, int y) => Labels[Index(x, y)];

    public bool InBounds(int x, int y) => x >= 0 && y >= 0 && x < Width && y < Height;

    public int MaxLabel => Labels.Length == 0 ? 0 : Labels.Max();

    public static LabelGrid Empty(int width, int height) =>
        new(width, height, new int[width * height]);

    public static LabelGrid FromMaskOrEmpty(
        MaskData? masks,
        int imageWidth,
        int imageHeight)
    {
        if (masks == null)
            return Empty(imageWidth, imageHeight);

        var labels = masks.Labels;
        if (labels.Length != masks.Width * masks.Height)
            return Empty(masks.Width, masks.Height);

        return new LabelGrid(masks.Width, masks.Height, (int[])labels.Clone());
    }
}

internal readonly record struct PixelRect(int Left, int Top, int Right, int Bottom)
{
    public static PixelRect? Normalize(int x0, int y0, int x1, int y1, int width, int height)
    {
        var left = Math.Clamp(Math.Min(x0, x1), 0, width - 1);
        var right = Math.Clamp(Math.Max(x0, x1), 0, width);
        var top = Math.Clamp(Math.Min(y0, y1), 0, height - 1);
        var bottom = Math.Clamp(Math.Max(y0, y1), 0, height);
        if (right <= left || bottom <= top)
            return null;

        return new PixelRect(left, top, right, bottom);
    }

    public bool Contains(int x, int y) =>
        x >= Left && x < Right && y >= Top && y < Bottom;
}
