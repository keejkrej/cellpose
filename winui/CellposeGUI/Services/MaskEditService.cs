using CellposeGUI.Models;

namespace CellposeGUI.Services;

public static class MaskEditService
{
    private const int MinNewCellPixels = 10;

    public static MaskData ApplyMasks(int[] labels, int width, int height, byte[][]? existingColors = null) =>
        BuildMaskData(new LabelGrid(width, height, (int[])labels.Clone()), existingColors);

    public static byte[][] EnsureColors(MaskData masks, byte[][]? existingColors) =>
        EnsureColors(masks.Labels.Max(), existingColors);

    public static byte[][] EnsureColors(int ncells, byte[][]? existingColors)
    {
        if (existingColors != null && existingColors.Length >= ncells)
            return existingColors;
        return DefaultColors(ncells);
    }

    public static byte[][] DefaultColors(int ncells)
    {
        var colors = new byte[Math.Max(ncells, 0)][];
        for (var i = 0; i < ncells; i++)
            colors[i] = ColorForClass(i);
        return colors;
    }

    public static byte[] ColorForClass(int classId)
    {
        var random = new Random(42 + classId * 7919);
        return [(byte)random.Next(50, 256), (byte)random.Next(50, 256), (byte)random.Next(50, 256)];
    }

    public static MaskData WithClassColors(MaskData masks, IReadOnlyList<int> classIds)
    {
        var ncells = masks.Labels.Length == 0 ? 0 : masks.Labels.Max();
        if (ncells <= 0)
            return masks;

        var colors = new byte[ncells][];
        for (var i = 0; i < ncells; i++)
        {
            var classId = i < classIds.Count ? classIds[i] : 0;
            colors[i] = ColorForClass(classId);
        }

        return new MaskData
        {
            Width = masks.Width,
            Height = masks.Height,
            Labels = masks.Labels,
            Colors = colors,
            OutlineLabels = masks.OutlineLabels,
        };
    }

    public static int[] ComputeOutlineLabels(int[] labels, int width, int height) =>
        ComputeOutlineLabels(new LabelGrid(width, height, labels));

    public static MaskData RemoveCells(MaskData masks, IReadOnlyList<int> indices)
    {
        var labels = (int[])masks.Labels.Clone();
        foreach (var idx in indices.OrderByDescending(v => v))
            RemoveCellLabel(labels, idx);

        var colors = masks.Colors.ToList();
        foreach (var idx in indices.OrderByDescending(v => v))
        {
            if (idx <= 0 || idx > colors.Count)
                continue;
            colors.RemoveAt(idx - 1);
        }

        return BuildMaskData(new LabelGrid(masks.Width, masks.Height, labels), colors.ToArray());
    }

    public static MaskData MergeCells(MaskData masks, int source, int target)
    {
        if (source == target)
            return masks;

        var labels = (int[])masks.Labels.Clone();
        for (var i = 0; i < labels.Length; i++)
        {
            if (labels[i] == source)
                labels[i] = target;
        }

        var merged = BuildMaskData(new LabelGrid(masks.Width, masks.Height, labels), masks.Colors);
        return RemoveCells(merged, [source]);
    }

    public static MaskData? AddMaskFromStrokes(
        MaskData? masks,
        int imageWidth,
        int imageHeight,
        IReadOnlyList<double[][]> strokes,
        int classId,
        byte[]? color = null)
    {
        var grid = LabelGrid.FromMaskOrEmpty(masks, imageWidth, imageHeight);
        var labels = grid.Labels;
        var colors = masks?.Colors?.ToList() ?? [];
        var fillColor = color ?? ColorForClass(classId);

        var stagedStrokes = new List<StrokeRasterizer.RasterizedStroke>();
        foreach (var stroke in strokes)
        {
            if (!StrokeRasterizer.TryParsePoints(stroke, out var points))
                continue;

            var raster = StrokeRasterizer.Rasterize(points, grid);
            if (!StrokeRasterizer.IsValidNewCellArea(raster.ImagePixels, grid, MinNewCellPixels))
                return null;

            stagedStrokes.Add(raster);
        }

        var nextLabel = grid.MaxLabel + 1;
        var filledPixels = 0;
        var outlinePixels = new List<(int Row, int Col)>();

        foreach (var raster in stagedStrokes)
        {
            outlinePixels.AddRange(StrokeRasterizer.TraceBoundary(raster));

            foreach (var (row, col) in raster.ImagePixels)
            {
                var index = grid.Index(col, row);
                if (labels[index] > 0)
                    continue;

                labels[index] = nextLabel;
                filledPixels++;
            }
        }

        if (filledPixels == 0)
            return null;

        foreach (var (row, col) in outlinePixels)
        {
            if (!grid.InBounds(col, row))
                continue;
            labels[grid.Index(col, row)] = nextLabel;
        }

        colors.Add(fillColor);
        return BuildMaskData(new LabelGrid(grid.Width, grid.Height, labels), colors.ToArray());
    }

    public static (int X0, int Y0, int X1, int Y1)? CellBounds(
        int[] labels,
        int width,
        int height,
        int label,
        int margin = 1)
    {
        if (label <= 0)
            return null;

        var grid = new LabelGrid(width, height, labels);
        var minY = height;
        var minX = width;
        var maxY = -1;
        var maxX = -1;

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                if (grid.At(x, y) != label)
                    continue;

                minY = Math.Min(minY, y);
                minX = Math.Min(minX, x);
                maxY = Math.Max(maxY, y);
                maxX = Math.Max(maxX, x);
            }
        }

        if (maxY < 0)
            return null;

        return (
            Math.Max(0, minX - margin),
            Math.Max(0, minY - margin),
            Math.Min(width - 1, maxX + margin),
            Math.Min(height - 1, maxY + margin));
    }

    public static List<int> CellsFullyInRect(
        int[] labels,
        int width,
        int height,
        int x0,
        int y0,
        int x1,
        int y1,
        int? filterClassId = null,
        IReadOnlyList<int>? classIds = null)
    {
        var rect = PixelRect.Normalize(x0, y0, x1, y1, width, height);
        if (rect == null)
            return [];

        var grid = new LabelGrid(width, height, labels);
        var bounds = rect.Value;
        var candidates = new HashSet<int>();

        for (var y = bounds.Top; y < bounds.Bottom; y++)
        {
            for (var x = bounds.Left; x < bounds.Right; x++)
            {
                var label = grid.At(x, y);
                if (label > 0)
                    candidates.Add(label);
            }
        }

        return candidates
            .Order()
            .Where(label => IsCellFullyInsideRect(grid, label, bounds))
            .Where(label => MatchesClassFilter(label, filterClassId, classIds))
            .ToList();
    }

    private static MaskData BuildMaskData(LabelGrid grid, byte[][]? existingColors)
    {
        var labels = (int[])grid.Labels.Clone();
        Renumber(labels);
        var ncells = labels.Length == 0 ? 0 : labels.Max();

        return new MaskData
        {
            Width = grid.Width,
            Height = grid.Height,
            Labels = labels,
            Colors = EnsureColors(ncells, existingColors),
            OutlineLabels = ComputeOutlineLabels(labels, grid.Width, grid.Height),
        };
    }

    private static int[] ComputeOutlineLabels(LabelGrid grid)
    {
        var outlines = new int[grid.Labels.Length];
        for (var y = 0; y < grid.Height; y++)
        {
            for (var x = 0; x < grid.Width; x++)
            {
                var label = grid.At(x, y);
                if (label <= 0)
                    continue;

                if (HasDifferentNeighbor(grid, x, y, label))
                    outlines[grid.Index(x, y)] = label;
            }
        }

        return outlines;
    }

    private static bool HasDifferentNeighbor(LabelGrid grid, int x, int y, int label)
    {
        if (x > 0 && grid.At(x - 1, y) != label)
            return true;
        if (x + 1 < grid.Width && grid.At(x + 1, y) != label)
            return true;
        if (y > 0 && grid.At(x, y - 1) != label)
            return true;
        if (y + 1 < grid.Height && grid.At(x, y + 1) != label)
            return true;
        return false;
    }

    private static void Renumber(int[] labels)
    {
        var max = labels.Max();
        if (max <= 0)
            return;

        var present = new bool[max + 1];
        foreach (var label in labels)
        {
            if (label > 0)
                present[label] = true;
        }

        var next = 1;
        var remap = new int[max + 1];
        for (var i = 1; i <= max; i++)
        {
            if (!present[i])
                continue;
            remap[i] = next++;
        }

        for (var i = 0; i < labels.Length; i++)
        {
            if (labels[i] > 0)
                labels[i] = remap[labels[i]];
        }
    }

    private static void RemoveCellLabel(int[] labels, int idx)
    {
        if (idx <= 0 || idx > labels.Max())
            return;

        for (var i = 0; i < labels.Length; i++)
        {
            if (labels[i] == idx)
                labels[i] = 0;
            else if (labels[i] > idx)
                labels[i]--;
        }
    }

    private static bool IsCellFullyInsideRect(LabelGrid grid, int label, PixelRect rect)
    {
        for (var y = 0; y < grid.Height; y++)
        {
            for (var x = 0; x < grid.Width; x++)
            {
                if (grid.At(x, y) != label)
                    continue;
                if (!rect.Contains(x, y))
                    return false;
            }
        }

        return true;
    }

    private static bool MatchesClassFilter(
        int label,
        int? filterClassId,
        IReadOnlyList<int>? classIds)
    {
        if (filterClassId == null || classIds == null)
            return true;

        var row = label - 1;
        return row >= 0 && row < classIds.Count && classIds[row] == filterClassId.Value;
    }
}
