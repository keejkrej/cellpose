using CellposeGUI.Models;

namespace CellposeGUI.Services;

public static class MaskEditService
{
    public static MaskData ApplyMasks(int[] labels, int width, int height, byte[][]? existingColors = null)
    {
        Renumber(labels);
        var ncells = labels.Length == 0 ? 0 : labels.Max();
        var colors = EnsureColors(ncells, existingColors);
        var outlines = ComputeOutlineLabels(labels, width, height);
        return new MaskData
        {
            Width = width,
            Height = height,
            Labels = labels,
            Colors = colors,
            OutlineLabels = outlines,
        };
    }

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
        if (masks.Labels.Length == 0)
            return masks;

        var ncells = masks.Labels.Max();
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

    public static int[] ComputeOutlineLabels(int[] labels, int width, int height)
    {
        var outlines = new int[labels.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var label = labels[y * width + x];
                if (label <= 0)
                    continue;
                if (x > 0 && labels[y * width + x - 1] != label)
                {
                    outlines[y * width + x] = label;
                    continue;
                }
                if (x + 1 < width && labels[y * width + x + 1] != label)
                {
                    outlines[y * width + x] = label;
                    continue;
                }
                if (y > 0 && labels[(y - 1) * width + x] != label)
                {
                    outlines[y * width + x] = label;
                    continue;
                }
                if (y + 1 < height && labels[(y + 1) * width + x] != label)
                    outlines[y * width + x] = label;
            }
        }
        return outlines;
    }

    public static MaskData RemoveCells(MaskData masks, IReadOnlyList<int> indices)
    {
        var labels = (int[])masks.Labels.Clone();
        var width = masks.Width;
        var height = masks.Height;
        foreach (var idx in indices.OrderByDescending(v => v))
        {
            if (idx <= 0 || idx > labels.Max())
                continue;
            for (var i = 0; i < labels.Length; i++)
            {
                if (labels[i] == idx)
                    labels[i] = 0;
                else if (labels[i] > idx)
                    labels[i]--;
            }
        }

        var colors = masks.Colors.ToList();
        foreach (var idx in indices.OrderByDescending(v => v))
        {
            if (idx <= 0 || idx > colors.Count)
                continue;
            colors.RemoveAt(idx - 1);
        }

        return ApplyMasks(labels, width, height, colors.ToArray());
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

        var merged = ApplyMasks(labels, masks.Width, masks.Height, masks.Colors);
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
        var labels = masks?.Labels ?? new int[imageWidth * imageHeight];
        var width = masks?.Width ?? imageWidth;
        var height = masks?.Height ?? imageHeight;
        if (labels.Length != width * height)
            labels = new int[width * height];

        var colors = masks?.Colors?.ToList() ?? [];
        var fillColor = color ?? ColorForClass(classId);
        var filledRows = new List<int>();
        var filledCols = new List<int>();
        var outlineRows = new List<int>();
        var outlineCols = new List<int>();

        foreach (var stroke in strokes)
        {
            if (stroke.Length == 0)
                continue;

            var points = stroke.Select(p => new Point((int)p[2], (int)p[1])).ToArray();
            if (points.Length < 3)
                continue;

            var minX = points.Min(p => p.X);
            var maxX = points.Max(p => p.X);
            var minY = points.Min(p => p.Y);
            var maxY = points.Max(p => p.Y);
            var localWidth = maxX - minX + 5;
            var localHeight = maxY - minY + 5;
            var bitmap = new bool[localWidth * localHeight];

            FillPolygon(bitmap, localWidth, localHeight, points, minX - 2, minY - 2);

            var areaRows = new List<int>();
            var areaCols = new List<int>();
            for (var y = 0; y < localHeight; y++)
            {
                for (var x = 0; x < localWidth; x++)
                {
                    if (!bitmap[y * localWidth + x])
                        continue;
                    var yr = y + minY - 2;
                    var xc = x + minX - 2;
                    if (yr < 0 || xc < 0 || yr >= height || xc >= width)
                        continue;
                    areaRows.Add(yr);
                    areaCols.Add(xc);
                }
            }

            if (areaRows.Count < 10)
                return null;

            var overlap = areaRows.Zip(areaCols, (r, c) => labels[r * width + c] > 0).Count(v => v);
            if (overlap > 0 && areaRows.Count - overlap < 10)
                return null;
        }

        var nextLabel = labels.Max() + 1;
        foreach (var stroke in strokes)
        {
            var points = stroke.Select(p => new Point((int)p[2], (int)p[1])).ToArray();
            if (points.Length < 3)
                continue;
            var minX = points.Min(p => p.X);
            var maxX = points.Max(p => p.X);
            var minY = points.Min(p => p.Y);
            var maxY = points.Max(p => p.Y);
            var localWidth = maxX - minX + 5;
            var localHeight = maxY - minY + 5;
            var bitmap = new bool[localWidth * localHeight];
            FillPolygon(bitmap, localWidth, localHeight, points, minX - 2, minY - 2);
            TraceBoundary(bitmap, localWidth, localHeight, minX - 2, minY - 2, outlineRows, outlineCols);
            for (var y = 0; y < localHeight; y++)
            {
                for (var x = 0; x < localWidth; x++)
                {
                    if (!bitmap[y * localWidth + x])
                        continue;
                    var yr = y + minY - 2;
                    var xc = x + minX - 2;
                    if (yr < 0 || xc < 0 || yr >= height || xc >= width)
                        continue;
                    if (labels[yr * width + xc] > 0)
                        continue;
                    labels[yr * width + xc] = nextLabel;
                    filledRows.Add(yr);
                    filledCols.Add(xc);
                }
            }
        }

        if (filledRows.Count == 0)
            return null;

        foreach (var (row, col) in outlineRows.Zip(outlineCols))
        {
            if (row >= 0 && col >= 0 && row < height && col < width)
                labels[row * width + col] = nextLabel;
        }

        colors.Add(fillColor);
        return ApplyMasks(labels, width, height, colors.ToArray());
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

    private static void FillPolygon(bool[] bitmap, int width, int height, Point[] points, int offsetX, int offsetY)
    {
        var minY = points.Min(p => p.Y);
        var maxY = points.Max(p => p.Y);
        for (var y = minY; y <= maxY; y++)
        {
            var intersections = new List<int>();
            for (var i = 0; i < points.Length; i++)
            {
                var p1 = points[i];
                var p2 = points[(i + 1) % points.Length];
                if (p1.Y == p2.Y)
                    continue;
                if (y < Math.Min(p1.Y, p2.Y) || y >= Math.Max(p1.Y, p2.Y))
                    continue;
                var x = p1.X + (double)(y - p1.Y) / (p2.Y - p1.Y) * (p2.X - p1.X);
                intersections.Add((int)Math.Round(x));
            }

            intersections.Sort();
            for (var i = 0; i + 1 < intersections.Count; i += 2)
            {
                var xStart = Math.Max(intersections[i], points.Min(p => p.X));
                var xEnd = Math.Min(intersections[i + 1], points.Max(p => p.X));
                for (var x = xStart; x <= xEnd; x++)
                {
                    var lx = x - offsetX;
                    var ly = y - offsetY;
                    if (lx < 0 || ly < 0 || lx >= width || ly >= height)
                        continue;
                    bitmap[ly * width + lx] = true;
                }
            }
        }
    }

    private static void TraceBoundary(
        bool[] bitmap,
        int width,
        int height,
        int offsetX,
        int offsetY,
        List<int> rows,
        List<int> cols)
    {
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                if (!bitmap[y * width + x])
                    continue;
                var onEdge = x == 0 || y == 0 || x + 1 == width || y + 1 == height
                    || !bitmap[y * width + x - 1]
                    || !bitmap[y * width + x + 1]
                    || !bitmap[(y - 1) * width + x]
                    || !bitmap[(y + 1) * width + x];
                if (!onEdge)
                    continue;
                rows.Add(y + offsetY);
                cols.Add(x + offsetX);
            }
        }
    }

    private readonly struct Point(int x, int y)
    {
        public int X { get; } = x;
        public int Y { get; } = y;
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

        var minY = height;
        var minX = width;
        var maxY = -1;
        var maxX = -1;
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                if (labels[y * width + x] != label)
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
        x0 = Math.Clamp(Math.Min(x0, x1), 0, width - 1);
        x1 = Math.Clamp(Math.Max(x0, x1), 0, width);
        y0 = Math.Clamp(Math.Min(y0, y1), 0, height - 1);
        y1 = Math.Clamp(Math.Max(y0, y1), 0, height);
        if (x1 <= x0 || y1 <= y0)
            return [];

        var candidates = new HashSet<int>();
        for (var y = y0; y < y1; y++)
        {
            for (var x = x0; x < x1; x++)
            {
                var label = labels[y * width + x];
                if (label > 0)
                    candidates.Add(label);
            }
        }

        var fullyCovered = new List<int>();
        foreach (var label in candidates.OrderBy(v => v))
        {
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    if (labels[y * width + x] != label)
                        continue;
                    if (y < y0 || y >= y1 || x < x0 || x >= x1)
                        goto nextCandidate;
                }
            }

            if (filterClassId != null && classIds != null)
            {
                var row = label - 1;
                if (row < 0 || row >= classIds.Count || classIds[row] != filterClassId.Value)
                    goto nextCandidate;
            }

            fullyCovered.Add(label);
            nextCandidate: ;
        }

        return fullyCovered;
    }
}
