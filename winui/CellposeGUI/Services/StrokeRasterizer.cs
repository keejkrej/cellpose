namespace CellposeGUI.Services;

internal static class StrokeRasterizer
{
    private const int LocalBitmapPadding = 5;
    private const int ImageOffsetPadding = 2;

    internal readonly record struct Point(int X, int Y);

    internal sealed class RasterizedStroke
    {
        public required bool[] Bitmap { get; init; }
        public required int LocalWidth { get; init; }
        public required int LocalHeight { get; init; }
        public required int OffsetX { get; init; }
        public required int OffsetY { get; init; }
        public required List<(int Row, int Col)> ImagePixels { get; init; }
    }

    public static bool TryParsePoints(IReadOnlyList<double[]> stroke, out Point[] points)
    {
        points = [];
        if (stroke.Count == 0)
            return false;

        points = stroke.Select(p => new Point((int)p[2], (int)p[1])).ToArray();
        return points.Length >= 3;
    }

    public static RasterizedStroke Rasterize(Point[] points, LabelGrid grid)
    {
        var minX = points.Min(p => p.X);
        var maxX = points.Max(p => p.X);
        var minY = points.Min(p => p.Y);
        var maxY = points.Max(p => p.Y);
        var localWidth = maxX - minX + LocalBitmapPadding;
        var localHeight = maxY - minY + LocalBitmapPadding;
        var offsetX = minX - ImageOffsetPadding;
        var offsetY = minY - ImageOffsetPadding;
        var bitmap = new bool[localWidth * localHeight];

        FillPolygon(bitmap, localWidth, localHeight, points, offsetX, offsetY);

        var imagePixels = new List<(int Row, int Col)>();
        for (var y = 0; y < localHeight; y++)
        {
            for (var x = 0; x < localWidth; x++)
            {
                if (!bitmap[y * localWidth + x])
                    continue;

                var row = y + offsetY;
                var col = x + offsetX;
                if (!grid.InBounds(col, row))
                    continue;

                imagePixels.Add((row, col));
            }
        }

        return new RasterizedStroke
        {
            Bitmap = bitmap,
            LocalWidth = localWidth,
            LocalHeight = localHeight,
            OffsetX = offsetX,
            OffsetY = offsetY,
            ImagePixels = imagePixels,
        };
    }

    public static bool IsValidNewCellArea(
        IReadOnlyList<(int Row, int Col)> imagePixels,
        LabelGrid grid,
        int minPixels = 10)
    {
        if (imagePixels.Count < minPixels)
            return false;

        var overlap = imagePixels.Count(p => grid.At(p.Col, p.Row) > 0);
        return overlap == 0 || imagePixels.Count - overlap >= minPixels;
    }

    public static List<(int Row, int Col)> TraceBoundary(RasterizedStroke stroke)
    {
        var boundary = new List<(int Row, int Col)>();
        var bitmap = stroke.Bitmap;
        var width = stroke.LocalWidth;
        var height = stroke.LocalHeight;

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

                boundary.Add((y + stroke.OffsetY, x + stroke.OffsetX));
            }
        }

        return boundary;
    }

    private static void FillPolygon(
        bool[] bitmap,
        int width,
        int height,
        Point[] points,
        int offsetX,
        int offsetY)
    {
        var minY = points.Min(p => p.Y);
        var maxY = points.Max(p => p.Y);
        var minPointX = points.Min(p => p.X);
        var maxPointX = points.Max(p => p.X);

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
                var xStart = Math.Max(intersections[i], minPointX);
                var xEnd = Math.Min(intersections[i + 1], maxPointX);
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
}
