using CellposeGUI.Models;

namespace CellposeGUI.Services;

public readonly record struct EllipseDiameters(double Major, double Minor)
{
    public static EllipseDiameters Invalid => new(double.NaN, double.NaN);
}

public static class MaskMetricsService
{
    public static EllipseDiameters[] EllipseDiameters(MaskData masks, int ncells)
    {
        if (ncells <= 0)
            return [];

        var coordsByLabel = new List<(double Y, double X)>[ncells + 1];
        for (var label = 0; label <= ncells; label++)
            coordsByLabel[label] = [];

        for (var y = 0; y < masks.Height; y++)
        {
            var rowOffset = y * masks.Width;
            for (var x = 0; x < masks.Width; x++)
            {
                var label = masks.Labels[rowOffset + x];
                if (label <= 0 || label > ncells)
                    continue;
                coordsByLabel[label].Add((y, x));
            }
        }

        var result = new EllipseDiameters[ncells];
        for (var label = 1; label <= ncells; label++)
        {
            var coords = coordsByLabel[label];
            result[label - 1] = coords.Count >= 5
                ? EquivalentEllipseDiameters(coords)
                : new EllipseDiameters(double.NaN, double.NaN);
        }

        return result;
    }

    public static EllipseDiameters EquivalentEllipseDiameters(IReadOnlyList<(double Y, double X)> coords)
    {
        var count = coords.Count;
        var yMean = 0.0;
        var xMean = 0.0;
        foreach (var (y, x) in coords)
        {
            yMean += y;
            xMean += x;
        }

        yMean /= count;
        xMean /= count;

        var mu20 = 0.0;
        var mu02 = 0.0;
        var mu11 = 0.0;
        foreach (var (y, x) in coords)
        {
            var dy = y - yMean;
            var dx = x - xMean;
            mu20 += dx * dx;
            mu02 += dy * dy;
            mu11 += dx * dy;
        }

        mu20 /= count;
        mu02 /= count;
        mu11 /= count;

        var common = Math.Sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11);
        var lambda1 = Math.Max((mu20 + mu02 + common) / 2, 0);
        var lambda2 = Math.Max((mu20 + mu02 - common) / 2, 0);
        return new EllipseDiameters(4 * Math.Sqrt(lambda1), 4 * Math.Sqrt(lambda2));
    }
}
