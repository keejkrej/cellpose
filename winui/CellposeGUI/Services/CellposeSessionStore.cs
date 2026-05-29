using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class CellposeSessionStore
{
    public string DefaultPath(string imagePath) =>
        Path.ChangeExtension(imagePath, null) + "_seg.npy";

    public void Save(string path, SessionState session)
    {
        if (session.ImagePath == null || session.Masks == null)
            throw new SidecarException("Nothing to save");

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var masks = session.Masks;
        var colors = MaskEditService.EnsureColors(masks, masks.Colors);
        var outlines = masks.OutlineLabels ?? MaskEditService.ComputeOutlineLabels(
            masks.Labels,
            masks.Width,
            masks.Height);

        var payload = new Dictionary<string, object?>
        {
            ["outlines"] = SegNpyIO.FromLabels(outlines, masks.Width, masks.Height),
            ["masks"] = SegNpyIO.FromLabels(masks.Labels, masks.Width, masks.Height),
            ["colors"] = SegNpyIO.FromColors(colors),
            ["filename"] = session.ImagePath,
            ["flows"] = session.Flows.Select(SegNpyIO.FromArrayPayload).Cast<object?>().ToList(),
            ["flow_threshold"] = session.Segmentation.FlowThreshold,
            ["cellprob_threshold"] = session.Segmentation.CellprobThreshold,
            ["diameter"] = session.Segmentation.Diameter > 0 ? session.Segmentation.Diameter : null,
            ["model_path"] = session.Model is "cpsam" or "0" ? 0 : session.Model,
            ["restore"] = null,
            ["ratio"] = 1.0,
        };

        SegNpyIO.Write(path, payload);
    }

    public LoadedSession Load(string path, ImageLoaderService imageLoader)
    {
        var payload = SegNpyIO.Read(path);
        var sourceImage = ResolveSourceImage(GetString(payload, "filename"), path);
        if (!File.Exists(sourceImage))
            throw new SidecarException($"Source image not found: {sourceImage}");

        return BuildLoadedSession(payload, sourceImage, imageLoader.Load(sourceImage));
    }

    public LoadedSession LoadCompanion(string sessionPath, string imagePath, ImageData image)
    {
        var payload = SegNpyIO.Read(sessionPath);
        var loaded = BuildLoadedSession(payload, imagePath, image);
        if (loaded.Masks.Width != image.Width || loaded.Masks.Height != image.Height)
        {
            throw new SidecarException(
                $"Session mask size {loaded.Masks.Width}x{loaded.Masks.Height} does not match image {image.Width}x{image.Height}");
        }

        return loaded;
    }

    private static LoadedSession BuildLoadedSession(
        Dictionary<string, object?> payload,
        string imagePath,
        ImageData image)
    {
        if (payload.GetValueOrDefault("masks") is not SegNpyArray masksArray)
            throw new SidecarException("Invalid _seg.npy file: missing masks");

        var (labels, width, height) = ReadLabelMatrix(masksArray);
        var colors = payload.GetValueOrDefault("colors") is SegNpyArray colorsArray
            ? SegNpyIO.ReadColors(colorsArray)
            : MaskEditService.DefaultColors(labels.Length == 0 ? 0 : labels.Max());

        var outlines = payload.GetValueOrDefault("outlines") is SegNpyArray outlinesArray
            ? SegNpyIO.ReadLabels(outlinesArray)
            : MaskEditService.ComputeOutlineLabels(labels, width, height);

        var flows = new List<ArrayPayload>();
        if (payload.GetValueOrDefault("flows") is List<object?> flowItems)
        {
            foreach (var item in flowItems)
            {
                if (item is SegNpyArray flowArray)
                    flows.Add(SegNpyIO.ToArrayPayload(flowArray));
            }
        }

        var diameter = payload.GetValueOrDefault("diameter");
        return new LoadedSession
        {
            ImagePath = imagePath,
            Image = image,
            Masks = new MaskData
            {
                Width = width,
                Height = height,
                Labels = labels,
                Colors = colors,
                OutlineLabels = outlines,
            },
            Flows = flows,
            RecomputeMasks = flows.Count > 0,
            Model = FormatModel(payload.GetValueOrDefault("model_path")),
            Segmentation = new SegmentationParameters
            {
                FlowThreshold = GetDouble(payload, "flow_threshold", 0.4),
                CellprobThreshold = GetDouble(payload, "cellprob_threshold", 0.0),
                Diameter = diameter is double d ? d : 0,
                Niter = 200,
                MinSize = 15,
            },
        };
    }

    private static (int[] Labels, int Width, int Height) ReadLabelMatrix(SegNpyArray array)
    {
        var labels = SegNpyIO.ReadLabels(array);
        if (array.Shape.Length == 2)
            return (labels, array.Shape[1], array.Shape[0]);
        if (array.Shape.Length == 3 && array.Shape[0] == 1)
            return (labels, array.Shape[2], array.Shape[1]);
        throw new SidecarException("Invalid mask shape in _seg.npy");
    }

    private static string ResolveSourceImage(string sourceImage, string sessionPath)
    {
        if (Path.IsPathRooted(sourceImage))
            return sourceImage;
        return Path.GetFullPath(Path.Combine(Path.GetDirectoryName(sessionPath)!, sourceImage));
    }

    private static string GetString(Dictionary<string, object?> payload, string key) =>
        payload.GetValueOrDefault(key)?.ToString() ?? "";

    private static double GetDouble(Dictionary<string, object?> payload, string key, double fallback) =>
        payload.GetValueOrDefault(key) switch
        {
            double value => value,
            long value => value,
            int value => value,
            _ => fallback,
        };

    private static string FormatModel(object? value) =>
        value switch
        {
            null => "cpsam",
            0 => "cpsam",
            0L => "cpsam",
            long l when l == 0 => "cpsam",
            int i when i == 0 => "cpsam",
            string s when s is "0" or "" => "cpsam",
            string s => s,
            _ => value.ToString() ?? "cpsam",
        };
}

public sealed class LoadedSession
{
    public string ImagePath { get; init; } = "";
    public ImageData Image { get; init; } = new();
    public MaskData Masks { get; init; } = new();
    public List<ArrayPayload> Flows { get; init; } = [];
    public bool RecomputeMasks { get; init; }
    public string Model { get; init; } = "cpsam";
    public SegmentationParameters Segmentation { get; init; } = new();
}
