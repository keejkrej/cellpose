using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class CellposeSessionStore
{
    private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };

    public string DefaultPath(string imagePath) =>
        Path.ChangeExtension(imagePath, null) + "_seg.cellpose";

    public void Save(string path, SessionState session)
    {
        if (session.ImagePath == null || session.Masks == null)
            throw new SidecarException("Nothing to save");

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        if (File.Exists(path))
            File.Delete(path);

        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        var arrays = new Dictionary<string, ManifestArray>();
        WriteUInt16Array(archive, arrays, "masks", session.Masks);

        for (var i = 0; i < session.Flows.Count; i++)
            WriteRawArray(archive, arrays, $"flows_{i}", session.Flows[i]);

        var colors = MaskEditService.EnsureColors(session.Masks, session.Masks.Colors);
        WriteColorArray(archive, arrays, colors);

        var manifest = new SessionManifest
        {
            Version = 1,
            SourceImage = session.ImagePath,
            Segmentation = session.Segmentation,
            Model = session.Model,
            RecomputeMasks = session.RecomputeMasks,
            Arrays = arrays,
        };

        var manifestEntry = archive.CreateEntry("manifest.json", CompressionLevel.Optimal);
        using (var stream = manifestEntry.Open())
        using (var writer = new StreamWriter(stream, Utf8NoBom))
            writer.Write(JsonSerializer.Serialize(manifest, JsonOptions));
    }

    public LoadedSession Load(string path, ImageLoaderService imageLoader)
    {
        using var archive = ZipFile.OpenRead(path);
        var manifestEntry = archive.GetEntry("manifest.json")
            ?? throw new SidecarException("Invalid session file: missing manifest.json");

        SessionManifest manifest;
        using (var stream = manifestEntry.Open())
            manifest = JsonSerializer.Deserialize<SessionManifest>(stream, JsonOptions)
                       ?? throw new SidecarException("Invalid session manifest");

        if (manifest.Version != 1)
            throw new SidecarException($"Unsupported session version: {manifest.Version}");

        var sourceImage = manifest.SourceImage;
        if (!Path.IsPathRooted(sourceImage))
            sourceImage = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(path)!, sourceImage));

        if (!File.Exists(sourceImage))
            throw new SidecarException($"Source image not found: {sourceImage}");

        var masksEntry = manifest.Arrays["masks"];
        var labels = ReadMaskLabels(archive, masksEntry);
        byte[][] colors = manifest.Arrays.TryGetValue("colors", out var colorsEntry)
            ? ReadColors(archive, colorsEntry)
            : MaskEditService.DefaultColors(labels.Max());

        var flows = new List<ArrayPayload>();
        for (var i = 0; ; i++)
        {
            if (!manifest.Arrays.TryGetValue($"flows_{i}", out var flowEntry))
                break;
            flows.Add(ReadPayload(archive, flowEntry));
        }

        var width = masksEntry.Shape[^1];
        var height = masksEntry.Shape[^2];
        var maskData = new MaskData
        {
            Width = width,
            Height = height,
            Labels = labels,
            Colors = colors,
            OutlineLabels = MaskEditService.ComputeOutlineLabels(labels, width, height),
        };

        return new LoadedSession
        {
            ImagePath = sourceImage,
            Image = imageLoader.Load(sourceImage),
            Masks = maskData,
            Flows = flows,
            RecomputeMasks = manifest.RecomputeMasks,
            Model = manifest.Model,
            Segmentation = manifest.Segmentation,
        };
    }

    private static void WriteUInt16Array(
        ZipArchive archive,
        Dictionary<string, ManifestArray> arrays,
        string name,
        MaskData masks)
    {
        var raw = new byte[masks.Labels.Length * sizeof(ushort)];
        var span = raw.AsSpan();
        for (var i = 0; i < masks.Labels.Length; i++)
        {
            var value = (ushort)Math.Clamp(masks.Labels[i], 0, ushort.MaxValue);
            BinaryPrimitives.WriteUInt16LittleEndian(span[(i * sizeof(ushort))..], value);
        }

        var relPath = $"arrays/{name}.uint16";
        var entry = archive.CreateEntry(relPath, CompressionLevel.Optimal);
        using (var stream = entry.Open())
            stream.Write(raw);

        arrays[name] = new ManifestArray
        {
            File = relPath,
            Dtype = "uint16",
            Shape = [masks.Height, masks.Width],
        };
    }

    private static void WriteRawArray(
        ZipArchive archive,
        Dictionary<string, ManifestArray> arrays,
        string name,
        ArrayPayload payload)
    {
        var relPath = $"arrays/{name}.{PayloadExtension(payload.Dtype)}";
        var raw = ArrayCodec.Decode(payload);
        var entry = archive.CreateEntry(relPath, CompressionLevel.Optimal);
        using (var stream = entry.Open())
            stream.Write(raw);

        arrays[name] = new ManifestArray
        {
            File = relPath,
            Dtype = payload.Dtype,
            Shape = payload.Shape,
        };
    }

    private static void WriteColorArray(
        ZipArchive archive,
        Dictionary<string, ManifestArray> arrays,
        byte[][] colors)
    {
        var raw = new byte[colors.Length * 3];
        for (var i = 0; i < colors.Length; i++)
        {
            raw[i * 3] = colors[i][0];
            raw[i * 3 + 1] = colors[i][1];
            raw[i * 3 + 2] = colors[i][2];
        }

        var relPath = "arrays/colors.uint8";
        var entry = archive.CreateEntry(relPath, CompressionLevel.Optimal);
        using (var stream = entry.Open())
            stream.Write(raw);

        arrays["colors"] = new ManifestArray
        {
            File = relPath,
            Dtype = "uint8",
            Shape = [colors.Length, 3],
        };
    }

    private static int[] ReadMaskLabels(ZipArchive archive, ManifestArray entry)
    {
        var raw = ReadRaw(archive, entry);
        var labels = new int[raw.Length / sizeof(ushort)];
        for (var i = 0; i < labels.Length; i++)
            labels[i] = BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(i * sizeof(ushort)));
        return labels;
    }

    private static byte[][] ReadColors(ZipArchive archive, ManifestArray entry)
    {
        var raw = ReadRaw(archive, entry);
        var count = entry.Shape[0];
        var colors = new byte[count][];
        for (var i = 0; i < count; i++)
            colors[i] = [raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]];
        return colors;
    }

    private static ArrayPayload ReadPayload(ZipArchive archive, ManifestArray entry)
    {
        var raw = ReadRaw(archive, entry);
        return ArrayCodec.EncodeRaw(raw, entry.Dtype, entry.Shape);
    }

    private static byte[] ReadRaw(ZipArchive archive, ManifestArray entry)
    {
        var zipEntry = archive.GetEntry(entry.File)
            ?? throw new SidecarException($"Missing array entry: {entry.File}");
        using var stream = zipEntry.Open();
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        return memory.ToArray();
    }

    private static string PayloadExtension(string dtype) => dtype switch
    {
        "uint8" => "uint8",
        "float32" => "float32",
        "float64" => "float64",
        _ => dtype,
    };

    private sealed class SessionManifest
    {
        public int Version { get; set; }
        [JsonPropertyName("source_image")]
        public string SourceImage { get; set; } = "";
        public SegmentationParameters Segmentation { get; set; } = new();
        public string Model { get; set; } = "cpsam";
        [JsonPropertyName("recompute_masks")]
        public bool RecomputeMasks { get; set; }
        public Dictionary<string, ManifestArray> Arrays { get; set; } = new();
    }

    private sealed class ManifestArray
    {
        public string File { get; set; } = "";
        public string Dtype { get; set; } = "";
        public int[] Shape { get; set; } = [];
    }
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
