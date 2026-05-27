using System.Buffers.Binary;
using System.IO.Compression;
using System.Globalization;
using System.Text;
using BitMiracle.LibTiff.Classic;
using CellposeGUI.Models;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace CellposeGUI.Services;

public sealed class ExportService
{
    public void ExportMasks(MaskData masks, string path, string format)
    {
        if (format.Equals("tif", StringComparison.OrdinalIgnoreCase) ||
            format.Equals("tiff", StringComparison.OrdinalIgnoreCase))
        {
            if (!path.EndsWith(".tif", StringComparison.OrdinalIgnoreCase) &&
                !path.EndsWith(".tiff", StringComparison.OrdinalIgnoreCase))
                path = Path.ChangeExtension(path, ".tif");
            WriteUInt16Tiff(path, masks.Labels, masks.Width, masks.Height);
            return;
        }

        if (!path.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            path = Path.ChangeExtension(path, ".png");

        using var image = new Image<Rgb24>(masks.Width, masks.Height);
        for (var y = 0; y < masks.Height; y++)
        {
            for (var x = 0; x < masks.Width; x++)
            {
                var label = masks.Labels[y * masks.Width + x];
                var value = (byte)Math.Clamp(label, 0, 255);
                image[x, y] = new Rgb24(value, value, value);
            }
        }

        image.SaveAsPng(path);
    }

    public void ExportOutlines(MaskData masks, string path)
    {
        if (!path.EndsWith(".txt", StringComparison.OrdinalIgnoreCase))
            path = Path.ChangeExtension(path, ".txt");

        var outlines = ExtractOutlines(masks);
        var builder = new StringBuilder();
        foreach (var (label, points) in outlines)
        {
            builder.Append(label);
            foreach (var (x, y) in points)
                builder.Append(CultureInfo.InvariantCulture, $" {x} {y}");
            builder.AppendLine();
        }

        File.WriteAllText(path, builder.ToString());
    }

    public void ExportFlows(IReadOnlyList<ArrayPayload> flows, string pathBase)
    {
        if (flows.Count < 4)
            throw new SidecarException("Flows not available for export");

        var basePath = Path.ChangeExtension(pathBase, null);
        WriteFloatTiff(basePath + "_cp_flows.tif", ArrayCodec.Decode(flows[0]), flows[0].Shape);
        WriteByteTiff(basePath + "_cp_cellprob.tif", ArrayCodec.Decode(flows[1]), flows[1].Shape);
    }

    public void ExportRois(MaskData masks, string path)
    {
        if (!path.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
            path = Path.ChangeExtension(path, ".zip");

        var outlines = ExtractOutlines(masks);
        if (File.Exists(path))
            File.Delete(path);

        using var archive = ZipFile.Open(path, ZipArchiveMode.Create);
        foreach (var (label, points) in outlines)
        {
            var entry = archive.CreateEntry($"{label:0000}.roi", CompressionLevel.Optimal);
            using var stream = entry.Open();
            using var writer = new BinaryWriter(stream);
            WriteImageJRoi(writer, points, label);
        }
    }

    private static List<(int Label, List<(int X, int Y)> Points)> ExtractOutlines(MaskData masks)
    {
        var result = new List<(int, List<(int, int)>)>();
        var max = masks.Labels.Max();
        for (var label = 1; label <= max; label++)
        {
            var points = new List<(int, int)>();
            for (var y = 0; y < masks.Height; y++)
            {
                for (var x = 0; x < masks.Width; x++)
                {
                    if (masks.OutlineLabels?[y * masks.Width + x] == label)
                        points.Add((x, y));
                }
            }

            if (points.Count > 0)
                result.Add((label, points));
        }

        return result;
    }

    private static void WriteUInt16Tiff(string path, int[] labels, int width, int height)
    {
        using var tiff = Tiff.Open(path, "w");
        if (tiff == null)
            throw new SidecarException($"Failed to write TIFF: {path}");

        tiff.SetField(TiffTag.IMAGEWIDTH, width);
        tiff.SetField(TiffTag.IMAGELENGTH, height);
        tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
        tiff.SetField(TiffTag.BITSPERSAMPLE, 16);
        tiff.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.TOPLEFT);
        tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
        tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

        var scanline = new byte[width * sizeof(ushort)];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var value = (ushort)Math.Clamp(labels[y * width + x], 0, ushort.MaxValue);
                BinaryPrimitives.WriteUInt16LittleEndian(
                    scanline.AsSpan(x * sizeof(ushort)),
                    value);
            }
            tiff.WriteScanline(scanline, y);
        }
    }

    private static void WriteByteTiff(string path, byte[] data, int[] shape)
    {
        var (width, height) = Resolve2DShape(shape);
        using var tiff = Tiff.Open(path, "w");
        if (tiff == null)
            throw new SidecarException($"Failed to write TIFF: {path}");

        tiff.SetField(TiffTag.IMAGEWIDTH, width);
        tiff.SetField(TiffTag.IMAGELENGTH, height);
        tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
        tiff.SetField(TiffTag.BITSPERSAMPLE, 8);
        tiff.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.TOPLEFT);
        tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
        tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

        for (var y = 0; y < height; y++)
            tiff.WriteScanline(data.AsSpan(y * width, width).ToArray(), y);
    }

    private static void WriteFloatTiff(string path, byte[] data, int[] shape)
    {
        var (width, height) = Resolve2DShape(shape);
        using var tiff = Tiff.Open(path, "w");
        if (tiff == null)
            throw new SidecarException($"Failed to write TIFF: {path}");

        tiff.SetField(TiffTag.IMAGEWIDTH, width);
        tiff.SetField(TiffTag.IMAGELENGTH, height);
        tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
        tiff.SetField(TiffTag.BITSPERSAMPLE, 32);
        tiff.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.IEEEFP);
        tiff.SetField(TiffTag.ORIENTATION, BitMiracle.LibTiff.Classic.Orientation.TOPLEFT);
        tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
        tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);

        var scanline = new byte[width * sizeof(float)];
        for (var y = 0; y < height; y++)
        {
            Buffer.BlockCopy(data, y * width * sizeof(float), scanline, 0, scanline.Length);
            tiff.WriteScanline(scanline, y);
        }
    }

    private static (int Width, int Height) Resolve2DShape(int[] shape)
    {
        if (shape.Length == 2)
            return (shape[1], shape[0]);
        if (shape.Length == 3)
            return (shape[2], shape[1]);
        throw new SidecarException("Unsupported flow shape for export");
    }

    private static void WriteImageJRoi(BinaryWriter writer, List<(int X, int Y)> points, int label)
    {
        writer.Write((short)0x494a);
        writer.Write((byte)0x52);
        writer.Write((byte)0x4f);
        writer.Write((byte)0x49);
        writer.Write((byte)0);
        writer.Write((byte)1);
        writer.Write((short)0);
        writer.Write((short)0);
        writer.Write((short)0);
        writer.Write((short)0);
        writer.Write((short)points.Count);
        var nameBytes = Encoding.ASCII.GetBytes($"{label:0000}");
        writer.Write(nameBytes);
        writer.Write(new byte[16 - nameBytes.Length]);
        foreach (var (x, y) in points)
        {
            writer.Write((short)x);
            writer.Write((short)y);
        }
    }
}
