using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class ImageLoaderService
{
    public ImageData Load(string path)
    {
        var extension = Path.GetExtension(path).ToLowerInvariant();
        return extension switch
        {
            ".tif" or ".tiff" => LoadTiff(path),
            ".png" or ".jpg" or ".jpeg" or ".gif" or ".bmp" => LoadRaster(path),
            _ => throw new SidecarException($"Unsupported image format: {extension}"),
        };
    }

    private static ImageData LoadRaster(string path)
    {
        using var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(path);
        var width = image.Width;
        var height = image.Height;
        var pixels = new byte[width * height * 3];
        image.CopyPixelDataTo(pixels);
        return new ImageData
        {
            Width = width,
            Height = height,
            Channels = 3,
            Pixels = pixels,
        };
    }

    private static ImageData LoadTiff(string path)
    {
        using var tiff = BitMiracle.LibTiff.Classic.Tiff.Open(path, "r");
        if (tiff == null)
            throw new SidecarException($"Failed to open TIFF: {path}");

        var width = tiff.GetField(BitMiracle.LibTiff.Classic.TiffTag.IMAGEWIDTH)[0].ToInt();
        var height = tiff.GetField(BitMiracle.LibTiff.Classic.TiffTag.IMAGELENGTH)[0].ToInt();
        var samplesPerPixel = tiff.GetField(BitMiracle.LibTiff.Classic.TiffTag.SAMPLESPERPIXEL)?[0].ToInt() ?? 1;
        var channels = Math.Clamp(samplesPerPixel, 1, 3);
        var raster = new byte[width * height * channels];
        var scanline = new byte[tiff.ScanlineSize()];

        for (var y = 0; y < height; y++)
        {
            if (!tiff.ReadScanline(scanline, y))
                throw new SidecarException($"Failed to read TIFF scanline {y}");

            if (channels == 1)
            {
                Buffer.BlockCopy(scanline, 0, raster, y * width, width);
                continue;
            }

            for (var x = 0; x < width; x++)
            {
                var dst = (y * width + x) * channels;
                for (var c = 0; c < channels; c++)
                    raster[dst + c] = scanline[x * samplesPerPixel + c];
            }
        }

        return new ImageData
        {
            Width = width,
            Height = height,
            Channels = channels,
            Pixels = raster,
        };
    }
}
