using System.IO.Compression;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

internal static class SegNpyArrayCodec
{
    public static byte[] Decode(ArrayPayload payload)
    {
        var compressed = Convert.FromBase64String(payload.DataB64);
        return DecompressZlib(compressed);
    }

    public static ArrayPayload EncodeRaw(byte[] raw, string dtype, int[] shape)
    {
        return new ArrayPayload
        {
            Dtype = dtype,
            Shape = shape,
            DataB64 = Convert.ToBase64String(CompressZlib(raw)),
        };
    }

    private static byte[] CompressZlib(byte[] data)
    {
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(output, CompressionLevel.Optimal))
            zlib.Write(data);
        return output.ToArray();
    }

    private static byte[] DecompressZlib(byte[] data)
    {
        using var input = new MemoryStream(data);
        using var zlib = new ZLibStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();
        zlib.CopyTo(output);
        return output.ToArray();
    }
}
