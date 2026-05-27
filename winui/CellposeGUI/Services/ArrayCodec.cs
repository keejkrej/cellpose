using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json.Serialization;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

public static class ArrayCodec
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

    public static MaskData? DecodeMasks(ArrayPayload? payload)
    {
        if (payload == null || DecodeInt32Labels(payload) is not { } decoded)
            return null;

        return new MaskData
        {
            Width = decoded.Width,
            Height = decoded.Height,
            Labels = decoded.Labels,
            Colors = MaskEditService.DefaultColors(decoded.Labels.Max()),
            OutlineLabels = MaskEditService.ComputeOutlineLabels(
                decoded.Labels,
                decoded.Width,
                decoded.Height),
        };
    }

    public static (int[] Labels, int Width, int Height)? DecodeInt32Labels(ArrayPayload? payload)
    {
        if (payload == null)
            return null;

        var raw = Decode(payload);
        var shape = payload.Shape;
        int width, height;
        if (shape.Length == 2)
        {
            height = shape[0];
            width = shape[1];
        }
        else if (shape.Length == 3 && shape[0] == 1)
        {
            height = shape[1];
            width = shape[2];
        }
        else
        {
            throw new SidecarException("Invalid mask shape from sidecar");
        }

        if (payload.Dtype is "uint16" or "uint32")
        {
            var labels = new int[raw.Length / sizeof(ushort)];
            for (var i = 0; i < labels.Length; i++)
                labels[i] = BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(i * sizeof(ushort)));
            return (labels, width, height);
        }

        var intLabels = new int[raw.Length / sizeof(int)];
        Buffer.BlockCopy(raw, 0, intLabels, 0, raw.Length);
        return (intLabels, width, height);
    }

    public static byte[][] DecodeColors(ArrayPayload? payload)
    {
        if (payload == null)
            return [];

        var raw = Decode(payload);
        var shape = payload.Shape;
        if (shape.Length != 2 || shape[1] != 3)
            return [];

        var colors = new byte[shape[0]][];
        for (var row = 0; row < shape[0]; row++)
        {
            var offset = row * 3;
            colors[row] = [raw[offset], raw[offset + 1], raw[offset + 2]];
        }
        return colors;
    }

    public static InferResult ToInferResult(InferResponsePayload response)
    {
        return new InferResult
        {
            Masks = DecodeMasks(response.Masks),
            Flows = response.Flows,
            Ncells = response.Ncells,
            RecomputeMasks = response.RecomputeMasks,
        };
    }

    public static RecomputeResult ToRecomputeResult(RecomputeResponsePayload response)
    {
        return new RecomputeResult
        {
            Masks = DecodeMasks(response.Masks),
            Ncells = response.Ncells,
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

public sealed class InferResponsePayload
{
    public ArrayPayload? Masks { get; set; }
    public List<ArrayPayload> Flows { get; set; } = [];
    public int Ncells { get; set; }
    [JsonPropertyName("recompute_masks")]
    public bool RecomputeMasks { get; set; }
}

public sealed class RecomputeResponsePayload
{
    public ArrayPayload? Masks { get; set; }
    public int Ncells { get; set; }
}
