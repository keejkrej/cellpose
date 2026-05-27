using System.IO.Compression;
using System.Text;
using System.Text.Json;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

public static class ArrayCodec
{
    public static byte[] Decode(ArrayPayload payload)
    {
        var compressed = Convert.FromBase64String(payload.DataB64);
        return DecompressZlib(compressed);
    }

    public static ImageData DecodeUInt8Image(ArrayPayload payload)
    {
        var raw = Decode(payload);
        var shape = payload.Shape;
        int width, height, channels;
        if (shape.Length == 3)
        {
            height = shape[0];
            width = shape[1];
            channels = shape[2];
        }
        else if (shape.Length == 2)
        {
            height = shape[0];
            width = shape[1];
            channels = 1;
        }
        else
        {
            throw new SidecarException("Invalid image shape from sidecar");
        }

        return new ImageData
        {
            Width = width,
            Height = height,
            Channels = channels,
            Pixels = raw,
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

        var labels = new int[raw.Length / sizeof(int)];
        Buffer.BlockCopy(raw, 0, labels, 0, raw.Length);
        return (labels, width, height);
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

    public static SegmentationResult ToSegmentationResult(SidecarSessionResponse response)
    {
        var image = DecodeUInt8Image(response.DisplayImage);
        MaskData? maskData = null;
        if (response.Masks != null &&
            DecodeInt32Labels(response.Masks) is { } maskDecoded)
        {
            var colors = DecodeColors(response.Colors);
            var outlines = DecodeInt32Labels(response.Outlines)?.Labels;
            maskData = new MaskData
            {
                Width = maskDecoded.Width,
                Height = maskDecoded.Height,
                Labels = maskDecoded.Labels,
                Colors = colors,
                OutlineLabels = outlines,
            };
        }

        return new SegmentationResult
        {
            SessionID = response.SessionID,
            Image = image,
            Masks = maskData,
            Ncells = response.Ncells,
            RecomputeMasks = response.RecomputeMasks,
            Filename = response.Filename,
        };
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

public sealed class SidecarClient
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    private readonly HttpClient _http;
    private readonly Uri _baseUri;

    public SidecarClient(Uri baseUri)
    {
        _baseUri = baseUri;
        _http = new HttpClient
        {
            Timeout = TimeSpan.FromHours(1),
        };
    }

    public async Task<T> GetAsync<T>(string path, CancellationToken cancellationToken = default)
    {
        var response = await _http.GetAsync(new Uri(_baseUri, path.TrimStart('/')), cancellationToken);
        return await ReadResponseAsync<T>(response, cancellationToken);
    }

    public async Task<T> PostAsync<T>(string path, object body, CancellationToken cancellationToken = default)
    {
        var json = JsonSerializer.Serialize(body, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await _http.PostAsync(new Uri(_baseUri, path.TrimStart('/')), content, cancellationToken);
        return await ReadResponseAsync<T>(response, cancellationToken);
    }

    private static async Task<T> ReadResponseAsync<T>(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        var data = await response.Content.ReadAsStringAsync(cancellationToken);
        if (response.IsSuccessStatusCode)
            return JsonSerializer.Deserialize<T>(data, JsonOptions)
                   ?? throw new SidecarException("Invalid response from sidecar");

        var detail = JsonSerializer.Deserialize<SidecarErrorResponse>(data, JsonOptions)?.Detail;
        throw new SidecarException(detail ?? data);
    }
}
