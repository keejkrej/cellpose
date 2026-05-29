using System.Text.Json.Serialization;

namespace CellposeGUI.Models;

public sealed class ArrayPayload
{
    public string Dtype { get; set; } = "";
    public int[] Shape { get; set; } = [];

    [JsonPropertyName("data_b64")]
    public string DataB64 { get; set; } = "";
}

public sealed class SidecarException : Exception
{
    public SidecarException(string message) : base(message) { }
}
