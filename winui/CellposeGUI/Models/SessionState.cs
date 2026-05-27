using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class SessionState
{
    public string? ImagePath { get; set; }
    public ImageData? Image { get; set; }
    public MaskData? Masks { get; set; }
    public List<ArrayPayload> Flows { get; set; } = [];
    public bool RecomputeMasks { get; set; }
    public string Model { get; set; } = "cpsam";
    public SegmentationParameters Segmentation { get; set; } = new();
    public bool[] IsManual { get; set; } = [];
    public SeriesDatasetPayload? SeriesDataset { get; set; }

    public int Ncells => Masks?.Labels.Length > 0 ? Masks.Labels.Max() : 0;

    public bool HasImage => Image != null && ImagePath != null;
    public bool HasMasks => Masks != null && Ncells > 0;
}
