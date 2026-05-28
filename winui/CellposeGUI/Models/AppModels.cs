using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using Microsoft.UI.Dispatching;

namespace CellposeGUI.Models;

public abstract class ObservableObject : INotifyPropertyChanged
{
    private DispatcherQueue? _dispatcher;

    public event PropertyChangedEventHandler? PropertyChanged;

    public void BindDispatcher(DispatcherQueue dispatcher) => _dispatcher = dispatcher;

    protected void SetProperty<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return;
        field = value;
        if (CanNotify(name))
            RaisePropertyChanged(name);
    }

    protected void Notify([CallerMemberName] string? name = null)
    {
        if (CanNotify(name))
            RaisePropertyChanged(name);
    }

    protected virtual bool CanNotify(string? name) => true;

    private void RaisePropertyChanged(string? name)
    {
        if (_dispatcher == null || _dispatcher.HasThreadAccess)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
            return;
        }

        _dispatcher.TryEnqueue(() =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name)));
    }
}

public sealed class SegmentationParameters : ObservableObject
{
    private double _diameter;
    private double _flowThreshold = 0.4;
    private double _cellprobThreshold;
    private double _percentileLow = 1;
    private double _percentileHigh = 99;
    private int _niter;
    private int _minSize = 15;

    public double Diameter
    {
        get => _diameter;
        set => SetProperty(ref _diameter, value);
    }

    public double FlowThreshold
    {
        get => _flowThreshold;
        set => SetProperty(ref _flowThreshold, value);
    }

    public double CellprobThreshold
    {
        get => _cellprobThreshold;
        set => SetProperty(ref _cellprobThreshold, value);
    }

    public double PercentileLow
    {
        get => _percentileLow;
        set => SetProperty(ref _percentileLow, value);
    }

    public double PercentileHigh
    {
        get => _percentileHigh;
        set => SetProperty(ref _percentileHigh, value);
    }

    public int Niter
    {
        get => _niter;
        set => SetProperty(ref _niter, value);
    }

    public int MinSize
    {
        get => _minSize;
        set => SetProperty(ref _minSize, value);
    }

    public void CopyFrom(SegmentationParameters other)
    {
        Diameter = other.Diameter;
        FlowThreshold = other.FlowThreshold;
        CellprobThreshold = other.CellprobThreshold;
        PercentileLow = other.PercentileLow;
        PercentileHigh = other.PercentileHigh;
        Niter = other.Niter;
        MinSize = other.MinSize;
    }

    public SegmentationParameters Clone() => new()
    {
        Diameter = Diameter,
        FlowThreshold = FlowThreshold,
        CellprobThreshold = CellprobThreshold,
        PercentileLow = PercentileLow,
        PercentileHigh = PercentileHigh,
        Niter = Niter,
        MinSize = MinSize,
    };
}

public sealed class DisplayParameters : ObservableObject
{
    private double _grayLow;
    private double _grayHigh = 255;

    public double GrayLow
    {
        get => _grayLow;
        set => SetProperty(ref _grayLow, value);
    }

    public double GrayHigh
    {
        get => _grayHigh;
        set => SetProperty(ref _grayHigh, value);
    }
}

public sealed class TrainingParameters
{
    [JsonPropertyName("learning_rate")]
    public double LearningRate { get; set; } = 1e-5;
    [JsonPropertyName("weight_decay")]
    public double WeightDecay { get; set; } = 0.1;
    [JsonPropertyName("n_epochs")]
    public int NEpochs { get; set; } = 100;
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = "";
    [JsonPropertyName("train_data_folder")]
    public string TrainDataFolder { get; set; } = "";
    [JsonPropertyName("model_save_folder")]
    public string ModelSaveFolder { get; set; } = "";

    public static TrainingParameters CreateDefault(string modelSaveFolder)
    {
        var suffix = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        return new TrainingParameters
        {
            ModelName = $"cpsam_{suffix}",
            ModelSaveFolder = modelSaveFolder,
        };
    }
}

public sealed class SeriesRecord
{
    public int Index { get; init; }
    public string Label { get; init; } = "";
    public string Path { get; init; } = "";
}

public sealed class SeriesState : ObservableObject
{
    public static readonly string[] AxisOrder = ["position", "time", "channel", "z"];
    public static readonly Dictionary<string, string> AxisLabels = new()
    {
        ["position"] = "P",
        ["time"] = "T",
        ["channel"] = "C",
        ["z"] = "Z",
    };

    private string _folder = "";
    private string _subfolderTemplate = "";
    private string _filenameTemplate = "";
    private int _recordIndex;
    private Dictionary<string, List<string>> _axisValues = new();
    private Dictionary<string, int> _axisSliderIndices = new()
    {
        ["position"] = 0,
        ["time"] = 0,
        ["channel"] = 0,
        ["z"] = 0,
    };
    private List<SeriesRecord> _records = [];
    private int _batchDepth;

    public void BeginBatchUpdate() => _batchDepth++;

    public void EndBatchUpdate()
    {
        if (_batchDepth == 0)
            return;
        _batchDepth--;
        if (_batchDepth == 0)
            Notify(string.Empty);
    }

    protected override bool CanNotify(string? name) => _batchDepth == 0;

    public string Folder
    {
        get => _folder;
        set => SetProperty(ref _folder, value);
    }

    public string SubfolderTemplate
    {
        get => _subfolderTemplate;
        set => SetProperty(ref _subfolderTemplate, value);
    }

    public string FilenameTemplate
    {
        get => _filenameTemplate;
        set => SetProperty(ref _filenameTemplate, value);
    }

    public List<SeriesRecord> Records
    {
        get => _records;
        set
        {
            _records = value;
            Notify();
            Notify(nameof(IsLoaded));
            Notify(nameof(CurrentRecord));
        }
    }

    public int RecordIndex
    {
        get => _recordIndex;
        set
        {
            SetProperty(ref _recordIndex, value);
            Notify(nameof(CurrentRecord));
        }
    }

    public Dictionary<string, List<string>> AxisValues
    {
        get => _axisValues;
        set
        {
            _axisValues = value;
            Notify();
        }
    }

    public Dictionary<string, int> AxisSliderIndices
    {
        get => _axisSliderIndices;
        set
        {
            _axisSliderIndices = value;
            Notify();
        }
    }

    public SeriesRecord? CurrentRecord =>
        _records.Count > _recordIndex && _recordIndex >= 0 ? _records[_recordIndex] : null;

    public bool IsLoaded => _records.Count > 0;

    public int GetAxisIndex(string axis) =>
        _axisSliderIndices.TryGetValue(axis, out var index) ? index : 0;

    public void SetAxisIndex(string axis, int index)
    {
        _axisSliderIndices[axis] = index;
        Notify(nameof(AxisSliderIndices));
    }
}

public sealed class InstanceClasses
{
    public List<int> Values { get; private set; } = [];

    public void Replace(int ncells, IReadOnlyList<int>? loaded = null)
    {
        var result = new int[ncells];
        if (loaded != null)
        {
            var n = Math.Min(ncells, loaded.Count);
            for (var i = 0; i < n; i++)
                result[i] = Math.Max(0, loaded[i]);
        }
        Values = [.. result];
    }

    public void SetClass(int row, int classId)
    {
        if (row < 0 || row >= Values.Count || classId < 0)
            return;
        Values[row] = classId;
    }

    public static int? ParseFilter(string text)
    {
        var trimmed = text.Trim();
        if (string.IsNullOrEmpty(trimmed))
            return null;
        return int.TryParse(trimmed, out var value) && value >= 0 ? value : null;
    }

    public static bool IsLabelVisible(
        int label,
        IReadOnlyList<int> classes,
        IReadOnlyList<bool> visibility,
        int? filterClassId)
    {
        if (label <= 0)
            return false;

        var row = label - 1;
        if (row < visibility.Count && !visibility[row])
            return false;

        if (filterClassId == null)
            return true;

        return row < classes.Count && classes[row] == filterClassId.Value;
    }
}

public sealed class InstanceVisibility
{
    public List<bool> Values { get; private set; } = [];

    public void Replace(int ncells, IReadOnlyList<bool>? loaded = null)
    {
        var result = new bool[ncells];
        if (loaded != null)
        {
            var n = Math.Min(ncells, loaded.Count);
            for (var i = 0; i < n; i++)
                result[i] = loaded[i];
        }
        else
        {
            var n = Math.Min(ncells, Values.Count);
            for (var i = 0; i < n; i++)
                result[i] = Values[i];
            for (var i = n; i < ncells; i++)
                result[i] = true;
        }

        Values = [.. result];
    }

    public void SetVisible(int row, bool visible)
    {
        if (row < 0 || row >= Values.Count)
            return;
        Values[row] = visible;
    }

    public void SetAll(int ncells, bool visible) =>
        Values = Enumerable.Repeat(visible, ncells).ToList();
}

public enum ViewMode
{
    Image,
    GradXY,
    Cellprob,
    Restored,
}

public static class ViewModeExtensions
{
    public static string Title(this ViewMode mode) => mode switch
    {
        ViewMode.Image => "image",
        ViewMode.GradXY => "gradXY",
        ViewMode.Cellprob => "cellprob",
        ViewMode.Restored => "restored",
        _ => mode.ToString(),
    };

    public static ViewMode[] All { get; } =
        [ViewMode.Image, ViewMode.GradXY, ViewMode.Cellprob, ViewMode.Restored];
}

public sealed class ImageData
{
    public int Width { get; init; }
    public int Height { get; init; }
    public int Channels { get; init; }
    public byte[] Pixels { get; init; } = [];

    public byte GetGrayValue(int pixelIndex)
    {
        if (Channels == 1)
            return pixelIndex < Pixels.Length ? Pixels[pixelIndex] : (byte)0;

        var offset = pixelIndex * Channels;
        return offset < Pixels.Length ? Pixels[offset] : (byte)0;
    }

    public (double Low, double High) ComputeSaturationLevels(double percentileLow, double percentileHigh)
    {
        var pixelCount = Width * Height;
        if (pixelCount <= 0)
            return (0, 255);

        var histogram = new int[256];
        for (var i = 0; i < pixelCount; i++)
            histogram[GetGrayValue(i)]++;

        var low = PercentileFromHistogram(histogram, pixelCount, percentileLow);
        var high = PercentileFromHistogram(histogram, pixelCount, percentileHigh);
        if (high - low <= 1e-3)
            return (0, 255);

        return (low, high);
    }

    private static double PercentileFromHistogram(int[] histogram, int total, double percentile)
    {
        if (total <= 0)
            return 0;

        var target = (total - 1) * percentile / 100.0;
        var cumulative = 0;
        for (var value = 0; value < histogram.Length; value++)
        {
            cumulative += histogram[value];
            if (cumulative > target)
                return value;
        }

        return histogram.Length - 1;
    }

    public static byte MapGrayLevel(byte raw, double low, double high)
    {
        if (high <= low + 1e-3)
            return raw;

        var scaled = (raw - low) / (high - low) * 255.0;
        return (byte)Math.Clamp((int)Math.Round(scaled), 0, 255);
    }
}

public sealed class MaskData
{
    public int Width { get; init; }
    public int Height { get; init; }
    public int[] Labels { get; init; } = [];
    public byte[][] Colors { get; init; } = [];
    public int[]? OutlineLabels { get; init; }

    public int LabelAt(int x, int y)
    {
        if (x < 0 || y < 0 || x >= Width || y >= Height)
            return 0;
        return Labels[y * Width + x];
    }

    public (byte R, byte G, byte B, float Alpha)? ColorAt(int x, int y)
    {
        if (x < 0 || y < 0 || x >= Width || y >= Height)
            return null;
        var label = Labels[y * Width + x];
        if (label <= 0)
            return null;
        var index = label - 1;
        if (index >= Colors.Length)
            return null;
        var color = Colors[index];
        return (color[0], color[1], color[2], 0.5f);
    }
}

public sealed class SegmentationResult
{
    public string SessionID { get; init; } = "";
    public ImageData Image { get; init; } = new();
    public MaskData? Masks { get; init; }
    public int Ncells { get; init; }
    public bool RecomputeMasks { get; init; }
    public string? Filename { get; init; }
}

public sealed class ArrayPayload
{
    public string Dtype { get; set; } = "";
    public int[] Shape { get; set; } = [];
    [JsonPropertyName("data_b64")]
    public string DataB64 { get; set; } = "";
}

public sealed class SidecarHealth
{
    public string Status { get; set; } = "";
    public string Version { get; set; } = "";
    [JsonPropertyName("model_loaded")]
    public bool ModelLoaded { get; set; }
    public string? Device { get; set; }
}

public sealed class SidecarModels
{
    public List<string> Builtin { get; set; } = [];
    public List<string> Custom { get; set; } = [];
    public IEnumerable<string> All => Builtin.Concat(Custom);
}

public sealed class SeriesTemplateSuggestion
{
    [JsonPropertyName("subfolder_template")]
    public string SubfolderTemplate { get; set; } = "";
    [JsonPropertyName("filename_template")]
    public string FilenameTemplate { get; set; } = "";
}

public sealed class SeriesDiscovery
{
    public string Folder { get; set; } = "";
    [JsonPropertyName("record_count")]
    public int RecordCount { get; set; }
    public List<SeriesDiscoveryRecord> Records { get; set; } = [];
    public Dictionary<string, List<string>> Axes { get; set; } = new();
    public SeriesDatasetPayload Dataset { get; set; } = new();
}

public sealed class SeriesDiscoveryRecord
{
    public int Index { get; set; }
    public string Label { get; set; } = "";
    public string Path { get; set; } = "";
}

public sealed class SeriesDatasetPayload
{
    public string Folder { get; set; } = "";
    [JsonPropertyName("subfolder_template")]
    public string SubfolderTemplate { get; set; } = "";
    [JsonPropertyName("filename_template")]
    public string FilenameTemplate { get; set; } = "";
    public Dictionary<string, List<string>> Axes { get; set; } = new();
    [JsonPropertyName("axis_index")]
    public Dictionary<string, Dictionary<string, int>>? AxisIndex { get; set; }
    public Dictionary<string, int>? Lookup { get; set; }
    public List<SeriesDatasetRecord> Records { get; set; } = [];
}

public sealed class SeriesDatasetRecord
{
    public string Label { get; set; } = "";
    public string Path { get; set; } = "";
    public string Position { get; set; } = "";
    public string Time { get; set; } = "";
    public string Channel { get; set; } = "";
    public string Z { get; set; } = "";
}

public sealed class TrainResult
{
    [JsonPropertyName("model_path")]
    public string ModelPath { get; set; } = "";
    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = "";
}

public sealed class SidecarSessionResponse
{
    [JsonPropertyName("session_id")]
    public string SessionID { get; set; } = "";
    public string? Filename { get; set; }
    public int[] Shape { get; set; } = [];
    public int Ncells { get; set; }
    public ArrayPayload? Masks { get; set; }
    public ArrayPayload? Outlines { get; set; }
    [JsonPropertyName("display_image")]
    public ArrayPayload DisplayImage { get; set; } = new();
    public ArrayPayload? Colors { get; set; }
    [JsonPropertyName("recompute_masks")]
    public bool RecomputeMasks { get; set; }
}

public sealed class SidecarErrorResponse
{
    public string Detail { get; set; } = "";
}

public sealed class SidecarException : Exception
{
    public SidecarException(string message) : base(message) { }
}
