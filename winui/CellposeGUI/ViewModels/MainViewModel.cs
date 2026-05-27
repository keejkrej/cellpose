using CellposeGUI.Models;
using CellposeGUI.Services;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Windows.Storage;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace CellposeGUI.ViewModels;

public sealed class MainViewModel : ObservableObject
{
    private readonly ISegmentationEngine _engine;
    private readonly DispatcherQueue _dispatcher;
    private bool _seriesNavigationLocked;
    private SeriesDatasetPayload? _seriesDataset;
    private readonly List<double[]> _currentStroke = [];
    private readonly List<double[][]> _pendingStrokes = [];

    private string? _sessionId;
    private string? _filename;
    private ImageData? _image;
    private MaskData? _masks;
    private int _ncells;
    private int _selectedCell;
    private bool _showMasks = true;
    private bool _showOutlines = true;
    private bool _autosave = true;
    private bool _autoloadMasks;
    private bool _disableAutosave;
    private bool _saveRestoredImage = true;
    private bool _imageLoaded;
    private bool _hasRestoredView;
    private bool _isBusy;
    private double _progress;
    private string _statusMessage = "Ready";
    private string? _errorMessage;
    private bool _recomputeMasks;
    private ViewMode _viewMode = ViewMode.Image;
    private int _selectedModelIndex;
    private string _classFilterText = "";
    private int _defaultClassId;
    private List<string> _models = ["CPSAM"];

    public MainViewModel(ISegmentationEngine engine, DispatcherQueue dispatcher)
    {
        _engine = engine;
        _dispatcher = dispatcher;
        BindDispatcher(dispatcher);
        SegmentationParams = new SegmentationParameters();
        PreprocessingParams = new PreprocessingParameters();
        DisplayParams = new DisplayParameters();
        DisplayParams.BindDispatcher(dispatcher);
        DisplayParams.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(DisplayParameters.GrayLow) or nameof(DisplayParameters.GrayHigh))
                NotifyCanvasChanged();
        };
        TrainingParams = TrainingParameters.CreateDefault(
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cellpose", "models", "custom"));
        SeriesState = new SeriesState();
        SeriesState.BindDispatcher(dispatcher);
        InstanceClasses = new InstanceClasses();
    }

    public string? SessionId
    {
        get => _sessionId;
        private set => SetProperty(ref _sessionId, value);
    }

    public string? Filename
    {
        get => _filename;
        private set
        {
            SetProperty(ref _filename, value);
            Notify(nameof(WindowTitle));
        }
    }

    public ImageData? Image
    {
        get => _image;
        private set => SetProperty(ref _image, value);
    }

    public MaskData? Masks
    {
        get => _masks;
        private set => SetProperty(ref _masks, value);
    }

    public int Ncells
    {
        get => _ncells;
        private set
        {
            SetProperty(ref _ncells, value);
            Notify(nameof(CanSaveMasks));
            Notify(nameof(FilteredCellCount));
        }
    }

    public int SelectedCell
    {
        get => _selectedCell;
        set => SetProperty(ref _selectedCell, value);
    }

    public bool ShowMasks
    {
        get => _showMasks;
        set => SetProperty(ref _showMasks, value);
    }

    public bool ShowOutlines
    {
        get => _showOutlines;
        set => SetProperty(ref _showOutlines, value);
    }

    public bool Autosave
    {
        get => _autosave;
        set => SetProperty(ref _autosave, value);
    }

    public bool AutoloadMasks
    {
        get => _autoloadMasks;
        set => SetProperty(ref _autoloadMasks, value);
    }

    public bool DisableAutosave
    {
        get => _disableAutosave;
        set => SetProperty(ref _disableAutosave, value);
    }

    public bool SaveRestoredImage
    {
        get => _saveRestoredImage;
        set => SetProperty(ref _saveRestoredImage, value);
    }

    public bool ImageLoaded
    {
        get => _imageLoaded;
        private set
        {
            SetProperty(ref _imageLoaded, value);
            Notify(nameof(CanRunSegmentation));
            Notify(nameof(CanSaveMasks));
        }
    }

    public bool HasRestoredView
    {
        get => _hasRestoredView;
        set => SetProperty(ref _hasRestoredView, value);
    }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            SetProperty(ref _isBusy, value);
            Notify(nameof(CanRunSegmentation));
        }
    }

    public double Progress
    {
        get => _progress;
        private set => SetProperty(ref _progress, value);
    }

    public string StatusMessage
    {
        get => _statusMessage;
        private set => SetProperty(ref _statusMessage, value);
    }

    public string? ErrorMessage
    {
        get => _errorMessage;
        set => SetProperty(ref _errorMessage, value);
    }

    public bool RecomputeMasks
    {
        get => _recomputeMasks;
        private set => SetProperty(ref _recomputeMasks, value);
    }

    public SegmentationParameters SegmentationParams { get; }
    public PreprocessingParameters PreprocessingParams { get; }
    public DisplayParameters DisplayParams { get; }
    public TrainingParameters TrainingParams { get; }
    public SeriesState SeriesState { get; }
    public InstanceClasses InstanceClasses { get; }

    public string ClassFilterText
    {
        get => _classFilterText;
        set
        {
            SetProperty(ref _classFilterText, value);
            Notify(nameof(FilteredCellCount));
        }
    }

    public int DefaultClassId
    {
        get => _defaultClassId;
        set => SetProperty(ref _defaultClassId, value);
    }

    public ViewMode ViewMode
    {
        get => _viewMode;
        set => SetProperty(ref _viewMode, value);
    }

    public List<string> Models
    {
        get => _models;
        private set
        {
            _models = value;
            Notify();
            Notify(nameof(IsCustomModel));
        }
    }

    public int SelectedModelIndex
    {
        get => _selectedModelIndex;
        set
        {
            SetProperty(ref _selectedModelIndex, value);
            Notify(nameof(SelectedModel));
            Notify(nameof(IsCustomModel));
        }
    }

    public string SelectedModel
    {
        get
        {
            var name = _models.Count > _selectedModelIndex ? _models[_selectedModelIndex] : "CPSAM";
            return name.Equals("cpsam", StringComparison.OrdinalIgnoreCase) ? "cpsam" : name;
        }
    }

    public bool IsCustomModel =>
        _selectedModelIndex > 0 && _models.FirstOrDefault()?.Equals("CPSAM", StringComparison.OrdinalIgnoreCase) == true;

    public bool CanSaveMasks => _imageLoaded && _ncells > 0;
    public bool CanRunSegmentation => _imageLoaded && !_isBusy;

    public string WindowTitle =>
        _filename != null ? Path.GetFileName(_filename) : "Cellpose";

    public string? PendingFolderPath { get; set; }
    public string LastSeriesSubfolderTemplate { get; set; } = "";
    public string LastSeriesFilenameTemplate { get; set; } = "img_{t}_{c}_{z}.jpg";

    public int FilteredCellCount
    {
        get
        {
            var filter = InstanceClasses.ParseFilter(_classFilterText);
            return filter == null
                ? _ncells
                : InstanceClasses.Values.Count(v => v == filter.Value);
        }
    }

    public void ApplyResult(SegmentationResult result)
    {
        if (!_dispatcher.HasThreadAccess)
        {
            RunOnUi(() => ApplyResult(result));
            return;
        }

        SessionId = result.SessionID;
        Filename = result.Filename;
        Image = result.Image;
        Masks = result.Masks;
        Ncells = result.Ncells;
        RecomputeMasks = result.RecomputeMasks;
        InstanceClasses.Replace(result.Ncells);
        Progress = 1;
        ImageLoaded = true;
        UpdateSaturationFromImage();
        Notify(nameof(CanvasRevision));
    }

    private void SetStatus(string message)
    {
        if (_dispatcher.HasThreadAccess)
            StatusMessage = message;
        else
            RunOnUi(() => StatusMessage = message);
    }

    public int CanvasRevision { get; private set; }

    public void NotifyCanvasChanged() => Notify(nameof(CanvasRevision));

    public async Task RefreshModelsAsync()
    {
        try
        {
            var listed = await _engine.ListModelsAsync();
            var custom = listed.Custom.Where(m => !m.Equals("cpsam", StringComparison.OrdinalIgnoreCase)).ToList();
            Models = ["CPSAM", .. custom];
        }
        catch
        {
            Models = ["CPSAM"];
        }
    }

    public async Task LoadImagePanelAsync()
    {
        var path = await PickFileAsync([
            ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".npy",
        ]);
        if (path != null)
            await LoadImageAsync(path);
    }

    public async Task LoadImageAsync(string path) =>
        await RunTaskAsync("Loading image…", async () =>
        {
            var result = await _engine.LoadImageAsync(path, load3D: false);
            ApplyResult(result);
            StatusMessage = $"Loaded {Path.GetFileName(path)}";
        });

    public async Task LoadSegPanelAsync()
    {
        var path = await PickFileAsync([".npy"]);
        if (path != null)
            await LoadSegAsync(path);
    }

    public async Task LoadSegAsync(string path) =>
        await RunTaskAsync("Loading segmentation…", async () =>
        {
            var result = await _engine.LoadSegAsync(path, load3D: false);
            ApplyResult(result);
            StatusMessage = $"Loaded {Path.GetFileName(path)}";
        });

    public async Task LoadSeriesFolderAsync(string subfolderTemplate, string filenameTemplate)
    {
        if (PendingFolderPath == null)
            return;

        LastSeriesSubfolderTemplate = subfolderTemplate;
        LastSeriesFilenameTemplate = filenameTemplate;

        await RunTaskAsync("Discovering series…", async () =>
        {
            var discovery = await _engine.DiscoverSeriesAsync(
                PendingFolderPath,
                subfolderTemplate,
                filenameTemplate).ConfigureAwait(false);

            await ApplyOnUiAsync(() =>
            {
                _seriesNavigationLocked = true;
                try
                {
                    ApplySeriesDiscovery(discovery, subfolderTemplate, filenameTemplate);
                }
                finally
                {
                    _seriesNavigationLocked = false;
                }
            }).ConfigureAwait(false);

            if (discovery.Records.Count == 0)
            {
                SetStatus($"Loaded series with {discovery.RecordCount} records");
                return;
            }

            var recordIndex = ResolveCurrentSeriesRecordIndex() ?? 0;
            recordIndex = Math.Clamp(recordIndex, 0, discovery.Records.Count - 1);
            var record = discovery.Records[recordIndex];
            var result = await _engine.LoadImageAsync(record.Path, load3D: false).ConfigureAwait(false);

            var loadedIndex = recordIndex;
            await ApplyOnUiAsync(() =>
            {
                SeriesState.RecordIndex = loadedIndex;
                ApplyResult(result);
                SetStatus($"Loaded series with {discovery.RecordCount} records");
            }).ConfigureAwait(false);
        });
    }

    private void ApplySeriesDiscovery(
        SeriesDiscovery discovery,
        string subfolderTemplate,
        string filenameTemplate)
    {
        _seriesDataset = discovery.Dataset;
        SeriesState.BeginBatchUpdate();
        try
        {
            SeriesState.Folder = discovery.Folder;
            SeriesState.SubfolderTemplate = subfolderTemplate;
            SeriesState.FilenameTemplate = filenameTemplate;
            SeriesState.Records = discovery.Records
                .Select(r => new SeriesRecord { Index = r.Index, Label = r.Label, Path = r.Path })
                .ToList();
            SeriesState.AxisValues = discovery.Axes;
            SeriesState.AxisSliderIndices = SeriesState.AxisOrder.ToDictionary(axis => axis, _ => 0);
        }
        finally
        {
            SeriesState.EndBatchUpdate();
        }
    }

    public async Task PresentLoadFolderPanelAsync()
    {
        var folder = await PickFolderAsync();
        if (folder == null)
            return;
        PendingFolderPath = folder;
    }

    public async Task<(string Subfolder, string Filename)> FetchSeriesTemplateSuggestionsAsync(string folder)
    {
        try
        {
            var suggestion = await _engine.SuggestSeriesTemplatesAsync(folder);
            return (suggestion.SubfolderTemplate, suggestion.FilenameTemplate);
        }
        catch
        {
            return (LastSeriesSubfolderTemplate, LastSeriesFilenameTemplate);
        }
    }

    public async Task NavigateSeriesAxisAsync(string axis, int delta)
    {
        if (!SeriesState.IsLoaded ||
            !SeriesState.AxisValues.TryGetValue(axis, out var values) ||
            values.Count == 0)
            return;

        var current = SeriesState.GetAxisIndex(axis);
        var next = Math.Clamp(current + delta, 0, values.Count - 1);
        if (next == current)
            return;

        SeriesState.SetAxisIndex(axis, next);
        await LoadCurrentSeriesRecordAsync();
    }

    public async Task SetSeriesAxisIndexAsync(string axis, int index)
    {
        if (_seriesNavigationLocked ||
            !SeriesState.IsLoaded ||
            !SeriesState.AxisValues.TryGetValue(axis, out var values) ||
            index < 0 ||
            index >= values.Count)
            return;

        if (SeriesState.GetAxisIndex(axis) == index)
            return;

        SeriesState.SetAxisIndex(axis, index);
        await LoadCurrentSeriesRecordAsync();
    }

    public async Task NavigateSeriesAsync(int delta)
    {
        if (!SeriesState.IsLoaded)
            return;

        var newIndex = Math.Clamp(SeriesState.RecordIndex + delta, 0, SeriesState.Records.Count - 1);
        if (newIndex == SeriesState.RecordIndex)
            return;

        SeriesState.RecordIndex = newIndex;
        SyncSeriesSlidersToRecordIndex();
        var record = SeriesState.CurrentRecord;
        if (record == null)
            return;

        await RunTaskAsync("Loading frame…", async () =>
        {
            var result = await _engine.LoadImageAsync(record.Path, load3D: false);
            ApplyResult(result);
            StatusMessage = record.Label;
        });
    }

    public async Task RunSegmentationAsync()
    {
        if (_sessionId == null)
            return;

        await RunTaskAsync("Running segmentation…", async () =>
        {
            Progress = 0.1;
            var result = await _engine.SegmentAsync(
                _sessionId,
                imagePayload: null,
                filename: _filename,
                modelName: SelectedModel,
                customModel: IsCustomModel,
                SegmentationParams,
                PreprocessingParams);
            Progress = 1;
            ApplyResult(result);
            StatusMessage = $"Found {result.Ncells} cells";
            if (_autosave && !_disableAutosave)
                await _engine.SaveSegAsync(result.SessionID, path: null);
        }, showProgress: true);
    }

    public async Task RecomputeFromThresholdsAsync()
    {
        if (!_recomputeMasks || _sessionId == null)
            return;

        await RunTaskAsync("Recomputing masks…", async () =>
        {
            var result = await _engine.RecomputeMasksAsync(_sessionId, SegmentationParams);
            ApplyResult(result);
            StatusMessage = $"Recomputed {result.Ncells} cells";
        });
    }

    public async Task ApplyPreprocessingAsync()
    {
        if (_sessionId == null)
            return;

        await RunTaskAsync("Applying filter…", async () =>
        {
            var result = await _engine.PreprocessAsync(_sessionId, PreprocessingParams);
            ApplyResult(result);
            HasRestoredView = true;
            StatusMessage = "Preprocessing applied";
        });
    }

    public void ClearRestore()
    {
        HasRestoredView = false;
        if (_viewMode == ViewMode.Restored)
            ViewMode = ViewMode.Image;
    }

    public Task ComputeSaturationAsync()
    {
        if (_image == null)
            return Task.CompletedTask;

        UpdateSaturationFromImage();
        StatusMessage = $"Saturation {DisplayParams.GrayLow:0}-{DisplayParams.GrayHigh:0}";
        return Task.CompletedTask;
    }

    private void UpdateSaturationFromImage()
    {
        if (_image == null)
            return;

        var (low, high) = _image.ComputeSaturationLevels(
            SegmentationParams.PercentileLow,
            SegmentationParams.PercentileHigh);
        DisplayParams.GrayLow = low;
        DisplayParams.GrayHigh = high;
    }

    public async Task SaveSegAsync()
    {
        if (_sessionId == null)
            return;

        await RunTaskAsync("Saving…", async () =>
        {
            var path = await _engine.SaveSegAsync(_sessionId, path: null);
            StatusMessage = $"Saved {Path.GetFileName(path)}";
        });
    }

    public async Task ExportMasksAsync()
    {
        await ExportWithPanelAsync("_cp_masks.png", [".png", ".tif", ".tiff"], async (sessionId, path) =>
        {
            var format = path.Contains("tif", StringComparison.OrdinalIgnoreCase) ? "tif" : "png";
            return await _engine.ExportMasksAsync(sessionId, path, format);
        });
    }

    public async Task LoadMasksPanelAsync()
    {
        var path = await PickFileAsync([".tif", ".tiff", ".png"]);
        if (path != null)
            StatusMessage = $"Load masks not yet wired for {Path.GetFileName(path)}";
    }

    public async Task ExportOutlinesAsync() =>
        await ExportWithPanelAsync("_outline.txt", [".txt"], (sessionId, path) =>
            _engine.ExportOutlinesAsync(sessionId, path));

    public async Task ExportFlowsAsync() =>
        await ExportWithPanelAsync("_flows.tif", [".tif", ".tiff"], (sessionId, path) =>
            _engine.ExportFlowsAsync(sessionId, path));

    public async Task ExportROIsAsync() =>
        await ExportWithPanelAsync("_rois.zip", [".zip"], (sessionId, path) =>
            _engine.ExportROIsAsync(sessionId, path));

    public async Task ClearAllMasksAsync()
    {
        if (_ncells <= 0)
            return;
        await RemoveCellsAsync(Enumerable.Range(1, _ncells).ToList());
    }

    public void UndoAction() { }
    public void UndoRemoveAction() { }

    public async Task RemoveSelectedModelAsync()
    {
        if (!IsCustomModel)
            return;

        var name = SelectedModel;
        await RunTaskAsync("Removing model…", async () =>
        {
            await _engine.RemoveModelAsync(name);
            await RefreshModelsAsync();
            SelectedModelIndex = 0;
            StatusMessage = $"Removed model {name}";
        });
    }

    public void RemoveCellAt(int x, int y, bool controlDown, bool altDown)
    {
        if (_masks == null)
            return;

        var label = _masks.LabelAt(x, y);
        if (label <= 0)
            return;

        if (altDown)
        {
            if (_selectedCell <= 0)
                return;
            _ = MergeCellsAsync(label, _selectedCell);
            return;
        }

        if (controlDown)
        {
            _ = RemoveCellsAsync([label]);
            return;
        }

        SelectedCell = label;
    }

    public async Task RemoveCellsAsync(IReadOnlyList<int> indices)
    {
        if (_sessionId == null)
            return;

        await RunTaskAsync("Removing cells…", async () =>
        {
            var result = await _engine.RemoveCellsAsync(_sessionId, indices);
            ApplyResult(result);
            SelectedCell = 0;
            if (_autosave && !_disableAutosave)
                await _engine.SaveSegAsync(result.SessionID, path: null);
        });
    }

    public async Task MergeCellsAsync(int source, int target)
    {
        if (_sessionId == null)
            return;

        await RunTaskAsync("Merging cells…", async () =>
        {
            var result = await _engine.MergeCellsAsync(_sessionId, source, target);
            ApplyResult(result);
        });
    }

    public void BeginStroke(int x, int y, int z = 0)
    {
        _currentStroke.Clear();
        _currentStroke.Add([z, y, x, 0]);
    }

    public void ContinueStroke(int x, int y, int z = 0) =>
        _currentStroke.Add([z, y, x, 0]);

    public void CommitStroke()
    {
        if (_currentStroke.Count == 0)
            return;
        _pendingStrokes.Add(_currentStroke.ToArray());
        _currentStroke.Clear();
    }

    public async Task FinishDrawingAsync()
    {
        if (_sessionId == null || _pendingStrokes.Count == 0)
            return;

        var strokes = _pendingStrokes.ToArray();
        _pendingStrokes.Clear();

        await RunTaskAsync("Adding cell…", async () =>
        {
            var result = await _engine.AddMaskAsync(_sessionId, strokes, _defaultClassId);
            ApplyResult(result);
            if (_autosave && !_disableAutosave)
                await _engine.SaveSegAsync(result.SessionID, path: null);
        });
    }

    public void CycleViewMode(bool forward)
    {
        var modes = ViewModeExtensions.All;
        var index = Array.IndexOf(modes, _viewMode);
        if (index < 0)
            return;
        ViewMode = modes[forward ? (index + 1) % modes.Length : (index + modes.Length - 1) % modes.Length];
    }

    public async Task TrainModelAsync()
    {
        await RunTaskAsync("Training model…", async () =>
        {
            Progress = 0.2;
            var result = await _engine.TrainAsync(TrainingParams);
            Progress = 1;
            StatusMessage = $"Trained model {result.ModelName}";
            await RefreshModelsAsync();
        }, showProgress: true);
    }

    public async Task AddCustomModelAsync()
    {
        var path = await PickFileAsync([".pth", ".pt", ".torch"]);
        if (path == null)
            return;

        await RunTaskAsync("Adding model…", async () =>
        {
            var name = await _engine.AddModelAsync(path);
            await RefreshModelsAsync();
            var index = _models.IndexOf(name);
            if (index >= 0)
                SelectedModelIndex = index;
            StatusMessage = $"Added model {name}";
        });
    }

    public async Task HandleDroppedPathsAsync(IReadOnlyList<string> paths)
    {
        var path = paths.FirstOrDefault();
        if (path == null)
            return;

        if (Path.GetExtension(path).Equals(".npy", StringComparison.OrdinalIgnoreCase) &&
            Path.GetFileName(path).Contains("_seg", StringComparison.OrdinalIgnoreCase))
            await LoadSegAsync(path);
        else
            await LoadImageAsync(path);
    }

    private async Task LoadCurrentSeriesRecordAsync()
    {
        if (_seriesNavigationLocked)
            return;

        if (ResolveCurrentSeriesRecordIndex() is not int recordIndex ||
            recordIndex == SeriesState.RecordIndex)
            return;

        SeriesState.RecordIndex = recordIndex;
        var record = SeriesState.CurrentRecord;
        if (record == null)
            return;

        await RunTaskAsync("Loading frame…", async () =>
        {
            var result = await _engine.LoadImageAsync(record.Path, load3D: false).ConfigureAwait(false);
            ApplyResult(result);
            SetStatus(record.Label);
        });
    }

    private int? ResolveCurrentSeriesRecordIndex()
    {
        if (_seriesDataset?.Lookup == null)
            return null;

        var position = AxisValue("position", _seriesDataset);
        var time = AxisValue("time", _seriesDataset);
        var channel = AxisValue("channel", _seriesDataset);
        var z = AxisValue("z", _seriesDataset);
        var key = $"{position}_{time}_{channel}_{z}";
        return _seriesDataset.Lookup.TryGetValue(key, out var index) ? index : null;
    }

    private string AxisValue(string axis, SeriesDatasetPayload dataset)
    {
        if (!dataset.Axes.TryGetValue(axis, out var values))
            return "";

        var index = SeriesState.GetAxisIndex(axis);
        return index >= 0 && index < values.Count ? values[index] : "";
    }

    private void SyncSeriesSlidersToRecordIndex()
    {
        if (_seriesDataset == null ||
            SeriesState.RecordIndex < 0 ||
            SeriesState.RecordIndex >= _seriesDataset.Records.Count)
            return;

        var record = _seriesDataset.Records[SeriesState.RecordIndex];
        foreach (var axis in SeriesState.AxisOrder)
        {
            var value = RecordValue(record, axis);
            if (_seriesDataset.AxisIndex?.TryGetValue(axis, out var axisMap) == true &&
                axisMap.TryGetValue(value, out var mappedIndex))
            {
                SeriesState.SetAxisIndex(axis, mappedIndex);
            }
            else if (_seriesDataset.Axes.TryGetValue(axis, out var values))
            {
                var valueIndex = values.IndexOf(value);
                if (valueIndex >= 0)
                    SeriesState.SetAxisIndex(axis, valueIndex);
            }
        }
    }

    private static string RecordValue(SeriesDatasetRecord record, string axis) => axis switch
    {
        "position" => record.Position,
        "time" => record.Time,
        "channel" => record.Channel,
        _ => record.Z,
    };

    private async Task ExportWithPanelAsync(
        string defaultSuffix,
        IReadOnlyList<string> extensions,
        Func<string, string, Task<string>> export)
    {
        if (_sessionId == null || _filename == null)
            return;

        var defaultName = Path.GetFileNameWithoutExtension(_filename) + defaultSuffix;
        var path = await PickSaveFileAsync(defaultName, extensions);
        if (path == null)
            return;

        await RunTaskAsync("Exporting…", async () =>
        {
            var saved = await export(_sessionId, path);
            StatusMessage = $"Exported {Path.GetFileName(saved)}";
        });
    }

    private async Task RunTaskAsync(
        string message,
        Func<Task> operation,
        bool showProgress = false)
    {
        RunOnUi(() =>
        {
            IsBusy = true;
            StatusMessage = message;
            ErrorMessage = null;
            if (showProgress)
                Progress = 0;
        });

        try
        {
            await operation().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            RunOnUi(() =>
            {
                ErrorMessage = ex.Message;
                StatusMessage = "Error";
            });
        }
        finally
        {
            RunOnUi(() =>
            {
                IsBusy = false;
                NotifyCanvasChanged();
            });
        }
    }

    private Task ApplyOnUiAsync(Action action)
    {
        if (_dispatcher.HasThreadAccess)
        {
            action();
            return Task.CompletedTask;
        }

        var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _dispatcher.TryEnqueue(() =>
        {
            try
            {
                action();
                tcs.SetResult();
            }
            catch (Exception ex)
            {
                tcs.SetException(ex);
            }
        });
        return tcs.Task;
    }

    private void RunOnUi(Action action)
    {
        if (_dispatcher.HasThreadAccess)
            action();
        else
            _dispatcher.TryEnqueue(() => action());
    }

    private Task<T> RunOnUiAsync<T>(Func<Task<T>> action)
    {
        if (_dispatcher.HasThreadAccess)
            return action();

        var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        _dispatcher.TryEnqueue(async () =>
        {
            try
            {
                tcs.SetResult(await action());
            }
            catch (Exception ex)
            {
                tcs.SetException(ex);
            }
        });
        return tcs.Task;
    }

    private static nint GetWindowHandle()
    {
        var window = App.CurrentWindow;
        return WindowNative.GetWindowHandle(window);
    }

    private Task<string?> PickFileAsync(IReadOnlyList<string> extensions) =>
        RunOnUiAsync(() => PickFileAsyncCore(extensions));

    private Task<string?> PickFolderAsync() =>
        RunOnUiAsync(() => PickFolderAsyncCore());

    private Task<string?> PickSaveFileAsync(string suggestedName, IReadOnlyList<string> extensions) =>
        RunOnUiAsync(() => PickSaveFileAsyncCore(suggestedName, extensions));

    private static async Task<string?> PickFileAsyncCore(IReadOnlyList<string> extensions)
    {
        var picker = new FileOpenPicker
        {
            ViewMode = PickerViewMode.List,
            SuggestedStartLocation = PickerLocationId.ComputerFolder,
        };
        InitializePicker(picker);
        foreach (var extension in extensions)
            picker.FileTypeFilter.Add(extension.TrimStart('.'));

        var file = await picker.PickSingleFileAsync();
        return file?.Path;
    }

    private static async Task<string?> PickFolderAsyncCore()
    {
        var picker = new FolderPicker();
        InitializePicker(picker);
        picker.FileTypeFilter.Add("*");
        var folder = await picker.PickSingleFolderAsync();
        return folder?.Path;
    }

    private static async Task<string?> PickSaveFileAsyncCore(string suggestedName, IReadOnlyList<string> extensions)
    {
        var picker = new FileSavePicker
        {
            SuggestedFileName = Path.GetFileNameWithoutExtension(suggestedName),
            SuggestedStartLocation = PickerLocationId.ComputerFolder,
        };
        InitializePicker(picker);
        foreach (var extension in extensions)
            picker.FileTypeChoices.Add(extension.TrimStart('.').ToUpperInvariant(), [extension]);

        var file = await picker.PickSaveFileAsync();
        return file?.Path;
    }

    private static void InitializePicker(object picker)
    {
        var hwnd = GetWindowHandle();
        if (picker is FileOpenPicker openPicker)
            InitializeWithWindow.Initialize(openPicker, hwnd);
        else if (picker is FolderPicker folderPicker)
            InitializeWithWindow.Initialize(folderPicker, hwnd);
        else if (picker is FileSavePicker savePicker)
            InitializeWithWindow.Initialize(savePicker, hwnd);
    }
}
