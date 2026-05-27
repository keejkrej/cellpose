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
    private readonly IMlInferenceEngine _ml;
    private readonly ImageLoaderService _imageLoader;
    private readonly CellposeSessionStore _sessionStore;
    private readonly SeriesDiscoveryService _seriesDiscovery;
    private readonly ExportService _exportService;
    private readonly SessionState _session = new();
    private readonly DispatcherQueue _dispatcher;
    private bool _seriesNavigationLocked;
    private SeriesDatasetPayload? _seriesDataset;
    private readonly List<double[]> _currentStroke = [];
    private readonly List<double[][]> _pendingStrokes = [];

    private string? _filename;
    private ImageData? _image;
    private MaskData? _masks;
    private int _ncells;
    private int _selectedCell;
    private bool _showMasks = true;
    private bool _showOutlines = true;
    private bool _autoloadMasks;
    private bool _disableAutosave;
    private bool _imageLoaded;
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

    public MainViewModel(
        IMlInferenceEngine ml,
        ImageLoaderService imageLoader,
        CellposeSessionStore sessionStore,
        SeriesDiscoveryService seriesDiscovery,
        ExportService exportService,
        DispatcherQueue dispatcher)
    {
        _ml = ml;
        _imageLoader = imageLoader;
        _sessionStore = sessionStore;
        _seriesDiscovery = seriesDiscovery;
        _exportService = exportService;
        _dispatcher = dispatcher;
        BindDispatcher(dispatcher);
        SegmentationParams = new SegmentationParameters();
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

    public void ApplyLoadedImage(string path, ImageData image)
    {
        if (!_dispatcher.HasThreadAccess)
        {
            RunOnUi(() => ApplyLoadedImage(path, image));
            return;
        }

        _session.ImagePath = path;
        _session.Image = image;
        _session.Masks = null;
        _session.Flows = [];
        _session.RecomputeMasks = false;
        _session.SeriesDataset = _seriesDataset;

        Filename = path;
        Image = image;
        Masks = null;
        Ncells = 0;
        RecomputeMasks = false;
        SelectedCell = 0;
        InstanceClasses.Replace(0);
        Progress = 1;
        ImageLoaded = true;
        UpdateSaturationFromImage();
        Notify(nameof(CanvasRevision));
    }

    public void ApplyLoadedSession(LoadedSession loaded)
    {
        if (!_dispatcher.HasThreadAccess)
        {
            RunOnUi(() => ApplyLoadedSession(loaded));
            return;
        }

        _session.ImagePath = loaded.ImagePath;
        _session.Image = loaded.Image;
        _session.Masks = loaded.Masks;
        _session.Flows = loaded.Flows;
        _session.RecomputeMasks = loaded.RecomputeMasks;
        _session.Model = loaded.Model;
        _session.Segmentation = loaded.Segmentation;
        SegmentationParams.Diameter = loaded.Segmentation.Diameter;
        SegmentationParams.FlowThreshold = loaded.Segmentation.FlowThreshold;
        SegmentationParams.CellprobThreshold = loaded.Segmentation.CellprobThreshold;
        SegmentationParams.Niter = loaded.Segmentation.Niter;
        SegmentationParams.MinSize = loaded.Segmentation.MinSize;

        Filename = loaded.ImagePath;
        Image = loaded.Image;
        Masks = loaded.Masks;
        Ncells = loaded.Masks.Labels.Length == 0 ? 0 : loaded.Masks.Labels.Max();
        RecomputeMasks = loaded.RecomputeMasks;
        InstanceClasses.Replace(Ncells);
        SelectedCell = 0;
        Progress = 1;
        ImageLoaded = true;
        UpdateSaturationFromImage();
        Notify(nameof(CanvasRevision));
    }

    public void ApplyMaskUpdate(MaskData masks)
    {
        if (!_dispatcher.HasThreadAccess)
        {
            RunOnUi(() => ApplyMaskUpdate(masks));
            return;
        }

        _session.Masks = masks;
        Masks = masks;
        Ncells = masks.Labels.Length == 0 ? 0 : masks.Labels.Max();
        InstanceClasses.Replace(Ncells);
        NotifyCanvasChanged();
    }

    public void ApplyInferenceResult(InferResult result)
    {
        if (!_dispatcher.HasThreadAccess)
        {
            RunOnUi(() => ApplyInferenceResult(result));
            return;
        }

        _session.Flows = result.Flows;
        _session.RecomputeMasks = result.RecomputeMasks;
        _session.Model = SelectedModel;
        _session.Segmentation = CloneSegmentationParams();
        RecomputeMasks = result.RecomputeMasks;
        if (result.Masks != null)
            ApplyMaskUpdate(result.Masks);
    }

    private SegmentationParameters CloneSegmentationParams() => new()
    {
        Diameter = SegmentationParams.Diameter,
        FlowThreshold = SegmentationParams.FlowThreshold,
        CellprobThreshold = SegmentationParams.CellprobThreshold,
        PercentileLow = SegmentationParams.PercentileLow,
        PercentileHigh = SegmentationParams.PercentileHigh,
        Niter = SegmentationParams.Niter,
        MinSize = SegmentationParams.MinSize,
        StitchThreshold = SegmentationParams.StitchThreshold,
        Anisotropy = SegmentationParams.Anisotropy,
        Flow3DSmooth = SegmentationParams.Flow3DSmooth,
        Do3D = SegmentationParams.Do3D,
    };

    private void SaveSessionIfNeeded()
    {
        if (_disableAutosave || _session.ImagePath == null || _session.Masks == null)
            return;

        _session.Segmentation = CloneSegmentationParams();
        _session.Model = SelectedModel;
        var path = _sessionStore.DefaultPath(_session.ImagePath);
        _sessionStore.Save(path, _session);
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
            var listed = await _ml.ListModelsAsync();
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
            ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif",
        ]);
        if (path != null)
            await LoadImageAsync(path);
    }

    public async Task LoadImageAsync(string path) =>
        await RunTaskAsync("Loading image…", async () =>
        {
            var image = await Task.Run(() => _imageLoader.Load(path));
            ApplyLoadedImage(path, image);
            StatusMessage = $"Loaded {Path.GetFileName(path)}";
        });

    public async Task LoadSegPanelAsync()
    {
        var path = await PickFileAsync([".cellpose"]);
        if (path != null)
            await LoadSegAsync(path);
    }

    public async Task LoadSegAsync(string path) =>
        await RunTaskAsync("Loading segmentation…", async () =>
        {
            var loaded = await Task.Run(() => _sessionStore.Load(path, _imageLoader));
            ApplyLoadedSession(loaded);
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
            var discovery = await Task.Run(() =>
                _seriesDiscovery.Discover(PendingFolderPath, subfolderTemplate, filenameTemplate));

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
            var image = await Task.Run(() => _imageLoader.Load(record.Path));

            var loadedIndex = recordIndex;
            await ApplyOnUiAsync(() =>
            {
                SeriesState.RecordIndex = loadedIndex;
                ApplyLoadedImage(record.Path, image);
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

    public Task<(string Subfolder, string Filename)> FetchSeriesTemplateSuggestionsAsync(string folder)
    {
        try
        {
            var suggestion = _seriesDiscovery.SuggestTemplates(folder);
            return Task.FromResult((suggestion.SubfolderTemplate, suggestion.FilenameTemplate));
        }
        catch
        {
            return Task.FromResult((LastSeriesSubfolderTemplate, LastSeriesFilenameTemplate));
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
            var image = await Task.Run(() => _imageLoader.Load(record.Path));
            ApplyLoadedImage(record.Path, image);
            StatusMessage = record.Label;
        });
    }

    public async Task RunSegmentationAsync()
    {
        if (_session.ImagePath == null)
            return;

        await RunTaskAsync("Running segmentation…", async () =>
        {
            Progress = 0.1;
            var result = await _ml.InferAsync(
                _session.ImagePath,
                SelectedModel,
                IsCustomModel,
                SegmentationParams);
            Progress = 1;
            ApplyInferenceResult(result);
            StatusMessage = $"Found {result.Ncells} cells";
            SaveSessionIfNeeded();
        }, showProgress: true);
    }

    public async Task RecomputeFromThresholdsAsync()
    {
        if (!_recomputeMasks || _session.Flows.Count == 0)
            return;

        await RunTaskAsync("Recomputing masks…", async () =>
        {
            var result = await _ml.RecomputeAsync(_session.Flows, SegmentationParams);
            if (result.Masks != null)
                ApplyMaskUpdate(result.Masks);
            StatusMessage = $"Recomputed {result.Ncells} cells";
            SaveSessionIfNeeded();
        });
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
        if (_session.ImagePath == null || _session.Masks == null)
            return;

        await RunTaskAsync("Saving…", async () =>
        {
            var path = await Task.Run(() =>
            {
                var savePath = _sessionStore.DefaultPath(_session.ImagePath!);
                _sessionStore.Save(savePath, _session);
                return savePath;
            });
            StatusMessage = $"Saved {Path.GetFileName(path)}";
        });
    }

    public async Task ExportMasksAsync()
    {
        await ExportWithPanelAsync("_cp_masks.png", [".png", ".tif", ".tiff"], (path) =>
        {
            if (_session.Masks == null)
                throw new SidecarException("No masks to export");
            var format = path.Contains("tif", StringComparison.OrdinalIgnoreCase) ? "tif" : "png";
            _exportService.ExportMasks(_session.Masks, path, format);
            return path;
        });
    }

    public async Task LoadMasksPanelAsync()
    {
        var path = await PickFileAsync([".tif", ".tiff", ".png"]);
        if (path != null)
            StatusMessage = $"Load masks not yet wired for {Path.GetFileName(path)}";
    }

    public async Task ExportOutlinesAsync() =>
        await ExportWithPanelAsync("_outline.txt", [".txt"], path =>
        {
            if (_session.Masks == null)
                throw new SidecarException("No masks to export");
            _exportService.ExportOutlines(_session.Masks, path);
            return path;
        });

    public async Task ExportFlowsAsync() =>
        await ExportWithPanelAsync("_flows.tif", [".tif", ".tiff"], path =>
        {
            _exportService.ExportFlows(_session.Flows, path);
            return path;
        });

    public async Task ExportROIsAsync() =>
        await ExportWithPanelAsync("_rois.zip", [".zip"], path =>
        {
            if (_session.Masks == null)
                throw new SidecarException("No masks to export");
            _exportService.ExportRois(_session.Masks, path);
            return path;
        });

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
            await _ml.RemoveModelAsync(name);
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

    public Task RemoveCellsAsync(IReadOnlyList<int> indices)
    {
        if (_session.Masks == null)
            return Task.CompletedTask;

        return RunTaskAsync("Removing cells…", async () =>
        {
            await Task.Yield();
            var updated = MaskEditService.RemoveCells(_session.Masks, indices);
            ApplyMaskUpdate(updated);
            SelectedCell = 0;
            SaveSessionIfNeeded();
        });
    }

    public Task MergeCellsAsync(int source, int target)
    {
        if (_session.Masks == null)
            return Task.CompletedTask;

        return RunTaskAsync("Merging cells…", async () =>
        {
            await Task.Yield();
            var updated = MaskEditService.MergeCells(_session.Masks, source, target);
            ApplyMaskUpdate(updated);
            SaveSessionIfNeeded();
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

    public Task FinishDrawingAsync()
    {
        if (_session.Image == null || _pendingStrokes.Count == 0)
            return Task.CompletedTask;

        var strokes = _pendingStrokes.ToArray();
        _pendingStrokes.Clear();

        return RunTaskAsync("Adding cell…", async () =>
        {
            await Task.Yield();
            var updated = MaskEditService.AddMaskFromStrokes(
                _session.Masks,
                _session.Image!.Width,
                _session.Image.Height,
                strokes,
                _defaultClassId);
            if (updated == null)
                throw new SidecarException("Cell too small to draw");
            ApplyMaskUpdate(updated);
            SaveSessionIfNeeded();
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
            var result = await _ml.TrainAsync(TrainingParams);
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
            var name = await _ml.AddModelAsync(path);
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

        if (Path.GetExtension(path).Equals(".cellpose", StringComparison.OrdinalIgnoreCase))
            await LoadSegAsync(path);
        else
            await LoadImageAsync(path);
    }

    private Task LoadCurrentSeriesRecordAsync()
    {
        if (_seriesNavigationLocked)
            return Task.CompletedTask;

        if (ResolveCurrentSeriesRecordIndex() is not int recordIndex ||
            recordIndex == SeriesState.RecordIndex)
            return Task.CompletedTask;

        SeriesState.RecordIndex = recordIndex;
        var record = SeriesState.CurrentRecord;
        if (record == null)
            return Task.CompletedTask;

        return RunTaskAsync("Loading frame…", async () =>
        {
            var image = await Task.Run(() => _imageLoader.Load(record.Path));
            ApplyLoadedImage(record.Path, image);
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
        Func<string, string> export)
    {
        if (_filename == null)
            return;

        var defaultName = Path.GetFileNameWithoutExtension(_filename) + defaultSuffix;
        var path = await PickSaveFileAsync(defaultName, extensions);
        if (path == null)
            return;

        await RunTaskAsync("Exporting…", async () =>
        {
            var saved = await Task.Run(() => export(path));
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
