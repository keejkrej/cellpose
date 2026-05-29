using System.ComponentModel;
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
    private bool _imageLoaded;
    private bool _isBusy;
    private bool _isSegmentationRunning;
    private double _progress;
    private double _segmentationProgress;
    private string _statusMessage = "Ready";
    private string? _errorMessage;
    private bool _recomputeMasks;
    private ViewMode _viewMode = ViewMode.Image;
    private int _selectedModelIndex;
    private string _classFilterText = "";
    private int _defaultClassId;
    private bool _brushMode;
    private bool _selectMode;
    private bool _inStroke;
    private readonly List<int> _selectedCells = [];
    private List<string> _models = ["CPSAM"];

    public MainViewModel(
        IMlInferenceEngine ml,
        ImageLoaderService imageLoader,
        CellposeSessionStore sessionStore,
        SeriesDiscoveryService seriesDiscovery,
        DispatcherQueue dispatcher)
    {
        _ml = ml;
        _imageLoader = imageLoader;
        _sessionStore = sessionStore;
        _seriesDiscovery = seriesDiscovery;
        _dispatcher = dispatcher;
        BindDispatcher(dispatcher);
        SegmentationParams = new SegmentationParameters();
        SegmentationParams.BindDispatcher(dispatcher);
        SegmentationParams.PropertyChanged += OnSegmentationParamsChanged;
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
        InstanceVisibility = new InstanceVisibility();
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
            Notify(nameof(CanUseLabelTools));
        }
    }

    public int SelectedCell
    {
        get => _selectedCells.Count > 0 ? _selectedCells[0] : _selectedCell;
        set => SetCellSelection(value > 0 ? [value] : []);
    }

    public IReadOnlyList<int> SelectedCells => _selectedCells;

    public int SelectionRevision { get; private set; }

    public bool CanUseLabelTools => _imageLoaded && _ncells > 0;

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

    public bool ImageLoaded
    {
        get => _imageLoaded;
        private set
        {
            SetProperty(ref _imageLoaded, value);
            Notify(nameof(CanRunSegmentation));
            Notify(nameof(CanSaveMasks));
            Notify(nameof(CanUseLabelTools));
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

    public bool IsSegmentationRunning
    {
        get => _isSegmentationRunning;
        private set
        {
            SetProperty(ref _isSegmentationRunning, value);
            Notify(nameof(SegmentationProgressVisibility));
            Notify(nameof(CanRunSegmentation));
        }
    }

    public Visibility SegmentationProgressVisibility =>
        _isSegmentationRunning ? Visibility.Visible : Visibility.Collapsed;

    public double SegmentationProgress
    {
        get => _segmentationProgress;
        private set => SetProperty(ref _segmentationProgress, value);
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
    public InstanceVisibility InstanceVisibility { get; }

    public bool? AllLabelsVisible
    {
        get
        {
            if (_ncells == 0)
                return false;

            var visibleCount = 0;
            for (var row = 0; row < _ncells; row++)
            {
                if (row < InstanceVisibility.Values.Count && InstanceVisibility.Values[row])
                    visibleCount++;
            }

            if (visibleCount == 0)
                return false;
            if (visibleCount == _ncells)
                return true;
            return null;
        }
        set
        {
            if (_ncells == 0)
                return;

            SetAllInstanceVisible(value != false);
        }
    }

    public string ClassFilterText
    {
        get => _classFilterText;
        set
        {
            SetProperty(ref _classFilterText, value);
            Notify(nameof(FilteredCellCount));
            NotifyCanvasChanged();
        }
    }

    public int DefaultClassId
    {
        get => _defaultClassId;
        set => SetProperty(ref _defaultClassId, value);
    }

    public bool BrushMode
    {
        get => _brushMode;
        set
        {
            if (_brushMode == value)
                return;

            SetProperty(ref _brushMode, value);
            if (value)
            {
                if (_selectMode)
                    SelectMode = false;
            }
            else
            {
                CancelStroke();
            }
        }
    }

    public bool SelectMode
    {
        get => _selectMode;
        set
        {
            if (_selectMode == value)
                return;

            SetProperty(ref _selectMode, value);
            if (value)
            {
                if (_brushMode)
                    BrushMode = false;
            }
        }
    }

    public bool IsCellSelected(int label) => label > 0 && _selectedCells.Contains(label);

    public bool InStroke => _inStroke;

    public IReadOnlyList<double[]> CurrentStroke => _currentStroke;

    public int StrokeRevision { get; private set; }

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
    public bool CanRunSegmentation => _imageLoaded && !_isBusy && !_isSegmentationRunning;

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

    public int LabelsRowsRevision { get; private set; }

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
        SetCellSelection([]);
        InstanceClasses.Replace(0);
        InstanceVisibility.Replace(0);
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
        SegmentationParams.CopyFrom(loaded.Segmentation);

        Filename = loaded.ImagePath;
        Image = loaded.Image;
        RecomputeMasks = loaded.RecomputeMasks;
        SegmentationParams.CopyFrom(loaded.Segmentation);
        SetCellSelection([]);
        Progress = 1;
        ImageLoaded = true;
        UpdateSaturationFromImage();
        ApplyMaskUpdate(loaded.Masks);
    }

    private async Task<(ImageData Image, LoadedSession? Companion)> LoadImageWithOptionalCompanionAsync(string path)
    {
        var image = await Task.Run(() => _imageLoader.Load(path));
        if (!File.Exists(_sessionStore.DefaultPath(path)))
            return (image, null);

        var companion = await Task.Run(() =>
        {
            try
            {
                return _sessionStore.LoadCompanion(_sessionStore.DefaultPath(path), path, image);
            }
            catch
            {
                return null;
            }
        });
        return (image, companion);
    }

    public void ApplyMaskUpdate(MaskData masks, int? appendedClassId = null)
    {
        if (!_dispatcher.HasThreadAccess)
        {
            RunOnUi(() => ApplyMaskUpdate(masks, appendedClassId));
            return;
        }

        var previousNcells = _ncells;
        var ncells = masks.Labels.Length == 0 ? 0 : masks.Labels.Max();
        InstanceClasses.Replace(ncells, InstanceClasses.Values);
        InstanceVisibility.Replace(ncells, InstanceVisibility.Values);

        if (appendedClassId.HasValue && ncells > previousNcells)
            InstanceClasses.SetClass(ncells - 1, appendedClassId.Value);

        masks = MaskEditService.WithClassColors(masks, InstanceClasses.Values);
        _session.Masks = masks;
        Masks = masks;
        Ncells = ncells;
        LabelsRowsRevision++;
        Notify(nameof(LabelsRowsRevision));
        Notify(nameof(FilteredCellCount));
        Notify(nameof(AllLabelsVisible));
        NotifyCanvasChanged();
    }

    public void SetInstanceClass(int row, int classId)
    {
        if (row < 0 || row >= InstanceClasses.Values.Count || classId < 0)
            return;

        InstanceClasses.SetClass(row, classId);
        if (_session.Masks == null)
            return;

        var updated = MaskEditService.WithClassColors(_session.Masks, InstanceClasses.Values);
        _session.Masks = updated;
        Masks = updated;
        LabelsRowsRevision++;
        Notify(nameof(LabelsRowsRevision));
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

    private SegmentationParameters CloneSegmentationParams() => SegmentationParams.Clone();

    private void SaveSessionIfNeeded()
    {
        if (_session.ImagePath == null || _session.Masks == null)
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

    public bool IsInstanceLabelVisible(int label) =>
        InstanceClasses.IsLabelVisible(
            label,
            InstanceClasses.Values,
            InstanceVisibility.Values,
            InstanceClasses.ParseFilter(_classFilterText));

    public void SetInstanceVisible(int row, bool visible)
    {
        if (row < 0 || row >= InstanceVisibility.Values.Count || InstanceVisibility.Values[row] == visible)
            return;

        InstanceVisibility.SetVisible(row, visible);
        LabelsRowsRevision++;
        Notify(nameof(LabelsRowsRevision));
        Notify(nameof(AllLabelsVisible));
        NotifyCanvasChanged();
    }

    public void SetAllInstanceVisible(bool visible)
    {
        InstanceVisibility.SetAll(_ncells, visible);
        LabelsRowsRevision++;
        Notify(nameof(LabelsRowsRevision));
        Notify(nameof(AllLabelsVisible));
        NotifyCanvasChanged();
    }

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
            var (image, companion) = await LoadImageWithOptionalCompanionAsync(path);
            if (companion != null)
            {
                ApplyLoadedSession(companion);
                StatusMessage = $"Loaded {Path.GetFileName(path)} with segmentation";
                return;
            }

            ApplyLoadedImage(path, image);
            StatusMessage = $"Loaded {Path.GetFileName(path)}";
        });

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
            var (image, companion) = await LoadImageWithOptionalCompanionAsync(record.Path);

            var loadedIndex = recordIndex;
            await ApplyOnUiAsync(() =>
            {
                SeriesState.RecordIndex = loadedIndex;
                if (companion != null)
                    ApplyLoadedSession(companion);
                else
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
            var (image, companion) = await LoadImageWithOptionalCompanionAsync(record.Path);
            if (companion != null)
                ApplyLoadedSession(companion);
            else
                ApplyLoadedImage(record.Path, image);
            StatusMessage = record.Label;
        });
    }

    public async void RunSegmentationInvoked(object sender, RoutedEventArgs e) =>
        await RunSegmentationAsync();

    public async Task RunSegmentationAsync()
    {
        if (_session.ImagePath == null)
            return;

        await RunTaskAsync("Running segmentation…", async () =>
        {
            IsSegmentationRunning = true;
            SegmentationProgress = 0;
            try
            {
                SegmentationProgress = 0.1;
                var result = await _ml.InferAsync(
                    _session.ImagePath,
                    SelectedModel,
                    IsCustomModel,
                    SegmentationParams);
                SegmentationProgress = 1;
                ApplyInferenceResult(result);
                StatusMessage = $"Found {result.Ncells} cells";
                SaveSessionIfNeeded();
            }
            finally
            {
                IsSegmentationRunning = false;
            }
        });
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

    private void OnSegmentationParamsChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(SegmentationParameters.FlowThreshold)
            or nameof(SegmentationParameters.CellprobThreshold)
            or nameof(SegmentationParameters.Niter))
        {
            _ = RecomputeFromThresholdsAsync();
        }
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

    public async Task SaveResultsAsync()
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

    public void SetCellSelection(IReadOnlyList<int> cells)
    {
        if (!_dispatcher.HasThreadAccess)
        {
            RunOnUi(() => SetCellSelection(cells));
            return;
        }

        var normalized = cells.Where(c => c > 0).Distinct().OrderBy(c => c).ToList();
        if (_selectedCells.SequenceEqual(normalized))
        {
            SelectionRevision++;
            Notify(nameof(SelectionRevision));
            return;
        }

        _selectedCells.Clear();
        _selectedCells.AddRange(normalized);
        _selectedCell = normalized.FirstOrDefault();
        SelectionRevision++;
        Notify(nameof(SelectedCell));
        Notify(nameof(SelectedCells));
        Notify(nameof(SelectionRevision));
    }

    public IReadOnlyList<int> SelectedCellIndices()
    {
        if (_selectedCells.Count > 0)
            return _selectedCells.ToList();
        if (_selectedCell > 0)
            return [_selectedCell];
        return [];
    }

    public void SelectCellAt(int x, int y, bool additive)
    {
        if (_masks == null)
            return;

        var label = _masks.LabelAt(x, y);
        if (label > 0)
        {
            if (additive)
            {
                var cells = _selectedCells.ToList();
                if (!cells.Contains(label))
                    cells.Add(label);
                SetCellSelection(cells);
            }
            else
            {
                SetCellSelection([label]);
            }

            return;
        }

        if (!additive)
            SetCellSelection([]);
    }

    public void SelectCellsInRect(int x0, int y0, int x1, int y1, bool additive)
    {
        if (_masks == null)
            return;

        var cells = MaskEditService.CellsFullyInRect(
            _masks.Labels,
            _masks.Width,
            _masks.Height,
            x0,
            y0,
            x1,
            y1,
            InstanceClasses.ParseFilter(_classFilterText),
            InstanceClasses.Values);

        if (additive)
        {
            var merged = _selectedCells.Concat(cells).Distinct().OrderBy(c => c).ToList();
            SetCellSelection(merged);
        }
        else
        {
            SetCellSelection(cells);
        }
    }

    public Task DeleteSelectedCellsAsync()
    {
        var cells = SelectedCellIndices();
        return cells.Count == 0 ? Task.CompletedTask : RemoveCellsAsync(cells);
    }

    public void ApplyClassToSelectedCells(int classId)
    {
        foreach (var idx in SelectedCellIndices())
        {
            var row = idx - 1;
            if (row >= 0)
                SetInstanceClass(row, classId);
        }
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
            if (SelectedCell <= 0)
                return;
            _ = MergeCellsAsync(label, SelectedCell);
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
            SetCellSelection([]);
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
        _pendingStrokes.Clear();
        _currentStroke.Add([z, y, x, 0]);
        _inStroke = true;
        Notify(nameof(InStroke));
        NotifyStrokeChanged();
    }

    public void ContinueStroke(int x, int y, int z = 0)
    {
        if (!_inStroke || _currentStroke.Count == 0)
            return;

        var last = _currentStroke[^1];
        if ((int)last[1] == y && (int)last[2] == x)
            return;

        _currentStroke.Add([z, y, x, 0]);
        NotifyStrokeChanged();
    }

    public void CancelStroke()
    {
        _currentStroke.Clear();
        _pendingStrokes.Clear();
        if (!_inStroke)
            return;

        _inStroke = false;
        Notify(nameof(InStroke));
        NotifyStrokeChanged();
    }

    public void CommitStroke()
    {
        if (_currentStroke.Count == 0)
            return;

        _pendingStrokes.Add(_currentStroke.ToArray());
        _currentStroke.Clear();
    }

    public async Task CompleteStrokeAsync(int x, int y, int z = 0)
    {
        if (!_inStroke)
            return;

        ContinueStroke(x, y, z);
        CommitStroke();
        _inStroke = false;
        Notify(nameof(InStroke));
        NotifyStrokeChanged();
        await FinishDrawingAsync();
    }

    private void NotifyStrokeChanged()
    {
        StrokeRevision++;
        Notify(nameof(StrokeRevision));
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
            {
                StatusMessage = "Cell not added";
                return;
            }

            ApplyMaskUpdate(updated, appendedClassId: _defaultClassId);
            SaveSessionIfNeeded();
            StatusMessage = $"Added cell {Ncells}";
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

        if (Path.GetExtension(path).Equals(".npy", StringComparison.OrdinalIgnoreCase) &&
            path.EndsWith("_seg.npy", StringComparison.OrdinalIgnoreCase))
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
            var (image, companion) = await LoadImageWithOptionalCompanionAsync(record.Path);
            if (companion != null)
                ApplyLoadedSession(companion);
            else
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
                if (StatusMessage == message)
                    StatusMessage = "Ready";
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

    private static void InitializePicker(object picker)
    {
        var hwnd = GetWindowHandle();
        if (picker is FileOpenPicker openPicker)
            InitializeWithWindow.Initialize(openPicker, hwnd);
        else if (picker is FolderPicker folderPicker)
            InitializeWithWindow.Initialize(folderPicker, hwnd);
    }
}
