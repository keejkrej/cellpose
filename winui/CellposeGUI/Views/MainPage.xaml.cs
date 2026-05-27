using CellposeGUI.ViewModels;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CellposeGUI.Views;

public sealed partial class MainPage : Page
{
    private MainViewModel? _viewModel;
    private bool _showingErrorDialog;

    public MainPage()
    {
        InitializeComponent();
        WireMenuEvents();
    }

    public MainViewModel? ViewModel
    {
        get => _viewModel;
        set
        {
            if (_viewModel != null)
                _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
            _viewModel = value;
            LeftSidebar.ViewModel = value;
            RightSidebar.ViewModel = value;
            Canvas.ViewModel = value;
            if (_viewModel != null)
            {
                _viewModel.PropertyChanged += OnViewModelPropertyChanged;
                UpdateBindings();
            }
        }
    }

    protected override void OnNavigatedTo(Microsoft.UI.Xaml.Navigation.NavigationEventArgs e)
    {
        base.OnNavigatedTo(e);
        if (e.Parameter is MainViewModel viewModel)
            ViewModel = viewModel;
    }

    private void WireMenuEvents()
    {
        LoadImageItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.LoadImagePanelAsync();
        };

        LoadFolderItem.Click += async (_, _) =>
        {
            if (_viewModel == null)
                return;
            await _viewModel.PresentLoadFolderPanelAsync();
            if (_viewModel.PendingFolderPath == null)
                return;
            await ShowLoadFolderDialogAsync();
        };

        AutoloadMasksItem.Click += (_, _) =>
        {
            if (_viewModel != null)
                _viewModel.AutoloadMasks = AutoloadMasksItem.IsChecked;
        };

        DisableAutosaveItem.Click += (_, _) =>
        {
            if (_viewModel != null)
                _viewModel.DisableAutosave = DisableAutosaveItem.IsChecked;
        };

        LoadMasksItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.LoadMasksPanelAsync();
        };

        LoadSegItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.LoadSegPanelAsync();
        };

        SaveSegItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.SaveSegAsync();
        };

        ExportMasksItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.ExportMasksAsync();
        };

        ExportOutlinesItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.ExportOutlinesAsync();
        };

        ExportROIsItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.ExportROIsAsync();
        };

        ExportFlowsItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.ExportFlowsAsync();
        };

        UndoItem.Click += (_, _) => _viewModel?.UndoAction();
        UndoRemoveItem.Click += (_, _) => _viewModel?.UndoRemoveAction();

        ClearMasksItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.ClearAllMasksAsync();
        };

        AddModelItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.AddCustomModelAsync();
        };

        RemoveModelItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.RemoveSelectedModelAsync();
        };

        TrainModelItem.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await ShowTrainDialogAsync();
        };

        Canvas.ZoomChanged += (_, zoom) => ZoomText.Text = $"{zoom * 100:0}%";
    }

    private async Task ShowLoadFolderDialogAsync()
    {
        if (_viewModel?.PendingFolderPath == null)
            return;

        var folder = _viewModel.PendingFolderPath;
        var suggested = await _viewModel.FetchSeriesTemplateSuggestionsAsync(folder);

        var templates = await RunOnUiAsync<(string Subfolder, string Filename)?>(async () =>
        {
            var subfolderBox = new TextBox { Header = "Subfolder template", Text = suggested.Subfolder };
            var filenameBox = new TextBox { Header = "Filename template", Text = suggested.Filename };

            var dialog = new ContentDialog
            {
                Title = "Load Folder with Pattern",
                PrimaryButtonText = "Load",
                CloseButtonText = "Cancel",
                DefaultButton = ContentDialogButton.Primary,
                XamlRoot = XamlRoot,
                Content = new StackPanel
                {
                    Spacing = 12,
                    Children =
                    {
                        new TextBlock
                        {
                            Text = "Use placeholders {t}, {p}, {c}, {z}. Subfolder matching is case-insensitive.",
                            TextWrapping = TextWrapping.Wrap,
                        },
                        subfolderBox,
                        filenameBox,
                    },
                },
            };

            if (await dialog.ShowAsync() != ContentDialogResult.Primary)
                return null;

            return (Subfolder: subfolderBox.Text, Filename: filenameBox.Text);
        });

        if (templates == null)
            return;

        await _viewModel.LoadSeriesFolderAsync(templates.Value.Subfolder, templates.Value.Filename);
    }

    private async Task ShowTrainDialogAsync()
    {
        if (_viewModel == null)
            return;

        var confirmed = await RunOnUiAsync(async () =>
        {
            var trainDataBox = new TextBox { Header = "Train data folder", Text = _viewModel.TrainingParams.TrainDataFolder };
            var modelNameBox = new TextBox { Header = "Model name", Text = _viewModel.TrainingParams.ModelName };
            var learningRateBox = new NumberBox { Header = "Learning rate", Value = _viewModel.TrainingParams.LearningRate, SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Inline };
            var weightDecayBox = new NumberBox { Header = "Weight decay", Value = _viewModel.TrainingParams.WeightDecay, SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Inline };
            var epochsBox = new NumberBox { Header = "Epochs", Value = _viewModel.TrainingParams.NEpochs, SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Inline };
            var saveFolderBox = new TextBox { Header = "Save folder", Text = _viewModel.TrainingParams.ModelSaveFolder };

            var dialog = new ContentDialog
            {
                Title = "Train Settings",
                PrimaryButtonText = "Train",
                CloseButtonText = "Cancel",
                DefaultButton = ContentDialogButton.Primary,
                XamlRoot = XamlRoot,
                Content = new StackPanel
                {
                    Spacing = 12,
                    MinWidth = 420,
                    Children =
                    {
                        trainDataBox,
                        modelNameBox,
                        learningRateBox,
                        weightDecayBox,
                        epochsBox,
                        saveFolderBox,
                    },
                },
            };

            if (await dialog.ShowAsync() != ContentDialogResult.Primary)
                return false;

            _viewModel.TrainingParams.TrainDataFolder = trainDataBox.Text;
            _viewModel.TrainingParams.ModelName = modelNameBox.Text;
            _viewModel.TrainingParams.LearningRate = learningRateBox.Value;
            _viewModel.TrainingParams.WeightDecay = weightDecayBox.Value;
            _viewModel.TrainingParams.NEpochs = (int)epochsBox.Value;
            _viewModel.TrainingParams.ModelSaveFolder = saveFolderBox.Text;
            return true;
        });

        if (confirmed)
            await _viewModel.TrainModelAsync();
    }

    private Task<T> RunOnUiAsync<T>(Func<Task<T>> action)
    {
        if (DispatcherQueue.HasThreadAccess)
            return action();

        var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
        DispatcherQueue.TryEnqueue(async () =>
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

    private void UpdateBindings()
    {
        if (_viewModel == null)
            return;

        StatusText.Text = _viewModel.StatusMessage;
        StatusProgress.Value = _viewModel.Progress;
        StatusProgress.Visibility = _viewModel.IsBusy ? Visibility.Visible : Visibility.Collapsed;
        CellCountText.Text = $"{_viewModel.Ncells} cells";
        ZoomText.Text = "100%";

        LoadMasksItem.IsEnabled = _viewModel.ImageLoaded;
        SaveSegItem.IsEnabled = _viewModel.CanSaveMasks;
        ExportMasksItem.IsEnabled = _viewModel.CanSaveMasks;
        ExportOutlinesItem.IsEnabled = _viewModel.CanSaveMasks;
        ExportROIsItem.IsEnabled = _viewModel.CanSaveMasks;
        ExportFlowsItem.IsEnabled = _viewModel.CanSaveMasks;
        ClearMasksItem.IsEnabled = _viewModel.CanSaveMasks;
        RemoveModelItem.IsEnabled = _viewModel.IsCustomModel;
        TrainModelItem.IsEnabled = _viewModel.ImageLoaded;

        AutoloadMasksItem.IsChecked = _viewModel.AutoloadMasks;
        DisableAutosaveItem.IsChecked = _viewModel.DisableAutosave;
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (!DispatcherQueue.HasThreadAccess)
        {
            DispatcherQueue.TryEnqueue(() => OnViewModelPropertyChanged(sender, e));
            return;
        }

        UpdateBindings();

        if (e.PropertyName == nameof(MainViewModel.WindowTitle))
            App.CurrentWindow.Title = _viewModel?.WindowTitle ?? "Cellpose";

        if (e.PropertyName == nameof(MainViewModel.ErrorMessage) &&
            !string.IsNullOrWhiteSpace(_viewModel?.ErrorMessage) &&
            !_showingErrorDialog)
        {
            _ = ShowErrorDialogAsync(_viewModel!.ErrorMessage!);
        }
    }

    private async Task ShowErrorDialogAsync(string message)
    {
        _showingErrorDialog = true;
        try
        {
            var dialog = new ContentDialog
            {
                Title = "Error",
                Content = message,
                CloseButtonText = "OK",
                XamlRoot = XamlRoot,
            };
            await dialog.ShowAsync();
            if (_viewModel != null)
                _viewModel.ErrorMessage = null;
        }
        finally
        {
            _showingErrorDialog = false;
        }
    }
}
