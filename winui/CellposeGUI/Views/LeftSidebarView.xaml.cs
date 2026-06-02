using CellposeGUI.Models;
using CellposeGUI.ViewModels;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CellposeGUI.Views;

public sealed partial class LeftSidebarView : UserControl
{
    private MainViewModel? _viewModel;
    private bool _suppressAxisSliderEvents;
    private bool _suppressGraySliderEvents;
    private bool _suppressMaskBlendSliderEvents;

    public LeftSidebarView()
    {
        InitializeComponent();
        WireEvents();
        BuildAxisRows();
    }

    public MainViewModel? ViewModel
    {
        get => _viewModel;
        set
        {
            if (_viewModel != null)
            {
                _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
                _viewModel.SeriesState.PropertyChanged -= OnSeriesStateChanged;
                _viewModel.DisplayParams.PropertyChanged -= OnDisplayParamsChanged;
            }
            _viewModel = value;
            if (_viewModel != null)
            {
                _viewModel.PropertyChanged += OnViewModelPropertyChanged;
                _viewModel.SeriesState.PropertyChanged += OnSeriesStateChanged;
                _viewModel.DisplayParams.PropertyChanged += OnDisplayParamsChanged;
            }
            BindViewModel();
        }
    }

    private void WireEvents()
    {
        ViewModeCombo.SelectionChanged += (_, _) =>
        {
            if (_viewModel == null || ViewModeCombo.SelectedItem is not ComboBoxItem item || item.Tag is not ViewMode mode)
                return;
            _viewModel.ViewMode = mode;
        };

        AutoSaturationButton.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.ComputeSaturationAsync();
        };

        GrayLowSlider.ValueChanged += (_, _) =>
        {
            if (_suppressGraySliderEvents)
                return;

            UpdateGraySliderLabels();
        };

        GrayLowSlider.PointerReleased += (_, _) => CommitGrayLow();
        GrayLowSlider.PointerCaptureLost += (_, _) => CommitGrayLow();

        GrayHighSlider.ValueChanged += (_, _) =>
        {
            if (_suppressGraySliderEvents)
                return;

            UpdateGraySliderLabels();
        };

        GrayHighSlider.PointerReleased += (_, _) => CommitGrayHigh();
        GrayHighSlider.PointerCaptureLost += (_, _) => CommitGrayHigh();

        MaskBlendSlider.ValueChanged += (_, _) =>
        {
            if (_suppressMaskBlendSliderEvents)
                return;

            CommitMaskBlend();
        };

        DefaultClassBox.ValueChanged += (_, e) =>
        {
            if (_viewModel != null)
                _viewModel.DefaultClassId = (int)e.NewValue;
        };

        BrushButton.Checked += (_, _) =>
        {
            if (_viewModel != null)
                _viewModel.BrushMode = true;
        };

        BrushButton.Unchecked += (_, _) =>
        {
            if (_viewModel != null)
                _viewModel.BrushMode = false;
        };

        SelectButton.Checked += (_, _) =>
        {
            if (_viewModel != null)
                _viewModel.SelectMode = true;
        };

        SelectButton.Unchecked += (_, _) =>
        {
            if (_viewModel != null)
                _viewModel.SelectMode = false;
        };

        DeleteSelectedButton.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.DeleteSelectedCellsAsync();
        };

        EditSelectedButton.Click += async (_, _) => await ShowEditClassDialogAsync();
    }

    private async Task ShowEditClassDialogAsync()
    {
        if (_viewModel == null)
            return;

        var cells = _viewModel.SelectedCellIndices();
        if (cells.Count == 0)
        {
            var info = new ContentDialog
            {
                Title = "Edit class",
                Content = "Select one or more cells first.",
                CloseButtonText = "OK",
                XamlRoot = XamlRoot,
            };
            await info.ShowAsync();
            return;
        }

        int? initial = null;
        var classIds = cells
            .Select(idx => idx - 1)
            .Where(row => row >= 0 && row < _viewModel.InstanceClasses.Values.Count)
            .Select(row => _viewModel.InstanceClasses.Values[row])
            .ToList();
        if (classIds.Count > 0 && classIds.Distinct().Count() == 1)
            initial = classIds[0];

        var classBox = new NumberBox
        {
            Minimum = 0,
            SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Inline,
            Value = initial ?? 0,
        };

        var dialog = new ContentDialog
        {
            Title = "Edit class",
            Content = new StackPanel
            {
                Spacing = 8,
                Children =
                {
                    new TextBlock { Text = "Class ID for selected cell(s):" },
                    classBox,
                },
            },
            PrimaryButtonText = "OK",
            CloseButtonText = "Cancel",
            XamlRoot = XamlRoot,
        };

        if (await dialog.ShowAsync() != ContentDialogResult.Primary)
            return;

        _viewModel.ApplyClassToSelectedCells((int)classBox.Value);
    }

    private void BuildAxisRows()
    {
        AxisPanel.Children.Clear();
        foreach (var axis in SeriesState.AxisOrder)
        {
            var label = SeriesState.AxisLabels[axis];
            var row = new Grid
            {
                Tag = axis,
                HorizontalAlignment = HorizontalAlignment.Stretch,
                ColumnSpacing = 6,
            };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(16) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });

            var text = new TextBlock { Text = label, VerticalAlignment = VerticalAlignment.Center };
            var prev = new Button { Content = "<", Tag = axis, MinWidth = 28 };
            var slider = new Slider { Minimum = 0, Maximum = 1, Tag = axis, HorizontalAlignment = HorizontalAlignment.Stretch };
            var next = new Button { Content = ">", Tag = axis, MinWidth = 28 };

            prev.Click += async (_, _) =>
            {
                if (_viewModel != null && prev.Tag is string axisName)
                    await _viewModel.NavigateSeriesAxisAsync(axisName, -1);
            };
            next.Click += async (_, _) =>
            {
                if (_viewModel != null && next.Tag is string axisName)
                    await _viewModel.NavigateSeriesAxisAsync(axisName, 1);
            };
            slider.ValueChanged += (_, e) =>
            {
                if (_suppressAxisSliderEvents || _viewModel == null || slider.Tag is not string axisName)
                    return;

                var target = (int)e.NewValue;
                if (target == _viewModel.SeriesState.GetAxisIndex(axisName))
                    return;

                _ = SetSeriesAxisSafeAsync(axisName, target);
            };

            Grid.SetColumn(text, 0);
            Grid.SetColumn(prev, 1);
            Grid.SetColumn(slider, 2);
            Grid.SetColumn(next, 3);
            row.Children.Add(text);
            row.Children.Add(prev);
            row.Children.Add(slider);
            row.Children.Add(next);
            AxisPanel.Children.Add(row);
        }
    }

    private void RefreshAxisRows()
    {
        if (_viewModel == null)
            return;

        _suppressAxisSliderEvents = true;
        try
        {
            foreach (var child in AxisPanel.Children.OfType<Grid>())
            {
                if (child.Tag is not string axis)
                    continue;

                var values = _viewModel.SeriesState.AxisValues.GetValueOrDefault(axis) ?? [];
                var maxIndex = Math.Max(values.Count - 1, 0);
                var index = _viewModel.SeriesState.GetAxisIndex(axis);
                var enabled = _viewModel.SeriesState.IsLoaded && values.Count > 1;

                foreach (var element in child.Children)
                {
                    switch (element)
                    {
                        case Button button:
                            button.IsEnabled = enabled;
                            break;
                        case Slider slider:
                            slider.Maximum = Math.Max(maxIndex, 1);
                            slider.Value = Math.Min(index, maxIndex);
                            slider.IsEnabled = enabled;
                            break;
                    }
                }
            }

            NavigationGroup.IsEnabled = _viewModel.SeriesState.IsLoaded;
        }
        finally
        {
            _suppressAxisSliderEvents = false;
        }
    }

    private async Task SetSeriesAxisSafeAsync(string axis, int index)
    {
        try
        {
            if (_viewModel != null)
                await _viewModel.SetSeriesAxisIndexAsync(axis, index);
        }
        catch (Exception ex)
        {
            if (_viewModel != null)
                _viewModel.ErrorMessage = ex.Message;
        }
    }

    private void BindViewModel()
    {
        if (_viewModel == null)
            return;

        RefreshViewModeCombo();
        SyncGraySliders();
        SyncMaskBlendSlider();
        UpdateControlStates();
        RefreshAxisRows();

        DefaultClassBox.Value = _viewModel.DefaultClassId;
        BrushButton.IsChecked = _viewModel.BrushMode;
        SelectButton.IsChecked = _viewModel.SelectMode;
        UpdateLabelToolButtons();
    }

    private void SyncGraySliders()
    {
        if (_viewModel == null)
            return;

        _suppressGraySliderEvents = true;
        try
        {
            GrayLowSlider.Value = _viewModel.DisplayParams.GrayLow;
            GrayHighSlider.Value = _viewModel.DisplayParams.GrayHigh;
            UpdateGraySliderLabels();
        }
        finally
        {
            _suppressGraySliderEvents = false;
        }
    }

    private void UpdateGraySliderLabels()
    {
        GrayLowValueText.Text = $"{GrayLowSlider.Value:0}";
        GrayHighValueText.Text = $"{GrayHighSlider.Value:0}";
    }

    private void SyncMaskBlendSlider()
    {
        if (_viewModel == null)
            return;

        _suppressMaskBlendSliderEvents = true;
        try
        {
            MaskBlendSlider.Value = _viewModel.DisplayParams.MaskBlend;
            MaskBlendValueText.Text = $"{MaskBlendSlider.Value:0.00}";
        }
        finally
        {
            _suppressMaskBlendSliderEvents = false;
        }
    }

    private void CommitMaskBlend()
    {
        if (_suppressMaskBlendSliderEvents || _viewModel == null)
            return;

        _viewModel.DisplayParams.MaskBlend = MaskBlendSlider.Value;
        MaskBlendValueText.Text = $"{MaskBlendSlider.Value:0.00}";
    }

    private void CommitGrayLow()
    {
        if (_suppressGraySliderEvents || _viewModel == null)
            return;

        var low = Math.Min(GrayLowSlider.Value, _viewModel.DisplayParams.GrayHigh - 1);
        _viewModel.DisplayParams.GrayLow = Math.Max(0, low);
        UpdateGraySliderLabels();
    }

    private void CommitGrayHigh()
    {
        if (_suppressGraySliderEvents || _viewModel == null)
            return;

        var high = Math.Max(GrayHighSlider.Value, _viewModel.DisplayParams.GrayLow + 1);
        _viewModel.DisplayParams.GrayHigh = Math.Min(255, high);
        UpdateGraySliderLabels();
    }

    private void UpdateControlStates()
    {
        if (_viewModel == null)
            return;

        AutoSaturationButton.IsEnabled = _viewModel.ImageLoaded && !_viewModel.IsBusy;
        BrushButton.IsEnabled = _viewModel.ImageLoaded;
        SelectButton.IsEnabled = _viewModel.CanUseLabelTools;
        DeleteSelectedButton.IsEnabled = _viewModel.CanUseLabelTools;
        EditSelectedButton.IsEnabled = _viewModel.CanUseLabelTools;
        GrayLowSlider.IsEnabled = _viewModel.ImageLoaded;
        GrayHighSlider.IsEnabled = _viewModel.ImageLoaded;
        MaskBlendSlider.IsEnabled = _viewModel.ImageLoaded;
    }

    private void UpdateLabelToolButtons()
    {
        if (_viewModel == null)
            return;

        SelectButton.IsEnabled = _viewModel.CanUseLabelTools;
        DeleteSelectedButton.IsEnabled = _viewModel.CanUseLabelTools;
        EditSelectedButton.IsEnabled = _viewModel.CanUseLabelTools;
    }

    private void RefreshViewModeCombo()
    {
        if (_viewModel == null)
            return;

        ViewModeCombo.Items.Clear();
        foreach (var mode in ViewModeExtensions.All)
        {
            ViewModeCombo.Items.Add(new ComboBoxItem
            {
                Content = mode.Title(),
                Tag = mode,
                IsEnabled = mode == ViewMode.Image,
            });
        }
        ViewModeCombo.SelectedIndex = Math.Max(0, Array.IndexOf(ViewModeExtensions.All, _viewModel.ViewMode));
    }

    private void OnDisplayParamsChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(DisplayParameters.GrayLow) or nameof(DisplayParameters.GrayHigh))
            SyncGraySliders();
        if (e.PropertyName is nameof(DisplayParameters.MaskBlend))
            SyncMaskBlendSlider();
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(MainViewModel.ImageLoaded):
            case nameof(MainViewModel.IsBusy):
            case nameof(MainViewModel.Ncells):
                UpdateControlStates();
                break;
            case nameof(MainViewModel.ViewMode):
                RefreshViewModeCombo();
                break;
            case nameof(MainViewModel.BrushMode):
                BrushButton.IsChecked = _viewModel?.BrushMode ?? false;
                break;
            case nameof(MainViewModel.SelectMode):
                SelectButton.IsChecked = _viewModel?.SelectMode ?? false;
                break;
            case nameof(MainViewModel.SelectionRevision):
                break;
        }
    }

    private void OnSeriesStateChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e) =>
        RefreshAxisRows();
}
