using CellposeGUI.ViewModels;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CellposeGUI.Views;

public sealed partial class RightSidebarView : UserControl
{
    private MainViewModel? _viewModel;
    private bool _suppressModelComboEvents;
    private IReadOnlyList<string> _boundModels = Array.Empty<string>();

    public RightSidebarView()
    {
        InitializeComponent();
        WireEvents();
    }

    public MainViewModel? ViewModel
    {
        get => _viewModel;
        set
        {
            if (_viewModel != null)
                _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
            _viewModel = value;
            if (_viewModel != null)
                _viewModel.PropertyChanged += OnViewModelPropertyChanged;
            _boundModels = Array.Empty<string>();
            BindViewModel();
        }
    }

    private void WireEvents()
    {
        ModelCombo.SelectionChanged += (_, _) =>
        {
            if (_suppressModelComboEvents || _viewModel == null || ModelCombo.SelectedIndex < 0)
                return;
            _viewModel.SelectedModelIndex = ModelCombo.SelectedIndex;
        };

        DiameterBox.ValueChanged += (_, e) =>
        {
            if (_viewModel != null)
                _viewModel.SegmentationParams.Diameter = e.NewValue;
        };

        FlowThresholdBox.ValueChanged += async (_, e) =>
        {
            if (_viewModel != null)
            {
                _viewModel.SegmentationParams.FlowThreshold = e.NewValue;
                await _viewModel.RecomputeFromThresholdsAsync();
            }
        };

        CellprobThresholdBox.ValueChanged += async (_, e) =>
        {
            if (_viewModel != null)
            {
                _viewModel.SegmentationParams.CellprobThreshold = e.NewValue;
                await _viewModel.RecomputeFromThresholdsAsync();
            }
        };

        PercentileLowBox.ValueChanged += (_, e) =>
        {
            if (_viewModel != null)
                _viewModel.SegmentationParams.PercentileLow = e.NewValue;
        };

        PercentileHighBox.ValueChanged += (_, e) =>
        {
            if (_viewModel != null)
                _viewModel.SegmentationParams.PercentileHigh = e.NewValue;
        };

        NiterBox.ValueChanged += async (_, e) =>
        {
            if (_viewModel != null)
            {
                _viewModel.SegmentationParams.Niter = (int)e.NewValue;
                await _viewModel.RecomputeFromThresholdsAsync();
            }
        };

        RunSegmentationButton.Click += async (_, _) =>
        {
            if (_viewModel != null)
                await _viewModel.RunSegmentationAsync();
        };

        ClassFilterBox.TextChanged += (_, _) =>
        {
            if (_viewModel != null)
                _viewModel.ClassFilterText = ClassFilterBox.Text;
        };
    }

    private void BindViewModel()
    {
        if (_viewModel == null)
            return;

        BindModelCombo();

        DiameterBox.Value = _viewModel.SegmentationParams.Diameter;
        FlowThresholdBox.Value = _viewModel.SegmentationParams.FlowThreshold;
        CellprobThresholdBox.Value = _viewModel.SegmentationParams.CellprobThreshold;
        PercentileLowBox.Value = _viewModel.SegmentationParams.PercentileLow;
        PercentileHighBox.Value = _viewModel.SegmentationParams.PercentileHigh;
        NiterBox.Value = _viewModel.SegmentationParams.Niter;

        ClassFilterBox.Text = _viewModel.ClassFilterText;

        RunSegmentationButton.IsEnabled = _viewModel.CanRunSegmentation;
        RunProgressBar.Visibility = _viewModel.IsBusy ? Visibility.Visible : Visibility.Collapsed;
        RunProgressBar.Value = _viewModel.Progress;

        RefreshInstanceList();
    }

    private void BindModelCombo()
    {
        if (_viewModel == null)
            return;

        var models = _viewModel.Models;
        var modelsChanged = _boundModels.Count != models.Count ||
            !_boundModels.SequenceEqual(models, StringComparer.OrdinalIgnoreCase);

        _suppressModelComboEvents = true;
        try
        {
            if (modelsChanged)
            {
                _boundModels = models;
                ModelCombo.ItemsSource = models;
            }

            if (models.Count == 0)
            {
                ModelCombo.SelectedIndex = -1;
                return;
            }

            var index = Math.Clamp(_viewModel.SelectedModelIndex, 0, models.Count - 1);
            ModelCombo.SelectedIndex = index;
            if (index != _viewModel.SelectedModelIndex)
                _viewModel.SelectedModelIndex = index;
        }
        finally
        {
            _suppressModelComboEvents = false;
        }
    }

    private void UpdateSelectedModelIndex()
    {
        if (_viewModel == null || ModelCombo.Items.Count == 0)
            return;

        var index = Math.Clamp(_viewModel.SelectedModelIndex, 0, ModelCombo.Items.Count - 1);
        if (ModelCombo.SelectedIndex == index)
            return;

        _suppressModelComboEvents = true;
        try
        {
            ModelCombo.SelectedIndex = index;
        }
        finally
        {
            _suppressModelComboEvents = false;
        }
    }

    private void RefreshInstanceList()
    {
        if (_viewModel == null)
            return;

        var rows = Enumerable.Range(0, _viewModel.Ncells)
            .Select(row => new InstanceRowViewModel(_viewModel, row))
            .ToList();
        InstanceList.ItemsSource = rows;
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(MainViewModel.Models):
                BindModelCombo();
                break;
            case nameof(MainViewModel.SelectedModelIndex):
                UpdateSelectedModelIndex();
                break;
            case nameof(MainViewModel.IsBusy):
            case nameof(MainViewModel.Progress):
            case nameof(MainViewModel.CanRunSegmentation):
                if (_viewModel != null)
                {
                    RunSegmentationButton.IsEnabled = _viewModel.CanRunSegmentation;
                    RunProgressBar.Visibility = _viewModel.IsBusy ? Visibility.Visible : Visibility.Collapsed;
                    RunProgressBar.Value = _viewModel.Progress;
                }
                break;
            case nameof(MainViewModel.Ncells):
                RefreshInstanceList();
                break;
        }
    }

    private sealed class InstanceRowViewModel
    {
        private readonly MainViewModel _viewModel;
        private readonly int _row;

        public InstanceRowViewModel(MainViewModel viewModel, int row)
        {
            _viewModel = viewModel;
            _row = row;
        }

        public string RoiLabel => (_row + 1).ToString();

        public double ClassId
        {
            get => _row < _viewModel.InstanceClasses.Values.Count ? _viewModel.InstanceClasses.Values[_row] : 0;
            set => _viewModel.InstanceClasses.SetClass(_row, (int)value);
        }
    }
}
