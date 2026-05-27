using Microsoft.UI.Xaml;

namespace CellposeGUI.ViewModels;

public enum SegmentationParamId
{
    Model,
    Diameter,
    FlowThreshold,
    CellprobThreshold,
    PercentileLow,
    PercentileHigh,
    Niter,
}

public sealed class SegmentationParamRowViewModel
{
    private readonly MainViewModel _viewModel;
    private readonly SegmentationParamId _paramId;

    public SegmentationParamRowViewModel(MainViewModel viewModel, SegmentationParamId paramId, string label)
    {
        _viewModel = viewModel;
        _paramId = paramId;
        Label = label;
    }

    public static IReadOnlyList<SegmentationParamRowViewModel> CreateRows(MainViewModel viewModel) =>
    [
        new(viewModel, SegmentationParamId.Model, "model"),
        new(viewModel, SegmentationParamId.Diameter, "diameter"),
        new(viewModel, SegmentationParamId.FlowThreshold, "flow threshold"),
        new(viewModel, SegmentationParamId.CellprobThreshold, "cellprob threshold"),
        new(viewModel, SegmentationParamId.PercentileLow, "norm percentile lower"),
        new(viewModel, SegmentationParamId.PercentileHigh, "norm percentile upper"),
        new(viewModel, SegmentationParamId.Niter, "niter dynamics"),
    ];

    public string Label { get; }

    public Visibility ModelEditorVisibility =>
        _paramId == SegmentationParamId.Model ? Visibility.Visible : Visibility.Collapsed;

    public Visibility PercentileNumberEditorVisibility =>
        _paramId is SegmentationParamId.PercentileLow or SegmentationParamId.PercentileHigh
            ? Visibility.Visible
            : Visibility.Collapsed;

    public Visibility GeneralNumberEditorVisibility =>
        _paramId is >= SegmentationParamId.Diameter and <= SegmentationParamId.Niter
            && _paramId is not (SegmentationParamId.PercentileLow or SegmentationParamId.PercentileHigh)
            ? Visibility.Visible
            : Visibility.Collapsed;

    public IList<string> Models => _viewModel.Models;

    public int SelectedModelIndex
    {
        get => _viewModel.SelectedModelIndex;
        set => _viewModel.SelectedModelIndex = value;
    }

    public double NumberValue
    {
        get => _paramId switch
        {
            SegmentationParamId.Diameter => _viewModel.SegmentationParams.Diameter,
            SegmentationParamId.FlowThreshold => _viewModel.SegmentationParams.FlowThreshold,
            SegmentationParamId.CellprobThreshold => _viewModel.SegmentationParams.CellprobThreshold,
            SegmentationParamId.PercentileLow => _viewModel.SegmentationParams.PercentileLow,
            SegmentationParamId.PercentileHigh => _viewModel.SegmentationParams.PercentileHigh,
            SegmentationParamId.Niter => _viewModel.SegmentationParams.Niter,
            _ => 0,
        };
        set
        {
            switch (_paramId)
            {
                case SegmentationParamId.Diameter:
                    _viewModel.SegmentationParams.Diameter = value;
                    break;
                case SegmentationParamId.FlowThreshold:
                    _viewModel.SegmentationParams.FlowThreshold = value;
                    break;
                case SegmentationParamId.CellprobThreshold:
                    _viewModel.SegmentationParams.CellprobThreshold = value;
                    break;
                case SegmentationParamId.PercentileLow:
                    _viewModel.SegmentationParams.PercentileLow = value;
                    break;
                case SegmentationParamId.PercentileHigh:
                    _viewModel.SegmentationParams.PercentileHigh = value;
                    break;
                case SegmentationParamId.Niter:
                    _viewModel.SegmentationParams.Niter = (int)value;
                    break;
            }
        }
    }

    public double SmallChange => _paramId switch
    {
        SegmentationParamId.FlowThreshold or SegmentationParamId.CellprobThreshold => 0.1,
        SegmentationParamId.PercentileLow or SegmentationParamId.PercentileHigh => 1,
        SegmentationParamId.Niter => 1,
        _ => 1,
    };
}
