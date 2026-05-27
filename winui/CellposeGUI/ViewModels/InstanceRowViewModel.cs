using CellposeGUI.Models;

namespace CellposeGUI.ViewModels;

public sealed class InstanceRowViewModel
{
    private readonly MainViewModel _viewModel;
    private readonly int _row;

    public InstanceRowViewModel(MainViewModel viewModel, int row)
    {
        _viewModel = viewModel;
        _row = row;
    }

    public string RoiLabel => (_row + 1).ToString();

    public bool Visible
    {
        get => _row < _viewModel.InstanceVisibility.Values.Count && _viewModel.InstanceVisibility.Values[_row];
        set => _viewModel.SetInstanceVisible(_row, value);
    }

    public double ClassId
    {
        get => _row < _viewModel.InstanceClasses.Values.Count ? _viewModel.InstanceClasses.Values[_row] : 0;
        set => _viewModel.InstanceClasses.SetClass(_row, (int)value);
    }

    public bool IsHiddenByFilter
    {
        get
        {
            var filter = InstanceClasses.ParseFilter(_viewModel.ClassFilterText);
            if (filter == null)
                return false;

            var classId = _row < _viewModel.InstanceClasses.Values.Count
                ? _viewModel.InstanceClasses.Values[_row]
                : 0;
            return classId != filter.Value;
        }
    }
}
