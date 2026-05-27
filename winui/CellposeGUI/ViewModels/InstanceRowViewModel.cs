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

    public double ClassId
    {
        get => _row < _viewModel.InstanceClasses.Values.Count ? _viewModel.InstanceClasses.Values[_row] : 0;
        set => _viewModel.InstanceClasses.SetClass(_row, (int)value);
    }
}
