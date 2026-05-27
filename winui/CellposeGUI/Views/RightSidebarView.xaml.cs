using CellposeGUI.ViewModels;
using CommunityToolkit.WinUI.UI.Controls;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CellposeGUI.Views;

public sealed partial class RightSidebarView : UserControl
{
    private bool _syncingLabelsSelection;

    public RightSidebarView()
    {
        InitializeComponent();
    }

    public MainViewModel? ViewModel
    {
        get => (MainViewModel?)GetValue(ViewModelProperty);
        set => SetValue(ViewModelProperty, value);
    }

    public static readonly DependencyProperty ViewModelProperty =
        DependencyProperty.Register(
            nameof(ViewModel),
            typeof(MainViewModel),
            typeof(RightSidebarView),
            new PropertyMetadata(null, OnViewModelChanged));

    private static void OnViewModelChanged(DependencyObject sender, DependencyPropertyChangedEventArgs args)
    {
        var view = (RightSidebarView)sender;
        if (args.OldValue is MainViewModel oldViewModel)
            oldViewModel.PropertyChanged -= view.OnViewModelPropertyChanged;
        if (args.NewValue is MainViewModel newViewModel)
        {
            newViewModel.PropertyChanged += view.OnViewModelPropertyChanged;
            view.RefreshSegmentationGrid();
            view.RefreshLabelsGrid();
        }
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        switch (e.PropertyName)
        {
            case nameof(MainViewModel.Models):
                RefreshSegmentationGrid();
                break;
            case nameof(MainViewModel.Ncells):
            case nameof(MainViewModel.ClassFilterText):
            case nameof(MainViewModel.LabelsRowsRevision):
                RefreshLabelsGrid();
                break;
            case nameof(MainViewModel.SelectionRevision):
                SyncLabelsGridSelection();
                break;
        }
    }

    private void RefreshSegmentationGrid()
    {
        if (ViewModel == null)
            return;

        SegmentationGrid.ItemsSource = SegmentationParamRowViewModel.CreateRows(ViewModel);
    }

    private void LabelsGrid_LoadingRow(object sender, DataGridRowEventArgs e)
    {
        if (e.Row.DataContext is LabelRowViewModel row)
            e.Row.Visibility = row.IsHiddenByFilter ? Visibility.Collapsed : Visibility.Visible;
    }

    private void RefreshLabelsGrid()
    {
        if (ViewModel == null)
            return;

        LabelsGrid.ItemsSource = Enumerable.Range(0, ViewModel.Ncells)
            .Select(row => new LabelRowViewModel(ViewModel, row))
            .ToList();
        SyncLabelsGridSelection();
    }

    private void LabelsGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_syncingLabelsSelection || ViewModel == null)
            return;

        var cells = LabelsGrid.SelectedItems
            .OfType<LabelRowViewModel>()
            .Select(row => row.Row + 1)
            .Distinct()
            .OrderBy(v => v)
            .ToList();
        ViewModel.SetCellSelection(cells);
    }

    private void SyncLabelsGridSelection()
    {
        if (ViewModel == null || LabelsGrid.ItemsSource == null)
            return;

        var selectedRows = ViewModel.SelectedCellIndices()
            .Select(idx => idx - 1)
            .Where(row => row >= 0 && row < ViewModel.Ncells)
            .ToHashSet();

        _syncingLabelsSelection = true;
        LabelsGrid.SelectedItems.Clear();
        if (LabelsGrid.ItemsSource is IEnumerable<LabelRowViewModel> rows)
        {
            foreach (var row in rows)
            {
                if (selectedRows.Contains(row.Row))
                    LabelsGrid.SelectedItems.Add(row);
            }
        }

        _syncingLabelsSelection = false;
    }
}
