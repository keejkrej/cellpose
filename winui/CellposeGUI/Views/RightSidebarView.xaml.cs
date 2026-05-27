using CellposeGUI.ViewModels;
using CommunityToolkit.WinUI.UI.Controls;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace CellposeGUI.Views;

public sealed partial class RightSidebarView : UserControl
{
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
            view.RefreshInstanceGrid();
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
            case nameof(MainViewModel.InstanceRowsRevision):
                RefreshInstanceGrid();
                break;
        }
    }

    private void RefreshSegmentationGrid()
    {
        if (ViewModel == null)
            return;

        SegmentationGrid.ItemsSource = SegmentationParamRowViewModel.CreateRows(ViewModel);
    }

    private void InstanceGrid_LoadingRow(object sender, DataGridRowEventArgs e)
    {
        if (e.Row.DataContext is InstanceRowViewModel row)
            e.Row.Visibility = row.IsHiddenByFilter ? Visibility.Collapsed : Visibility.Visible;
    }

    private void RefreshInstanceGrid()
    {
        if (ViewModel == null)
            return;

        InstanceGrid.ItemsSource = Enumerable.Range(0, ViewModel.Ncells)
            .Select(row => new InstanceRowViewModel(ViewModel, row))
            .ToList();
    }
}
