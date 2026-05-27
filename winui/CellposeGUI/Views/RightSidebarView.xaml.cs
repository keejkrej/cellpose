using CellposeGUI.ViewModels;
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
            view.RefreshInstanceGrid();
        }
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(MainViewModel.Ncells))
            RefreshInstanceGrid();
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
