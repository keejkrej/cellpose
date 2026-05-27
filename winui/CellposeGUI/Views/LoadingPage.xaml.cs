using Microsoft.UI.Xaml.Controls;

namespace CellposeGUI.Views;

public sealed partial class LoadingPage : Page
{
    public LoadingPage()
    {
        InitializeComponent();
    }

    protected override void OnNavigatedTo(Microsoft.UI.Xaml.Navigation.NavigationEventArgs e)
    {
        base.OnNavigatedTo(e);
        SetStatus(e.Parameter switch
        {
            string message => message,
            null => "Starting sidecar…",
            _ => e.Parameter.ToString() ?? "Starting sidecar…",
        });
    }

    public void SetStatus(string message, bool failed = false)
    {
        StatusText.Text = message;
        LoadingRing.IsActive = !failed;
    }
}
