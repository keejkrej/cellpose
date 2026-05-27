using CellposeGUI.Services;
using CellposeGUI.ViewModels;
using CellposeGUI.Views;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Navigation;

namespace CellposeGUI;

public partial class App : Application
{
    private Window? _window;
    private SidecarProcessManager? _sidecarManager;

    public static Window CurrentWindow =>
        (Current as App)?._window ?? throw new InvalidOperationException("Application window is not initialized.");

    public App()
    {
        InitializeComponent();
        UnhandledException += OnUnhandledException;
    }

    private void OnUnhandledException(object sender, Microsoft.UI.Xaml.UnhandledExceptionEventArgs e)
    {
        System.Diagnostics.Debug.WriteLine($"Unhandled UI exception: {e.Exception}");
        e.Handled = true;
    }

    protected override void OnLaunched(LaunchActivatedEventArgs e)
    {
        _window = new Window
        {
            Title = "Cellpose",
        };
        _window.ExtendsContentIntoTitleBar = false;

        var rootFrame = new Frame();
        rootFrame.NavigationFailed += OnNavigationFailed;
        _window.Content = rootFrame;

        rootFrame.Navigate(typeof(LoadingPage), "Starting sidecar…");
        _window.Activate();
        _ = InitializeAsync(rootFrame);
    }

    private async Task InitializeAsync(Frame rootFrame)
    {
        var dispatcher = rootFrame.DispatcherQueue;

        try
        {
            _sidecarManager = new SidecarProcessManager();
            _sidecarManager.StatusChanged += message =>
            {
                dispatcher.TryEnqueue(() => SetLoadingStatus(rootFrame, message));
            };

            await _sidecarManager.StartIfNeededAsync().ConfigureAwait(false);

            if (!_sidecarManager.IsReady)
            {
                dispatcher.TryEnqueue(() => SetLoadingStatus(rootFrame, _sidecarManager.StatusMessage, failed: true));
                return;
            }

            var engine = new SidecarSegmentationEngine(new SidecarClient(_sidecarManager.BaseUri));
            var viewModel = new MainViewModel(engine, dispatcher);
            await viewModel.RefreshModelsAsync().ConfigureAwait(false);

            dispatcher.TryEnqueue(() =>
            {
                if (!rootFrame.Navigate(typeof(MainPage), viewModel))
                {
                    SetLoadingStatus(rootFrame, "Failed to open main window.", failed: true);
                    return;
                }

                if (_window?.AppWindow is { } appWindow)
                {
                    appWindow.Resize(new Windows.Graphics.SizeInt32(1280, 800));
                    _window.Title = viewModel.WindowTitle;
                }
            });
        }
        catch (Exception ex)
        {
            dispatcher.TryEnqueue(() =>
                SetLoadingStatus(rootFrame, $"Startup failed: {ex.Message}", failed: true));
        }
    }

    private static void SetLoadingStatus(Frame rootFrame, string message, bool failed = false)
    {
        if (rootFrame.Content is LoadingPage loadingPage)
        {
            loadingPage.SetStatus(message, failed);
            return;
        }

        rootFrame.Navigate(typeof(LoadingPage), message);
        if (rootFrame.Content is LoadingPage page)
            page.SetStatus(message, failed);
    }

    private static void OnNavigationFailed(object sender, NavigationFailedEventArgs e) =>
        throw new Exception($"Failed to load page {e.SourcePageType.FullName}");
}
