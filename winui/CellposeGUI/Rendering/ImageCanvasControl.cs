using CellposeGUI.Models;
using CellposeGUI.ViewModels;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.ApplicationModel.DataTransfer;
using Windows.Foundation;
using Windows.Storage;
using Windows.System;

namespace CellposeGUI.Rendering;

public sealed class ImageCanvasControl : Grid
{
    private readonly Canvas _canvas = new();
    private readonly Image _imageControl = new()
    {
        Stretch = Stretch.Fill,
    };

    private WriteableBitmap? _bitmap;
    private MainViewModel? _viewModel;
    private double _zoom = 1;
    private Point _panOffset;
    private bool _isDraggingStroke;

    public ImageCanvasControl()
    {
        HorizontalAlignment = HorizontalAlignment.Stretch;
        VerticalAlignment = VerticalAlignment.Stretch;
        Background = new SolidColorBrush(Microsoft.UI.Colors.Black);
        Children.Add(_canvas);
        _canvas.Children.Add(_imageControl);
        SizeChanged += (_, _) =>
        {
            if (_viewModel?.Image is { } image)
                UpdateImageLayout(image.Width, image.Height);
        };
        AllowDrop = true;
        DragOver += OnDragOver;
        Drop += OnDrop;
        PointerWheelChanged += OnPointerWheelChanged;
        PointerPressed += OnPointerPressed;
        PointerMoved += OnPointerMoved;
        PointerReleased += OnPointerReleased;
        KeyDown += OnKeyDown;
        IsTabStop = true;
    }

    public MainViewModel? ViewModel
    {
        get => _viewModel;
        set
        {
            if (_viewModel != null)
            {
                _viewModel.PropertyChanged -= OnViewModelPropertyChanged;
                _viewModel.DisplayParams.PropertyChanged -= OnDisplayParamsChanged;
            }
            _viewModel = value;
            if (_viewModel != null)
            {
                _viewModel.PropertyChanged += OnViewModelPropertyChanged;
                _viewModel.DisplayParams.PropertyChanged += OnDisplayParamsChanged;
            }
            Redraw();
        }
    }

    private void OnDisplayParamsChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(DisplayParameters.GrayLow) or nameof(DisplayParameters.GrayHigh))
            Redraw();
    }

    public double Zoom
    {
        get => _zoom;
        set
        {
            _zoom = Math.Clamp(value, 0.1, 20);
            Redraw();
            ZoomChanged?.Invoke(this, _zoom);
        }
    }

    public event EventHandler<double>? ZoomChanged;

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(MainViewModel.Image) or
            nameof(MainViewModel.Masks) or
            nameof(MainViewModel.ShowMasks) or
            nameof(MainViewModel.ShowOutlines) or
            nameof(MainViewModel.SelectedCell) or
            nameof(MainViewModel.ViewMode) or
            nameof(MainViewModel.CanvasRevision))
        {
            Redraw();
        }
    }

    private void Redraw()
    {
        if (!DispatcherQueue.HasThreadAccess)
        {
            DispatcherQueue.TryEnqueue(Redraw);
            return;
        }

        try
        {
            RedrawCore();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Canvas redraw failed: {ex}");
        }
    }

    private void RedrawCore()
    {
        var image = _viewModel?.Image;
        if (image == null)
        {
            _imageControl.Source = null;
            return;
        }

        var pixelCount = image.Width * image.Height;
        if (pixelCount <= 0)
            return;

        var (grayLow, grayHigh) = GetDisplayLevels();
        _bitmap = new WriteableBitmap(image.Width, image.Height);
        var bgra = new byte[pixelCount * 4];
        for (var i = 0; i < pixelCount; i++)
        {
            var gray = ImageData.MapGrayLevel(image.GetGrayValue(i), grayLow, grayHigh);
            var bgraOffset = i * 4;
            bgra[bgraOffset] = gray;
            bgra[bgraOffset + 1] = gray;
            bgra[bgraOffset + 2] = gray;
            bgra[bgraOffset + 3] = 255;
        }

        using (var stream = _bitmap.PixelBuffer.AsStream())
        {
            stream.Write(bgra, 0, bgra.Length);
            stream.Flush();
        }

        _bitmap.Invalidate();

        if (_viewModel?.ShowMasks == true || _viewModel?.ShowOutlines == true)
            DrawMaskOverlay(_bitmap, _viewModel);

        _imageControl.Source = _bitmap;
        UpdateImageLayout(image.Width, image.Height);
    }

    private Rect GetImageDrawRect(int imageWidth, int imageHeight)
    {
        var scale = FitScale(imageWidth, imageHeight) * _zoom;
        var drawWidth = imageWidth * scale;
        var drawHeight = imageHeight * scale;
        var originX = (ActualWidth - drawWidth) / 2 + _panOffset.X;
        var originY = (ActualHeight - drawHeight) / 2 + _panOffset.Y;
        return new Rect(originX, originY, drawWidth, drawHeight);
    }

    private void UpdateImageLayout(int imageWidth, int imageHeight)
    {
        var rect = GetImageDrawRect(imageWidth, imageHeight);
        _imageControl.Width = rect.Width;
        _imageControl.Height = rect.Height;
        Canvas.SetLeft(_imageControl, rect.X);
        Canvas.SetTop(_imageControl, rect.Y);
        _imageControl.RenderTransform = null;
    }

    protected override Size ArrangeOverride(Size finalSize)
    {
        _canvas.Width = finalSize.Width;
        _canvas.Height = finalSize.Height;
        var size = base.ArrangeOverride(finalSize);
        if (_viewModel?.Image is { } image)
            UpdateImageLayout(image.Width, image.Height);
        return size;
    }

    private void DrawMaskOverlay(WriteableBitmap bitmap, MainViewModel viewModel)
    {
        var masks = viewModel.Masks;
        if (masks == null)
            return;

        using var stream = bitmap.PixelBuffer.AsStream();
        var pixels = new byte[stream.Length];
        stream.Position = 0;
        stream.ReadExactly(pixels);

        for (var y = 0; y < masks.Height; y++)
        {
            for (var x = 0; x < masks.Width; x++)
            {
                var label = viewModel.ShowOutlines
                    ? masks.OutlineLabels?[y * masks.Width + x] ?? 0
                    : masks.LabelAt(x, y);
                if (label <= 0 || !viewModel.IsInstanceLabelVisible(label))
                    continue;

                var color = masks.ColorAt(x, y);
                if (color == null)
                    continue;

                var alpha = label == viewModel.SelectedCell ? 0.75f : color.Value.Alpha;
                var offset = (y * masks.Width + x) * 4;
                BlendPixel(pixels, offset, color.Value.R, color.Value.G, color.Value.B, alpha);
            }
        }

        stream.Position = 0;
        stream.Write(pixels);
    }

    private static void BlendPixel(byte[] pixels, int offset, byte r, byte g, byte b, float alpha)
    {
        var inv = 1f - alpha;
        pixels[offset] = (byte)(pixels[offset] * inv + r * alpha);
        pixels[offset + 1] = (byte)(pixels[offset + 1] * inv + g * alpha);
        pixels[offset + 2] = (byte)(pixels[offset + 2] * inv + b * alpha);
        pixels[offset + 3] = 255;
    }

    private (double Low, double High) GetDisplayLevels()
    {
        if (_viewModel == null)
            return (0, 255);

        if (_viewModel.ViewMode is ViewMode.Image or ViewMode.Restored)
            return (_viewModel.DisplayParams.GrayLow, _viewModel.DisplayParams.GrayHigh);

        return (0, 255);
    }

    private double FitScale(int imageWidth, int imageHeight)
    {
        if (imageWidth <= 0 || imageHeight <= 0 || ActualWidth <= 0 || ActualHeight <= 0)
            return 1;
        return Math.Min(ActualWidth / imageWidth, ActualHeight / imageHeight);
    }

    private Point? ImagePointFromPointer(Point location)
    {
        var image = _viewModel?.Image;
        if (image == null)
            return null;

        var rect = GetImageDrawRect(image.Width, image.Height);
        var localX = location.X - rect.X;
        var localY = location.Y - rect.Y;
        if (localX < 0 || localY < 0 || localX > rect.Width || localY > rect.Height)
            return null;

        var x = (int)(localX / (rect.Width / image.Width));
        var y = (int)((rect.Height - localY) / (rect.Height / image.Height));
        return new Point(x, y);
    }

    private void OnPointerPressed(object sender, PointerRoutedEventArgs e)
    {
        Focus(FocusState.Programmatic);
        var point = e.GetCurrentPoint(this);
        if (point.Properties.IsRightButtonPressed)
            return;

        if (ImagePointFromPointer(point.Position) is not Point imagePoint || _viewModel == null)
            return;

        var shiftDown = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift).HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);
        var controlDown = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Control).HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);
        var altDown = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Menu).HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);

        if (shiftDown)
        {
            _isDraggingStroke = true;
            _viewModel.BeginStroke((int)imagePoint.X, (int)imagePoint.Y);
        }
        else
        {
            _viewModel.RemoveCellAt((int)imagePoint.X, (int)imagePoint.Y, controlDown, altDown);
        }

        CapturePointer(e.Pointer);
    }

    private void OnPointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (!_isDraggingStroke || _viewModel == null)
            return;

        var point = e.GetCurrentPoint(this);
        if (ImagePointFromPointer(point.Position) is Point imagePoint)
            _viewModel.ContinueStroke((int)imagePoint.X, (int)imagePoint.Y);
    }

    private async void OnPointerReleased(object sender, PointerRoutedEventArgs e)
    {
        if (_isDraggingStroke && _viewModel != null)
        {
            _viewModel.CommitStroke();
            await _viewModel.FinishDrawingAsync();
        }

        _isDraggingStroke = false;
        ReleasePointerCapture(e.Pointer);
    }

    private void OnPointerWheelChanged(object sender, PointerRoutedEventArgs e)
    {
        var point = e.GetCurrentPoint(this);
        var controlDown = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Control).HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);
        if (controlDown)
        {
            _panOffset = new Point(_panOffset.X + point.Properties.MouseWheelDelta / 4.0, _panOffset.Y);
            Redraw();
        }
        else
        {
            var delta = point.Properties.MouseWheelDelta;
            Zoom = delta > 0 ? Zoom * 1.1 : Zoom / 1.1;
        }
    }

    private async void OnKeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (_viewModel == null)
            return;

        switch (e.Key)
        {
            case VirtualKey.Left:
            case VirtualKey.A:
                await _viewModel.NavigateSeriesAsync(-1);
                break;
            case VirtualKey.Right:
            case VirtualKey.D:
                await _viewModel.NavigateSeriesAsync(1);
                break;
            case VirtualKey.PageUp:
                _viewModel.CycleViewMode(forward: false);
                break;
            case VirtualKey.PageDown:
                _viewModel.CycleViewMode(forward: true);
                break;
            case VirtualKey.Add:
            case (VirtualKey)187:
                Zoom *= 1.1;
                break;
            case VirtualKey.Subtract:
            case (VirtualKey)189:
                Zoom /= 1.1;
                break;
        }
    }

    private void OnDragOver(object sender, DragEventArgs e)
    {
        e.AcceptedOperation = DataPackageOperation.Copy;
        e.DragUIOverride.Caption = "Load image";
    }

    private async void OnDrop(object sender, DragEventArgs e)
    {
        if (_viewModel == null || !e.DataView.Contains(StandardDataFormats.StorageItems))
            return;

        var items = await e.DataView.GetStorageItemsAsync();
        var paths = items.OfType<StorageFile>().Select(f => f.Path).ToList();
        if (paths.Count > 0)
            await _viewModel.HandleDroppedPathsAsync(paths);
    }
}
