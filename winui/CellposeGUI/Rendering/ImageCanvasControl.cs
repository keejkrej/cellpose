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
    private bool _isPanning;
    private Point _panStart;
    private Point _panOffsetStart;
    private Point _lastPointerPosition;
    private byte[]? _baseLayerPixels;
    private int _baseLayerRevision = -1;
    private int _baseLayerWidth;
    private int _baseLayerHeight;
    private readonly Canvas _overlayCanvas = new() { IsHitTestVisible = false };
    private Point? _selectStartImage;
    private bool _selectDragging;
    private (int X0, int Y0, int X1, int Y1)? _selectPreviewBounds;
    private const int SelectDragThreshold = 4;

    public ImageCanvasControl()
    {
        HorizontalAlignment = HorizontalAlignment.Stretch;
        VerticalAlignment = VerticalAlignment.Stretch;
        Background = new SolidColorBrush(Microsoft.UI.Colors.Black);
        Children.Add(_canvas);
        _canvas.Children.Add(_imageControl);
        _canvas.Children.Add(_overlayCanvas);
        Canvas.SetZIndex(_overlayCanvas, 1);
        Loaded += (_, _) => UpdateClip();
        SizeChanged += (_, _) =>
        {
            UpdateClip();
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
        PointerEntered += (_, e) => UpdateCursor(e.GetCurrentPoint(this).Position);
        PointerExited += (_, _) => ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.Arrow);
        ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.Arrow);
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
            var clamped = Math.Clamp(value, 0.1, 20);
            if (Math.Abs(clamped - _zoom) < 1e-9)
                return;

            if (ActualWidth > 0 && ActualHeight > 0)
            {
                ZoomAt(new Point(ActualWidth / 2, ActualHeight / 2), clamped / _zoom);
                return;
            }

            _zoom = clamped;
            UpdateViewport();
            ZoomChanged?.Invoke(this, _zoom);
        }
    }

    public event EventHandler<double>? ZoomChanged;

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(MainViewModel.Image))
        {
            _zoom = 1;
            _panOffset = default;
            ZoomChanged?.Invoke(this, _zoom);
        }

        if (e.PropertyName is nameof(MainViewModel.Image) or
            nameof(MainViewModel.Masks) or
            nameof(MainViewModel.ShowMasks) or
            nameof(MainViewModel.ShowOutlines) or
            nameof(MainViewModel.SelectedCell) or
            nameof(MainViewModel.SelectionRevision) or
            nameof(MainViewModel.ViewMode) or
            nameof(MainViewModel.CanvasRevision))
        {
            _baseLayerRevision = -1;
            Redraw();
            return;
        }

        if (e.PropertyName is nameof(MainViewModel.StrokeRevision) or nameof(MainViewModel.InStroke))
            Redraw();

        if (e.PropertyName is nameof(MainViewModel.SelectionRevision))
            UpdateSelectionOverlay();

        if (e.PropertyName is nameof(MainViewModel.BrushMode) or nameof(MainViewModel.SelectMode))
            UpdateCursor(_lastPointerPosition);
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
            _baseLayerPixels = null;
            _baseLayerRevision = -1;
            return;
        }

        var pixelCount = image.Width * image.Height;
        if (pixelCount <= 0)
            return;

        if (_baseLayerRevision != _viewModel!.CanvasRevision ||
            _baseLayerPixels == null ||
            _baseLayerWidth != image.Width ||
            _baseLayerHeight != image.Height)
        {
            _baseLayerPixels = BuildBaseLayerPixels(image, _viewModel);
            _baseLayerRevision = _viewModel.CanvasRevision;
            _baseLayerWidth = image.Width;
            _baseLayerHeight = image.Height;
        }

        var pixels = (byte[])_baseLayerPixels.Clone();
        if (_viewModel.InStroke && _viewModel.CurrentStroke.Count > 0)
            DrawLiveStrokeOverlay(pixels, image.Width, image.Height, _viewModel.CurrentStroke);

        _bitmap = new WriteableBitmap(image.Width, image.Height);
        using (var stream = _bitmap.PixelBuffer.AsStream())
        {
            stream.Write(pixels, 0, pixels.Length);
            stream.Flush();
        }

        _bitmap.Invalidate();
        _imageControl.Source = _bitmap;
        UpdateViewport();
        UpdateSelectionOverlay();
    }

    private byte[] BuildBaseLayerPixels(ImageData image, MainViewModel viewModel)
    {
        var pixelCount = image.Width * image.Height;
        var (grayLow, grayHigh) = GetDisplayLevels();
        var pixels = new byte[pixelCount * 4];
        for (var i = 0; i < pixelCount; i++)
        {
            var gray = ImageData.MapGrayLevel(image.GetGrayValue(i), grayLow, grayHigh);
            var bgraOffset = i * 4;
            pixels[bgraOffset] = gray;
            pixels[bgraOffset + 1] = gray;
            pixels[bgraOffset + 2] = gray;
            pixels[bgraOffset + 3] = 255;
        }

        if (viewModel.ShowMasks || viewModel.ShowOutlines)
            DrawMaskOverlay(pixels, viewModel);

        return pixels;
    }

    private void UpdateClip()
    {
        if (ActualWidth <= 0 || ActualHeight <= 0)
        {
            Clip = null;
            return;
        }

        Clip = new RectangleGeometry
        {
            Rect = new Rect(0, 0, ActualWidth, ActualHeight),
        };
    }

    private void UpdateViewport()
    {
        if (_viewModel?.Image is not { } image)
            return;

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
        UpdateSelectionOverlay();
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

    private void DrawMaskOverlay(byte[] pixels, MainViewModel viewModel)
    {
        var masks = viewModel.Masks;
        if (masks == null)
            return;

        const byte outlineR = 200;
        const byte outlineG = 200;
        const byte outlineB = 255;
        const float outlineAlpha = 200f / 255f;

        for (var y = 0; y < masks.Height; y++)
        {
            for (var x = 0; x < masks.Width; x++)
            {
                var offset = (y * masks.Width + x) * 4;
                var fillLabel = masks.LabelAt(x, y);

                if (viewModel.ShowMasks &&
                    fillLabel > 0 &&
                    viewModel.IsInstanceLabelVisible(fillLabel) &&
                    masks.ColorAt(x, y) is { } fillColor)
                {
                    var alpha = viewModel.IsCellSelected(fillLabel) ? 0.75f : fillColor.Alpha;
                    BlendPixel(pixels, offset, fillColor.R, fillColor.G, fillColor.B, alpha);
                }

                if (!viewModel.ShowOutlines)
                    continue;

                var outlineLabel = masks.OutlineLabels?[y * masks.Width + x] ?? 0;
                if (outlineLabel <= 0 || !viewModel.IsInstanceLabelVisible(outlineLabel))
                    continue;

                var outlinePixelAlpha = viewModel.IsCellSelected(outlineLabel) ? 0.85f : outlineAlpha;
                BlendPixel(pixels, offset, outlineR, outlineG, outlineB, outlinePixelAlpha);
            }
        }
    }

    private static void DrawLiveStrokeOverlay(
        byte[] pixels,
        int width,
        int height,
        IReadOnlyList<double[]> stroke)
    {
        if (stroke.Count == 0)
            return;

        const byte strokeR = 255;
        const byte strokeG = 0;
        const byte strokeB = 255;
        const float strokeAlpha = 100f / 255f;
        const int brushSize = 1;

        var points = new List<(int X, int Y)>();
        for (var i = 0; i < stroke.Count; i++)
        {
            var y = (int)stroke[i][1];
            var x = (int)stroke[i][2];
            if (i == 0)
            {
                points.Add((x, y));
                continue;
            }

            var prevY = (int)stroke[i - 1][1];
            var prevX = (int)stroke[i - 1][2];
            foreach (var point in LinePoints(prevX, prevY, x, y))
                points.Add(point);
        }

        foreach (var (x, y) in points)
            StampBrush(pixels, width, height, x, y, brushSize, strokeR, strokeG, strokeB, strokeAlpha);
    }

    private static void StampBrush(
        byte[] pixels,
        int width,
        int height,
        int centerX,
        int centerY,
        int brushSize,
        byte r,
        byte g,
        byte b,
        float alpha)
    {
        var radius = Math.Max(0, brushSize / 2);
        for (var dy = -radius; dy <= radius; dy++)
        {
            for (var dx = -radius; dx <= radius; dx++)
            {
                var x = centerX + dx;
                var y = centerY + dy;
                if (x < 0 || y < 0 || x >= width || y >= height)
                    continue;

                BlendPixel(pixels, (y * width + x) * 4, r, g, b, alpha);
            }
        }
    }

    private static IEnumerable<(int X, int Y)> LinePoints(int x0, int y0, int x1, int y1)
    {
        var dx = Math.Abs(x1 - x0);
        var dy = -Math.Abs(y1 - y0);
        var sx = x0 < x1 ? 1 : -1;
        var sy = y0 < y1 ? 1 : -1;
        var err = dx + dy;

        while (true)
        {
            yield return (x0, y0);
            if (x0 == x1 && y0 == y1)
                break;

            var e2 = 2 * err;
            if (e2 >= dy)
            {
                err += dy;
                x0 += sx;
            }

            if (e2 <= dx)
            {
                err += dx;
                y0 += sy;
            }
        }
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

    private void ZoomAt(Point screenPoint, double factor)
    {
        var image = _viewModel?.Image;
        if (image == null || ActualWidth <= 0 || ActualHeight <= 0)
            return;

        var newZoom = Math.Clamp(_zoom * factor, 0.1, 20);
        if (Math.Abs(newZoom - _zoom) < 1e-9)
            return;

        var fit = FitScale(image.Width, image.Height);
        var oldRect = GetImageDrawRect(image.Width, image.Height);
        var anchor = screenPoint;
        var u = (anchor.X - oldRect.X) / oldRect.Width;
        var v = (anchor.Y - oldRect.Y) / oldRect.Height;
        if (u < 0 || u > 1 || v < 0 || v > 1 || oldRect.Width <= 0 || oldRect.Height <= 0)
        {
            anchor = new Point(ActualWidth / 2, ActualHeight / 2);
            u = (anchor.X - oldRect.X) / oldRect.Width;
            v = (anchor.Y - oldRect.Y) / oldRect.Height;
            if (u < 0 || u > 1 || v < 0 || v > 1)
            {
                u = 0.5;
                v = 0.5;
            }
        }

        _zoom = newZoom;
        var newScale = fit * _zoom;
        var newDrawWidth = image.Width * newScale;
        var newDrawHeight = image.Height * newScale;
        _panOffset = new Point(
            anchor.X - (ActualWidth - newDrawWidth) / 2 - u * newDrawWidth,
            anchor.Y - (ActualHeight - newDrawHeight) / 2 - v * newDrawHeight);
        UpdateViewport();
        ZoomChanged?.Invoke(this, _zoom);
    }

    private void UpdateCursor(Point position)
    {
        if (_isPanning)
        {
            ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.SizeAll);
            return;
        }

        if (_viewModel?.SelectMode == true && ImagePointFromPointer(position) != null)
        {
            ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.Cross);
            return;
        }

        ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.Arrow);
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
        var y = (int)(localY / (rect.Height / image.Height));
        x = Math.Clamp(x, 0, image.Width - 1);
        y = Math.Clamp(y, 0, image.Height - 1);
        return new Point(x, y);
    }

    private void OnPointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var point = e.GetCurrentPoint(this);

        if (point.Properties.IsMiddleButtonPressed ||
            (point.Properties.IsRightButtonPressed && _viewModel?.BrushMode != true))
        {
            _isPanning = true;
            _panStart = point.Position;
            _panOffsetStart = _panOffset;
            UpdateCursor(point.Position);
            CapturePointer(e.Pointer);
            return;
        }

        if (ImagePointFromPointer(point.Position) is not Point imagePoint || _viewModel == null)
            return;

        if (_viewModel.BrushMode && point.Properties.IsLeftButtonPressed)
        {
            // First click starts; second click completes (hover/move only extends preview).
            if (_viewModel.InStroke)
            {
                _ = _viewModel.CompleteStrokeAsync((int)imagePoint.X, (int)imagePoint.Y);
            }
            else
            {
                _viewModel.BeginStroke((int)imagePoint.X, (int)imagePoint.Y);
            }

            return;
        }

        if (_viewModel.SelectMode && point.Properties.IsLeftButtonPressed)
        {
            _selectStartImage = imagePoint;
            _selectDragging = false;
            _selectPreviewBounds = null;
            UpdateSelectionOverlay();
            CapturePointer(e.Pointer);
            return;
        }

        var controlDown = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Control).HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);
        var altDown = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Menu).HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);

        _viewModel.RemoveCellAt((int)imagePoint.X, (int)imagePoint.Y, controlDown, altDown);
        CapturePointer(e.Pointer);
    }

    private void OnPointerMoved(object sender, PointerRoutedEventArgs e)
    {
        var point = e.GetCurrentPoint(this);
        _lastPointerPosition = point.Position;
        UpdateCursor(point.Position);

        if (_isPanning)
        {
            _panOffset = new Point(
                _panOffsetStart.X + (point.Position.X - _panStart.X),
                _panOffsetStart.Y + (point.Position.Y - _panStart.Y));
            UpdateViewport();
            return;
        }

        if (_viewModel?.BrushMode == true && _viewModel.InStroke &&
            ImagePointFromPointer(point.Position) is Point hoverPoint)
        {
            _viewModel.ContinueStroke((int)hoverPoint.X, (int)hoverPoint.Y);
            return;
        }

        if (_viewModel?.SelectMode == true &&
            _selectStartImage is Point start &&
            point.Properties.IsLeftButtonPressed &&
            ImagePointFromPointer(point.Position) is Point current)
        {
            if (!_selectDragging)
            {
                if (Math.Max(Math.Abs(current.X - start.X), Math.Abs(current.Y - start.Y)) < SelectDragThreshold)
                    return;

                _selectDragging = true;
            }

            _selectPreviewBounds = (
                (int)Math.Min(start.X, current.X),
                (int)Math.Min(start.Y, current.Y),
                (int)Math.Max(start.X, current.X),
                (int)Math.Max(start.Y, current.Y));
            UpdateSelectionOverlay();
        }
    }

    private void OnPointerReleased(object sender, PointerRoutedEventArgs e)
    {
        if (_isPanning)
        {
            _isPanning = false;
            UpdateCursor(e.GetCurrentPoint(this).Position);
            ReleasePointerCapture(e.Pointer);
            return;
        }

        if (_viewModel?.SelectMode == true && _selectStartImage is Point start)
        {
            var shiftDown = InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift)
                .HasFlag(Windows.UI.Core.CoreVirtualKeyStates.Down);
            if (_selectDragging && _selectPreviewBounds is { } bounds)
            {
                _viewModel.SelectCellsInRect(bounds.X0, bounds.Y0, bounds.X1, bounds.Y1, shiftDown);
            }
            else if (ImagePointFromPointer(e.GetCurrentPoint(this).Position) is Point end)
            {
                _viewModel.SelectCellAt((int)end.X, (int)end.Y, shiftDown);
            }

            _selectStartImage = null;
            _selectDragging = false;
            _selectPreviewBounds = null;
            UpdateSelectionOverlay();
            ReleasePointerCapture(e.Pointer);
            return;
        }

        ReleasePointerCapture(e.Pointer);
    }

    private void UpdateSelectionOverlay()
    {
        _overlayCanvas.Children.Clear();
        var image = _viewModel?.Image;
        var masks = _viewModel?.Masks;
        if (image == null || masks == null || ActualWidth <= 0 || ActualHeight <= 0)
            return;

        var rect = GetImageDrawRect(image.Width, image.Height);
        _overlayCanvas.Width = rect.Width;
        _overlayCanvas.Height = rect.Height;
        Canvas.SetLeft(_overlayCanvas, rect.X);
        Canvas.SetTop(_overlayCanvas, rect.Y);

        if (_selectPreviewBounds is { } preview)
            AddOverlayRect(preview.X0, preview.Y0, preview.X1, preview.Y1, image.Width, image.Height, 0, 255, 255);

        foreach (var label in _viewModel!.SelectedCellIndices())
        {
            if (MaskEditService.CellBounds(masks.Labels, masks.Width, masks.Height, label) is not { } bounds)
                continue;
            AddOverlayRect(bounds.X0, bounds.Y0, bounds.X1, bounds.Y1, image.Width, image.Height, 255, 255, 0);
        }
    }

    private void AddOverlayRect(
        int x0,
        int y0,
        int x1,
        int y1,
        int imageWidth,
        int imageHeight,
        byte r,
        byte g,
        byte b)
    {
        if (_overlayCanvas.Width <= 0 || _overlayCanvas.Height <= 0)
            return;

        var left = (double)x0 / imageWidth * _overlayCanvas.Width;
        var top = (double)y0 / imageHeight * _overlayCanvas.Height;
        var right = (double)(x1 + 1) / imageWidth * _overlayCanvas.Width;
        var bottom = (double)(y1 + 1) / imageHeight * _overlayCanvas.Height;
        var rectangle = new Microsoft.UI.Xaml.Shapes.Rectangle
        {
            Width = Math.Max(1, right - left),
            Height = Math.Max(1, bottom - top),
            Stroke = new SolidColorBrush(Windows.UI.Color.FromArgb(255, r, g, b)),
            StrokeThickness = 2,
            StrokeDashArray = new DoubleCollection { 4, 3 },
            Fill = null,
        };
        Canvas.SetLeft(rectangle, left);
        Canvas.SetTop(rectangle, top);
        _overlayCanvas.Children.Add(rectangle);
    }

    private void OnPointerWheelChanged(object sender, PointerRoutedEventArgs e)
    {
        var point = e.GetCurrentPoint(this);
        var delta = point.Properties.MouseWheelDelta;
        var factor = delta > 0 ? 1.1 : 1.0 / 1.1;
        ZoomAt(point.Position, factor);
        e.Handled = true;
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
