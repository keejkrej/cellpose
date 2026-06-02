import AppKit
import SwiftUI

struct ImageCanvasView: NSViewRepresentable {
    @Bindable var viewModel: MainViewModel
    @Binding var zoom: CGFloat

    func makeNSView(context: Context) -> ImageCanvasNSView {
        let view = ImageCanvasNSView()
        view.onClick = { point in
            if viewModel.brushMode {
                // First click starts; second click completes (hover only extends preview).
                if viewModel.inStroke {
                    Task { await viewModel.completeStroke(at: Int(point.x), y: Int(point.y)) }
                } else {
                    viewModel.beginStroke(at: Int(point.x), y: Int(point.y))
                    view.liveStroke = viewModel.currentStrokePoints
                    view.inStroke = viewModel.inStroke
                    view.needsDisplay = true
                }
            } else if !viewModel.selectMode {
                viewModel.removeCell(at: Int(point.x), y: Int(point.y), modifierFlags: view.lastModifierFlags)
            }
        }
        view.onSelectClick = { point, additive in
            viewModel.selectCellAt(x: Int(point.x), y: Int(point.y), additive: additive)
        }
        view.onSelectRect = { x0, y0, x1, y1, additive in
            viewModel.selectCellsInRect(x0: x0, y0: y0, x1: x1, y1: y1, additive: additive)
        }
        view.onHover = { point in
            if viewModel.brushMode, viewModel.inStroke {
                viewModel.continueStroke(at: Int(point.x), y: Int(point.y))
                view.liveStroke = viewModel.currentStrokePoints
                view.needsDisplay = true
            }
        }
        view.onScroll = { delta in
            zoom = max(0.1, min(20, zoom * (delta > 0 ? 1.1 : 0.9)))
        }
        return view
    }

    func updateNSView(_ nsView: ImageCanvasNSView, context: Context) {
        nsView.image = viewModel.image?.cgImage
        nsView.masks = viewModel.masks
        nsView.showMasks = viewModel.showMasks
        nsView.showOutlines = viewModel.showOutlines
        nsView.selectedCells = Set(viewModel.selectedCellIndices())
        nsView.liveStroke = viewModel.currentStrokePoints
        nsView.inStroke = viewModel.inStroke
        nsView.brushMode = viewModel.brushMode
        nsView.selectMode = viewModel.selectMode
        nsView.selectionRevision = viewModel.selectionRevision
        nsView.maskBlend = viewModel.displayParams.maskBlend
        nsView.zoom = zoom
        nsView.needsDisplay = true
    }
}

final class ImageCanvasNSView: NSView {
    var image: CGImage?
    var masks: MaskData?
    var showMasks = true
    var showOutlines = false
    var selectedCells: Set<Int32> = []
    var liveStroke: [[Double]] = []
    var inStroke = false
    var brushMode = false
    var selectMode = false
    var selectionRevision = 0
    var maskBlend: Double = 0.5
    var zoom: CGFloat = 1
    var onClick: ((NSPoint) -> Void)?
    var onSelectClick: ((NSPoint, Bool) -> Void)?
    var onSelectRect: ((Int, Int, Int, Int, Bool) -> Void)?
    var onHover: ((NSPoint) -> Void)?
    var onScroll: ((CGFloat) -> Void)?

    private var panOffset = NSPoint.zero
    private var trackingArea: NSTrackingArea?
    var lastModifierFlags: NSEvent.ModifierFlags = []
    private var selectStart: NSPoint?
    private var selectDragging = false
    private var selectPreviewBounds: (x0: Int, y0: Int, x1: Int, y1: Int)?
    private let selectDragThreshold = 4
    private let brushSize = 3

    override var acceptsFirstResponder: Bool { false }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingArea {
            removeTrackingArea(trackingArea)
        }
        let options: NSTrackingArea.Options = [.mouseMoved, .activeInKeyWindow, .inVisibleRect]
        trackingArea = NSTrackingArea(rect: bounds, options: options, owner: self, userInfo: nil)
        if let trackingArea {
            addTrackingArea(trackingArea)
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        guard let context = NSGraphicsContext.current?.cgContext else { return }
        NSColor.windowBackgroundColor.setFill()
        dirtyRect.fill()

        guard let image else { return }
        let imageSize = NSSize(width: image.width, height: image.height)
        let scale = fitScale(for: imageSize) * zoom
        let drawSize = NSSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let origin = NSPoint(
            x: (bounds.width - drawSize.width) / 2 + panOffset.x,
            y: (bounds.height - drawSize.height) / 2 + panOffset.y
        )
        let rect = NSRect(origin: origin, size: drawSize)
        context.interpolationQuality = .high
        drawImageLayer(image: image, in: rect, context: context)
        if showMasks || showOutlines, let masks {
            drawMaskLayer(masks: masks, in: rect, context: context)
        }
        if inStroke, !liveStroke.isEmpty {
            drawLiveStroke(in: rect, context: context)
        }
        drawSelectionLayer(in: rect, context: context)
    }

    private func drawImageLayer(image: CGImage, in rect: NSRect, context: CGContext) {
        let imageWeight = max(0, min(1, 1 - maskBlend))
        guard imageWeight > 0 else { return }
        context.saveGState()
        context.setAlpha(imageWeight)
        context.draw(image, in: rect)
        context.restoreGState()
    }

    private func drawSelectionLayer(in rect: NSRect, context: CGContext) {
        if let preview = selectPreviewBounds {
            drawBoundsRect(preview, in: rect, context: context, r: 0, g: 1, b: 1)
        }
        for label in selectedCells.sorted() {
            guard let masks,
                  let bounds = MaskEditService.cellBounds(
                    labels: masks.labels,
                    width: masks.width,
                    height: masks.height,
                    label: label
                  ) else { continue }
            drawBoundsRect(bounds, in: rect, context: context, r: 1, g: 1, b: 0)
        }
    }

    private func drawBoundsRect(
        _ bounds: (x0: Int, y0: Int, x1: Int, y1: Int),
        in rect: NSRect,
        context: CGContext,
        r: CGFloat,
        g: CGFloat,
        b: CGFloat
    ) {
        guard let image else { return }
        let pixelWidth = rect.width / CGFloat(image.width)
        let pixelHeight = rect.height / CGFloat(image.height)
        let x = rect.minX + CGFloat(bounds.x0) * pixelWidth
        let y = rect.minY + CGFloat(image.height - bounds.y1 - 1) * pixelHeight
        let width = CGFloat(bounds.x1 - bounds.x0 + 1) * pixelWidth
        let height = CGFloat(bounds.y1 - bounds.y0 + 1) * pixelHeight
        context.setStrokeColor(red: r, green: g, blue: b, alpha: 1)
        context.setLineWidth(2)
        context.setLineDash(phase: 0, lengths: [4, 3])
        context.stroke(CGRect(x: x, y: y, width: max(width, 1), height: max(height, 1)))
        context.setLineDash(phase: 0, lengths: [])
    }

    private func drawLiveStroke(in rect: NSRect, context: CGContext) {
        guard let image else { return }

        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { return }

        let pixelWidth = rect.width / CGFloat(width)
        let pixelHeight = rect.height / CGFloat(height)
        context.setFillColor(red: 1, green: 0, blue: 1, alpha: 100.0 / 255.0)

        for (x, y) in interpolatedStrokePoints(liveStroke) {
            stampBrush(
                at: x,
                y: y,
                imageWidth: width,
                imageHeight: height,
                in: rect,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight,
                context: context
            )
        }
    }

    private func stampBrush(
        at x: Int,
        y: Int,
        imageWidth: Int,
        imageHeight: Int,
        in rect: NSRect,
        pixelWidth: CGFloat,
        pixelHeight: CGFloat,
        context: CGContext
    ) {
        let radius = max(0, brushSize / 2)
        for dy in -radius ... radius {
            for dx in -radius ... radius {
                let px = x + dx
                let py = y + dy
                guard px >= 0, py >= 0, px < imageWidth, py < imageHeight else { continue }
                let cellRect = NSRect(
                    x: rect.minX + CGFloat(px) * pixelWidth,
                    y: rect.minY + CGFloat(imageHeight - py - 1) * pixelHeight,
                    width: max(pixelWidth, 1),
                    height: max(pixelHeight, 1)
                )
                context.fill(cellRect)
            }
        }
    }

    private func interpolatedStrokePoints(_ stroke: [[Double]]) -> [(Int, Int)] {
        guard !stroke.isEmpty else { return [] }

        var points: [(Int, Int)] = []
        for index in stroke.indices {
            let y = Int(stroke[index][1])
            let x = Int(stroke[index][2])
            if index == 0 {
                points.append((x, y))
                continue
            }

            let prevY = Int(stroke[index - 1][1])
            let prevX = Int(stroke[index - 1][2])
            points.append(contentsOf: linePoints(x0: prevX, y0: prevY, x1: x, y1: y))
        }
        return points
    }

    private func linePoints(x0: Int, y0: Int, x1: Int, y1: Int) -> [(Int, Int)] {
        var points: [(Int, Int)] = []
        var x0 = x0
        var y0 = y0
        let dx = abs(x1 - x0)
        let dy = -abs(y1 - y0)
        let sx = x0 < x1 ? 1 : -1
        let sy = y0 < y1 ? 1 : -1
        var err = dx + dy

        while true {
            points.append((x0, y0))
            if x0 == x1, y0 == y1 {
                break
            }

            let e2 = 2 * err
            if e2 >= dy {
                err += dy
                x0 += sx
            }
            if e2 <= dx {
                err += dx
                y0 += sy
            }
        }

        return points
    }

    private func drawMaskLayer(
        masks: MaskData,
        in rect: NSRect,
        context: CGContext
    ) {
        let blend = maskBlend
        let width = masks.width
        let height = masks.height
        guard width > 0, height > 0 else { return }

        let maskWeight = max(0, min(1, blend))
        guard maskWeight > 0 else { return }

        let pixelWidth = rect.width / CGFloat(width)
        let pixelHeight = rect.height / CGFloat(height)

        for y in 0 ..< height {
            for x in 0 ..< width {
                let fillLabel = masks.labels[y * width + x]

                if showMasks, fillLabel > 0, let color = masks.color(at: x, y: y) {
                    let alpha = CGFloat(color.3) * CGFloat(maskWeight)
                    context.setFillColor(
                        red: CGFloat(color.0) / 255,
                        green: CGFloat(color.1) / 255,
                        blue: CGFloat(color.2) / 255,
                        alpha: alpha
                    )
                    let cellRect = NSRect(
                        x: rect.minX + CGFloat(x) * pixelWidth,
                        y: rect.minY + CGFloat(height - y - 1) * pixelHeight,
                        width: max(pixelWidth, 1),
                        height: max(pixelHeight, 1)
                    )
                    context.fill(cellRect)
                }

                guard showOutlines else { continue }
                let outlineLabel = masks.outlineLabels?[y * width + x] ?? 0
                guard outlineLabel > 0 else { continue }

                let outlineAlpha = (200.0 / 255.0) * CGFloat(maskWeight)
                context.setFillColor(red: 200.0 / 255, green: 200.0 / 255, blue: 1, alpha: outlineAlpha)
                let outlineRect = NSRect(
                    x: rect.minX + CGFloat(x) * pixelWidth,
                    y: rect.minY + CGFloat(height - y - 1) * pixelHeight,
                    width: max(pixelWidth, 1),
                    height: max(pixelHeight, 1)
                )
                context.fill(outlineRect)
            }
        }
    }

    private func fitScale(for imageSize: NSSize) -> CGFloat {
        guard imageSize.width > 0, imageSize.height > 0 else { return 1 }
        let widthScale = bounds.width / imageSize.width
        let heightScale = bounds.height / imageSize.height
        return min(widthScale, heightScale)
    }

    private func imagePoint(from location: NSPoint) -> NSPoint? {
        guard let image else { return nil }
        let imageSize = NSSize(width: image.width, height: image.height)
        let scale = fitScale(for: imageSize) * zoom
        let drawSize = NSSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let origin = NSPoint(
            x: (bounds.width - drawSize.width) / 2 + panOffset.x,
            y: (bounds.height - drawSize.height) / 2 + panOffset.y
        )
        let localX = location.x - origin.x
        let localY = location.y - origin.y
        guard localX >= 0, localY >= 0, localX <= drawSize.width, localY <= drawSize.height else {
            return nil
        }
        let x = Int(localX / (drawSize.width / CGFloat(image.width)))
        let y = Int((drawSize.height - localY) / (drawSize.height / CGFloat(image.height)))
        return NSPoint(x: x, y: y)
    }

    override func mouseDown(with event: NSEvent) {
        lastModifierFlags = event.modifierFlags
        guard let point = imagePoint(from: convert(event.locationInWindow, from: nil)) else { return }

        if selectMode {
            selectStart = point
            selectDragging = false
            selectPreviewBounds = nil
            needsDisplay = true
            return
        }

        onClick?(point)
    }

    override func mouseDragged(with event: NSEvent) {
        lastModifierFlags = event.modifierFlags
        let location = convert(event.locationInWindow, from: nil)
        updateCursor(at: location)

        if selectMode,
           let start = selectStart,
           let current = imagePoint(from: location) {
            if !selectDragging {
                if max(abs(current.x - start.x), abs(current.y - start.y)) < CGFloat(selectDragThreshold) {
                    return
                }
                selectDragging = true
            }
            selectPreviewBounds = (
                Int(min(start.x, current.x)),
                Int(min(start.y, current.y)),
                Int(max(start.x, current.x)),
                Int(max(start.y, current.y))
            )
            needsDisplay = true
            return
        }

        guard let point = imagePoint(from: location) else { return }
        onHover?(point)
    }

    override func mouseUp(with event: NSEvent) {
        lastModifierFlags = event.modifierFlags
        guard selectMode, let start = selectStart else { return }

        let additive = event.modifierFlags.contains(.shift)
        if selectDragging, let preview = selectPreviewBounds {
            onSelectRect?(preview.x0, preview.y0, preview.x1, preview.y1, additive)
        } else if let end = imagePoint(from: convert(event.locationInWindow, from: nil)) {
            onSelectClick?(end, additive)
        }

        selectStart = nil
        selectDragging = false
        selectPreviewBounds = nil
        needsDisplay = true
    }

    override func mouseMoved(with event: NSEvent) {
        let location = convert(event.locationInWindow, from: nil)
        updateCursor(at: location)
        guard let point = imagePoint(from: location) else { return }
        onHover?(point)
    }

    func updateCursor(at location: NSPoint) {
        guard image != nil else {
            NSCursor.arrow.set()
            return
        }

        if imagePoint(from: location) == nil {
            NSCursor.arrow.set()
            return
        }

        if selectMode {
            NSCursor.crosshair.set()
        } else {
            NSCursor.arrow.set()
        }
    }

    override func resetCursorRects() {
        super.resetCursorRects()
        discardCursorRects()
        addCursorRect(bounds, cursor: selectMode ? .crosshair : .arrow)
    }

    override func scrollWheel(with event: NSEvent) {
        if event.modifierFlags.contains(.command) {
            panOffset.x += event.deltaX
            panOffset.y -= event.deltaY
            needsDisplay = true
        } else {
            onScroll?(event.deltaY == 0 ? event.scrollingDeltaY : event.deltaY)
        }
    }

    override func magnify(with event: NSEvent) {
        zoom = max(0.1, min(20, zoom * (1 + event.magnification)))
        needsDisplay = true
    }
}
