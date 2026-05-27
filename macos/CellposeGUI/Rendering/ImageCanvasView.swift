import AppKit
import SwiftUI

struct ImageCanvasView: NSViewRepresentable {
    @Bindable var viewModel: MainViewModel
    @Binding var zoom: CGFloat

    func makeNSView(context: Context) -> ImageCanvasNSView {
        let view = ImageCanvasNSView()
        view.onClick = { point in
            if viewModel.brushMode {
                if viewModel.inStroke {
                    Task { await viewModel.completeStroke(at: Int(point.x), y: Int(point.y)) }
                } else {
                    viewModel.beginStroke(at: Int(point.x), y: Int(point.y))
                    view.liveStroke = viewModel.currentStrokePoints
                    view.inStroke = viewModel.inStroke
                    view.needsDisplay = true
                }
            } else {
                viewModel.removeCell(at: Int(point.x), y: Int(point.y), modifierFlags: view.lastModifierFlags)
            }
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
        view.onKeyPress = { event in
            switch event.keyCode {
            case 123, 0: // left arrow or A
                Task { await viewModel.navigateSeries(delta: -1) }
            case 124, 2: // right arrow or D
                Task { await viewModel.navigateSeries(delta: 1) }
            case 116: // page up
                viewModel.cycleViewMode(forward: false)
            case 121: // page down
                viewModel.cycleViewMode(forward: true)
            default:
                break
            }
        }
        return view
    }

    func updateNSView(_ nsView: ImageCanvasNSView, context: Context) {
        nsView.image = viewModel.image?.cgImage
        nsView.masks = viewModel.masks
        nsView.showMasks = viewModel.showMasks
        nsView.showOutlines = viewModel.showOutlines
        nsView.selectedCell = viewModel.selectedCell
        nsView.liveStroke = viewModel.currentStrokePoints
        nsView.inStroke = viewModel.inStroke
        nsView.zoom = zoom
        nsView.needsDisplay = true
    }
}

final class ImageCanvasNSView: NSView {
    var image: CGImage?
    var masks: MaskData?
    var showMasks = true
    var showOutlines = false
    var selectedCell: Int32 = 0
    var liveStroke: [[Double]] = []
    var inStroke = false
    var zoom: CGFloat = 1
    var onClick: ((NSPoint) -> Void)?
    var onHover: ((NSPoint) -> Void)?
    var onScroll: ((CGFloat) -> Void)?

    private var panOffset = NSPoint.zero
    private var trackingArea: NSTrackingArea?
    var lastModifierFlags: NSEvent.ModifierFlags = []

    var onKeyPress: ((NSEvent) -> Void)?

    override var acceptsFirstResponder: Bool { true }

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
        context.draw(image, in: rect)

        if showMasks || showOutlines, let masks {
            drawMaskOverlay(masks: masks, in: rect, context: context)
        }

        if inStroke, !liveStroke.isEmpty {
            drawLiveStroke(in: rect, context: context)
        }
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
            let cellRect = NSRect(
                x: rect.minX + CGFloat(x) * pixelWidth,
                y: rect.minY + CGFloat(height - y - 1) * pixelHeight,
                width: max(pixelWidth, 1),
                height: max(pixelHeight, 1)
            )
            context.fill(cellRect)
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

    private func drawMaskOverlay(masks: MaskData, in rect: NSRect, context: CGContext) {
        let width = masks.width
        let height = masks.height
        guard width > 0, height > 0 else { return }

        let pixelWidth = rect.width / CGFloat(width)
        let pixelHeight = rect.height / CGFloat(height)

        for y in 0 ..< height {
            for x in 0 ..< width {
                let fillLabel = masks.labels[y * width + x]

                if showMasks, fillLabel > 0, let color = masks.color(at: x, y: y) {
                    let alpha: CGFloat = fillLabel == selectedCell ? 0.75 : CGFloat(color.3)
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

                let outlineAlpha: CGFloat = outlineLabel == selectedCell ? 0.85 : 200.0 / 255.0
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
        onClick?(point)
    }

    override func mouseMoved(with event: NSEvent) {
        guard let point = imagePoint(from: convert(event.locationInWindow, from: nil)) else { return }
        onHover?(point)
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

    override func keyDown(with event: NSEvent) {
        onKeyPress?(event)
        switch event.charactersIgnoringModifiers {
        case "+", "=":
            zoom = min(20, zoom * 1.1)
            needsDisplay = true
        case "-":
            zoom = max(0.1, zoom / 1.1)
            needsDisplay = true
        default:
            super.keyDown(with: event)
        }
    }
}
