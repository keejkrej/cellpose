import AppKit
import SwiftUI

struct ImageCanvasView: NSViewRepresentable {
    @Bindable var viewModel: MainViewModel
    @Binding var zoom: CGFloat

    func makeNSView(context: Context) -> ImageCanvasNSView {
        let view = ImageCanvasNSView()
        view.onClick = { point, flags in
            if flags.contains(.shift) {
                viewModel.beginStroke(at: Int(point.x), y: Int(point.y))
            } else {
                viewModel.removeCell(at: Int(point.x), y: Int(point.y), modifierFlags: flags)
            }
        }
        view.onDrag = { point, flags in
            if flags.contains(.shift) {
                viewModel.continueStroke(at: Int(point.x), y: Int(point.y))
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
    var zoom: CGFloat = 1
    var onClick: ((NSPoint, NSEvent.ModifierFlags) -> Void)?
    var onDrag: ((NSPoint, NSEvent.ModifierFlags) -> Void)?
    var onScroll: ((CGFloat) -> Void)?

    private var panOffset = NSPoint.zero
    private var isDragging = false

    var onKeyPress: ((NSEvent) -> Void)?

    override var acceptsFirstResponder: Bool { true }

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
    }

    private func drawMaskOverlay(masks: MaskData, in rect: NSRect, context: CGContext) {
        let width = masks.width
        let height = masks.height
        guard width > 0, height > 0 else { return }

        let pixelWidth = rect.width / CGFloat(width)
        let pixelHeight = rect.height / CGFloat(height)

        for y in 0 ..< height {
            for x in 0 ..< width {
                let label = showOutlines
                    ? (masks.outlineLabels?[y * width + x] ?? 0)
                    : masks.labels[y * width + x]
                guard label > 0 else { continue }
                if let color = masks.color(at: x, y: y) {
                    let alpha: CGFloat = label == selectedCell ? 0.75 : CGFloat(color.3)
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
        guard let point = imagePoint(from: convert(event.locationInWindow, from: nil)) else { return }
        if event.modifierFlags.contains(.shift) {
            isDragging = true
        }
        onClick?(point, event.modifierFlags)
    }

    override func mouseDragged(with event: NSEvent) {
        guard isDragging, let point = imagePoint(from: convert(event.locationInWindow, from: nil)) else { return }
        onDrag?(point, event.modifierFlags)
    }

    override func mouseUp(with event: NSEvent) {
        isDragging = false
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
