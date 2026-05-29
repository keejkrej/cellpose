import CoreGraphics
import CellposeGUICore
import Foundation
import ImageIO

final class ImageLoaderService {
    func load(path: String) throws -> ImageData {
        let ext = URL(fileURLWithPath: path).pathExtension.lowercased()
        switch ext {
        case "tif", "tiff":
            return try loadTiff(path: path)
        case "png", "jpg", "jpeg", "gif", "bmp":
            return try loadRaster(path: path)
        default:
            throw SidecarError.serverError("Unsupported image format: .\(ext)")
        }
    }

    private func loadRaster(path: String) throws -> ImageData {
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw SidecarError.serverError("Failed to open image: \(path)")
        }
        return try imageData(from: cgImage)
    }

    private func loadTiff(path: String) throws -> ImageData {
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw SidecarError.serverError("Failed to open TIFF: \(path)")
        }
        return try imageData(from: cgImage)
    }

    private func imageData(from cgImage: CGImage) throws -> ImageData {
        let width = cgImage.width
        let height = cgImage.height
        let channels = min(cgImage.bitsPerPixel / max(cgImage.bitsPerComponent, 1), 3)
        let bytesPerRow = width * max(channels, 1)
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 3,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw SidecarError.serverError("Failed to create image context")
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ImageData(width: width, height: height, channels: 3, pixels: pixels)
    }
}
