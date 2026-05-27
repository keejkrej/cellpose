import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

final class ExportService {
    func exportMasks(masks: MaskData, path: String, format: String) throws {
        var outputPath = path
        if format.lowercased() == "tif" || format.lowercased() == "tiff" {
            if !outputPath.lowercased().hasSuffix(".tif") && !outputPath.lowercased().hasSuffix(".tiff") {
                outputPath = URL(fileURLWithPath: outputPath).deletingPathExtension().path + ".tif"
            }
            try writeUInt16Tiff(path: outputPath, labels: masks.labels, width: masks.width, height: masks.height)
            return
        }

        if !outputPath.lowercased().hasSuffix(".png") {
            outputPath = URL(fileURLWithPath: outputPath).deletingPathExtension().path + ".png"
        }
        try writeMaskPng(path: outputPath, labels: masks.labels, width: masks.width, height: masks.height)
    }

    func exportOutlines(masks: MaskData, path: String) throws {
        var outputPath = path
        if !outputPath.lowercased().hasSuffix(".txt") {
            outputPath = URL(fileURLWithPath: outputPath).deletingPathExtension().path + ".txt"
        }
        let outlines = extractOutlines(masks: masks)
        var lines: [String] = []
        for (label, points) in outlines {
            var line = "\(label)"
            for (x, y) in points {
                line += " \(x) \(y)"
            }
            lines.append(line)
        }
        try lines.joined(separator: "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
    }

    func exportFlows(flows: [ArrayPayload], pathBase: String) throws {
        guard flows.count >= 4 else {
            throw SidecarError.serverError("Flows not available for export")
        }
        let base = URL(fileURLWithPath: pathBase).deletingPathExtension().path
        try writeFloatTiff(path: base + "_cp_flows.tif", data: try ArrayCodec.decode(flows[0]), shape: flows[0].shape)
        try writeByteTiff(path: base + "_cp_cellprob.tif", data: try ArrayCodec.decode(flows[1]), shape: flows[1].shape)
    }

    func exportROIs(masks: MaskData, path: String) throws {
        var outputPath = path
        if !outputPath.lowercased().hasSuffix(".zip") {
            outputPath = URL(fileURLWithPath: outputPath).deletingPathExtension().path + ".zip"
        }
        let staging = FileManager.default.temporaryDirectory
            .appendingPathComponent("cellpose-rois-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }

        for (label, points) in extractOutlines(masks: masks) {
            let roiURL = staging.appendingPathComponent(String(format: "%04d.roi", label))
            let data = try writeImageJRoi(points: points, label: label)
            try data.write(to: roiURL)
        }
        if FileManager.default.fileExists(atPath: outputPath) {
            try FileManager.default.removeItem(atPath: outputPath)
        }
        try ZipArchiveHelper.writeArchive(at: URL(fileURLWithPath: outputPath), from: staging)
    }

    private func extractOutlines(masks: MaskData) -> [(Int, [(Int, Int)])] {
        let maxLabel = Int(masks.labels.max() ?? 0)
        var result: [(Int, [(Int, Int)])] = []
        for label in 1 ... maxLabel {
            var points: [(Int, Int)] = []
            for y in 0 ..< masks.height {
                for x in 0 ..< masks.width where masks.outlineLabels?[y * masks.width + x] == Int32(label) {
                    points.append((x, y))
                }
            }
            if !points.isEmpty {
                result.append((label, points))
            }
        }
        return result
    }

    private func writeMaskPng(path: String, labels: [Int32], width: Int, height: Int) throws {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let value = UInt8(clamping: Int(labels[y * width + x]))
                let offset = (y * width + x) * 4
                pixels[offset] = value
                pixels[offset + 1] = value
                pixels[offset + 2] = value
                pixels[offset + 3] = 255
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else {
            throw SidecarError.serverError("Failed to create mask PNG")
        }
        guard let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw SidecarError.serverError("Failed to write PNG: \(path)")
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw SidecarError.serverError("Failed to finalize PNG: \(path)")
        }
    }

    private func writeUInt16Tiff(path: String, labels: [Int32], width: Int, height: Int) throws {
        var raw = Data(capacity: width * height * MemoryLayout<UInt16>.size)
        for label in labels {
            var value = UInt16(clamping: Int(label)).littleEndian
            raw.append(Data(bytes: &value, count: MemoryLayout<UInt16>.size))
        }
        try writeTiff(path: path, raw: raw, width: width, height: height, bitsPerSample: 16, sampleFormat: nil)
    }

    private func writeByteTiff(path: String, data: Data, shape: [Int]) throws {
        let (width, height) = resolve2DShape(shape)
        try writeTiff(path: path, raw: data, width: width, height: height, bitsPerSample: 8, sampleFormat: nil)
    }

    private func writeFloatTiff(path: String, data: Data, shape: [Int]) throws {
        let (width, height) = resolve2DShape(shape)
        try writeTiff(path: path, raw: data, width: width, height: height, bitsPerSample: 32, sampleFormat: 3)
    }

    private func writeTiff(
        path: String,
        raw: Data,
        width: Int,
        height: Int,
        bitsPerSample: Int,
        sampleFormat: Int?
    ) throws {
        let bytesPerSample = bitsPerSample / 8
        let scanline = width * bytesPerSample
        var imageData = Data(capacity: scanline * height)
        for y in 0 ..< height {
            imageData.append(raw.subdata(in: y * scanline ..< (y + 1) * scanline))
        }

        var metadata: [CFString: Any] = [
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFPhotometricInterpretation: 1,
                kCGImagePropertyTIFFCompression: 1,
            ],
        ]
        if let sampleFormat {
            metadata[kCGImagePropertyTIFFDictionary] = [
                kCGImagePropertyTIFFPhotometricInterpretation: 1,
                kCGImagePropertyTIFFCompression: 1,
                kCGImagePropertyTIFFSampleFormat: sampleFormat,
            ]
        }

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: imageData as CFData),
              let cgImage = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: bitsPerSample,
                  bitsPerPixel: bitsPerSample,
                  bytesPerRow: scanline,
                  space: colorSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              ),
              let destination = CGImageDestinationCreateWithURL(
                  URL(fileURLWithPath: path) as CFURL,
                  UTType.tiff.identifier as CFString,
                  1,
                  nil
              )
        else {
            throw SidecarError.serverError("Failed to write TIFF: \(path)")
        }
        CGImageDestinationAddImage(destination, cgImage, metadata as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw SidecarError.serverError("Failed to finalize TIFF: \(path)")
        }
    }

    private func resolve2DShape(_ shape: [Int]) -> (Int, Int) {
        if shape.count == 2 { return (shape[1], shape[0]) }
        if shape.count == 3 { return (shape[2], shape[1]) }
        throw SidecarError.serverError("Unsupported flow shape for export")
    }

    private func writeImageJRoi(points: [(Int, Int)], label: Int) throws -> Data {
        var data = Data()
        data.append(contentsOf: [0x49, 0x4a, 0x52, 0x4f, 0x49, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
        var count = UInt16(points.count).littleEndian
        data.append(Data(bytes: &count, count: MemoryLayout<UInt16>.size))
        let name = String(format: "%04d", label)
        var nameBytes = Array(name.utf8.prefix(16))
        while nameBytes.count < 16 { nameBytes.append(0) }
        data.append(contentsOf: nameBytes)
        for (x, y) in points {
            var sx = Int16(x).littleEndian
            var sy = Int16(y).littleEndian
            data.append(Data(bytes: &sx, count: MemoryLayout<Int16>.size))
            data.append(Data(bytes: &sy, count: MemoryLayout<Int16>.size))
        }
        return data
    }
}
