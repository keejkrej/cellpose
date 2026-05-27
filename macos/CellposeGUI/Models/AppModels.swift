import CoreGraphics
import Foundation

struct SegmentationParameters: Codable, Equatable {
    var diameter: Double = 0
    var flowThreshold: Double = 0.4
    var cellprobThreshold: Double = 0
    var percentileLow: Double = 1
    var percentileHigh: Double = 99
    var niter: Int = 0
    var minSize: Int = 15
    var stitchThreshold: Double = 0
    var anisotropy: Double = 1
    var flow3DSmooth: Double = 0
    var do3D: Bool = false

    enum CodingKeys: String, CodingKey {
        case diameter
        case flowThreshold = "flow_threshold"
        case cellprobThreshold = "cellprob_threshold"
        case percentileLow = "percentile_low"
        case percentileHigh = "percentile_high"
        case niter
        case minSize = "min_size"
        case stitchThreshold = "stitch_threshold"
        case anisotropy
        case flow3DSmooth = "flow3D_smooth"
        case do3D = "do_3D"
    }

    static func fromValues(
        diameter: Double,
        flowThreshold: Double,
        cellprobThreshold: Double,
        percentileLow: Double,
        percentileHigh: Double,
        niter: Int
    ) throws -> SegmentationParameters {
        guard (0 ... 100).contains(percentileLow),
              (0 ... 100).contains(percentileHigh),
              percentileLow < percentileHigh
        else {
            throw ValidationError.invalidPercentileRange
        }
        return SegmentationParameters(
            diameter: diameter,
            flowThreshold: flowThreshold,
            cellprobThreshold: cellprobThreshold,
            percentileLow: percentileLow,
            percentileHigh: percentileHigh,
            niter: max(niter, 1)
        )
    }

    var segmentationDiameter: Double? {
        diameter > 0 ? diameter : nil
    }

    enum ValidationError: LocalizedError {
        case invalidPercentileRange

        var errorDescription: String? {
            switch self {
            case .invalidPercentileRange:
                "Normalization percentile range must be 0 <= low < high <= 100"
            }
        }
    }
}

struct PreprocessingParameters: Codable, Equatable {
    var sharpenRadius: Double = 0
    var smoothRadius: Double = 0
    var tileNormBlocksize: Double = 0
    var tileNormSmooth3D: Double = 0
    var norm3D: Bool = true

    enum CodingKeys: String, CodingKey {
        case sharpenRadius = "sharpen_radius"
        case smoothRadius = "smooth_radius"
        case tileNormBlocksize = "tile_norm_blocksize"
        case tileNormSmooth3D = "tile_norm_smooth3D"
        case norm3D = "norm3D"
    }
}

struct DisplayParameters: Equatable {
    var grayLow: Double = 0
    var grayHigh: Double = 255
}

struct TrainingParameters: Codable, Equatable {
    var modelIndex: Int = 0
    var learningRate: Double = 1e-5
    var weightDecay: Double = 0.1
    var nEpochs: Int = 100
    var modelName: String = ""
    var trainDataFolder: String = ""
    var modelSaveFolder: String = ""

    enum CodingKeys: String, CodingKey {
        case modelIndex = "model_index"
        case learningRate = "learning_rate"
        case weightDecay = "weight_decay"
        case nEpochs = "n_epochs"
        case modelName = "model_name"
        case trainDataFolder = "train_data_folder"
        case modelSaveFolder = "model_save_folder"
    }

    static func createDefault(modelSaveFolder: String) -> TrainingParameters {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let suffix = formatter.string(from: Date())
        return TrainingParameters(
            modelName: "cpsam_\(suffix)",
            modelSaveFolder: modelSaveFolder
        )
    }
}

struct SeriesRecord: Identifiable, Equatable {
    let index: Int
    let label: String
    let path: String

    var id: Int { index }
}

struct SeriesState: Equatable {
    var folder: String = ""
    var subfolderTemplate: String = ""
    var filenameTemplate: String = ""
    var records: [SeriesRecord] = []
    var recordIndex: Int = 0
    var axisValues: [String: [String]] = [:]
    var axisSliderIndices: [String: Int] = [
        "position": 0,
        "time": 0,
        "channel": 0,
        "z": 0,
    ]

    static let axisOrder = ["position", "time", "channel", "z"]
    static let axisLabels = ["position": "P", "time": "T", "channel": "C", "z": "Z"]

    var currentRecord: SeriesRecord? {
        guard records.indices.contains(recordIndex) else { return nil }
        return records[recordIndex]
    }

    var isLoaded: Bool { !records.isEmpty }
}

struct InstanceClasses {
    var values: [Int32] = []

    mutating func ensureSize(_ ncells: Int) {
        if values.count < ncells {
            values.append(contentsOf: Array(repeating: 0, count: ncells - values.count))
        } else if values.count > ncells {
            values = Array(values.prefix(ncells))
        }
    }

    mutating func replace(ncells: Int, loaded: [Int32]? = nil) {
        var result = Array(repeating: Int32(0), count: ncells)
        if let loaded {
            let n = min(ncells, loaded.count)
            for i in 0 ..< n {
                result[i] = max(0, loaded[i])
            }
        }
        values = result
    }

    mutating func setClass(row: Int, classID: Int32) {
        guard row >= 0, row < values.count, classID >= 0 else { return }
        values[row] = classID
    }

    static func parseFilter(_ text: String) -> Int32? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return nil }
        guard let value = Int32(trimmed), value >= 0 else { return nil }
        return value
    }
}

enum ViewMode: String, CaseIterable, Identifiable {
    case image
    case gradXY
    case cellprob
    case restored

    var id: String { rawValue }

    var title: String { rawValue }
}

struct ImageData {
    let width: Int
    let height: Int
    let channels: Int
    let pixels: [UInt8]

    var cgImage: CGImage? {
        ImageData.makeCGImage(width: width, height: height, channels: channels, pixels: pixels)
    }

    static func makeCGImage(width: Int, height: Int, channels: Int, pixels: [UInt8]) -> CGImage? {
        guard channels == 3 || channels == 1 else { return nil }
        let bytesPerRow = width * channels
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        if channels == 3 {
            return CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 24,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            )
        }
        var rgb = [UInt8]()
        rgb.reserveCapacity(width * height * 3)
        for value in pixels {
            rgb.append(value)
            rgb.append(value)
            rgb.append(value)
        }
        guard let grayProvider = CGDataProvider(data: Data(rgb) as CFData) else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 24,
            bytesPerRow: width * 3,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: grayProvider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}

struct MaskData {
    let width: Int
    let height: Int
    let labels: [Int32]
    let colors: [[UInt8]]
    let outlineLabels: [Int32]?

    func color(at x: Int, y: Int) -> (UInt8, UInt8, UInt8, Float)? {
        guard x >= 0, y >= 0, x < width, y < height else { return nil }
        let label = labels[y * width + x]
        guard label > 0 else { return nil }
        let index = Int(label - 1)
        guard colors.indices.contains(index) else { return nil }
        let color = colors[index]
        return (color[0], color[1], color[2], 0.5)
    }

    func label(at x: Int, y: Int) -> Int32 {
        guard x >= 0, y >= 0, x < width, y < height else { return 0 }
        return labels[y * width + x]
    }
}
