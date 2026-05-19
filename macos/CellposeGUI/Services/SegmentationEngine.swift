import Compression
import Foundation

protocol SegmentationEngine: Sendable {
    func health() async throws -> SidecarHealth
    func listModels() async throws -> SidecarModels
    func loadImage(path: String, load3D: Bool) async throws -> SegmentationResult
    func loadSeg(path: String, load3D: Bool) async throws -> SegmentationResult
    func saveSeg(sessionID: String, path: String?) async throws -> String
    func exportMasks(sessionID: String, path: String, format: String) async throws -> String
    func segment(
        sessionID: String?,
        imagePayload: ArrayPayload?,
        filename: String?,
        modelName: String?,
        customModel: Bool,
        params: SegmentationParameters,
        preprocess: PreprocessingParameters
    ) async throws -> SegmentationResult
    func recomputeMasks(sessionID: String, params: SegmentationParameters) async throws -> SegmentationResult
    func preprocess(sessionID: String, params: PreprocessingParameters) async throws -> SegmentationResult
    func removeCells(sessionID: String, indices: [Int]) async throws -> SegmentationResult
    func addMask(sessionID: String, strokes: [[[Double]]], classID: Int32) async throws -> SegmentationResult
    func mergeCells(sessionID: String, source: Int, target: Int) async throws -> SegmentationResult
    func discoverSeries(
        folder: String,
        subfolderTemplate: String,
        filenameTemplate: String
    ) async throws -> SeriesDiscovery
    func suggestSeriesTemplates(folder: String) async throws -> SeriesTemplateSuggestion
    func exportOutlines(sessionID: String, path: String) async throws -> String
    func exportFlows(sessionID: String, path: String) async throws -> String
    func exportROIs(sessionID: String, path: String) async throws -> String
    func navigateSeries(sessionID: String, recordIndex: Int) async throws -> SegmentationResult
    func train(params: TrainingParameters) async throws -> TrainResult
    func addModel(path: String) async throws -> String
    func removeModel(name: String) async throws
}

struct SidecarHealth: Codable {
    let status: String
    let version: String
    let modelLoaded: Bool
    let device: String?

    enum CodingKeys: String, CodingKey {
        case status
        case version
        case modelLoaded = "model_loaded"
        case device
    }
}

struct SidecarModels: Codable {
    let builtin: [String]
    let custom: [String]

    var all: [String] { builtin + custom }
}

struct SeriesTemplateSuggestion: Codable {
    let subfolderTemplate: String
    let filenameTemplate: String

    enum CodingKeys: String, CodingKey {
        case subfolderTemplate = "subfolder_template"
        case filenameTemplate = "filename_template"
    }
}

struct SeriesDiscovery: Codable {
    let folder: String
    let recordCount: Int
    let records: [SeriesDiscoveryRecord]
    let axes: [String: [String]]
    let dataset: SeriesDatasetPayload

    enum CodingKeys: String, CodingKey {
        case folder
        case recordCount = "record_count"
        case records
        case axes
        case dataset
    }
}

struct SeriesDiscoveryRecord: Codable {
    let index: Int
    let label: String
    let path: String
}

struct SeriesDatasetPayload: Codable {
    let folder: String
    let subfolderTemplate: String
    let filenameTemplate: String
    let axes: [String: [String]]
    let axisIndex: [String: [String: Int]]?
    let lookup: [String: Int]?
    let records: [SeriesDatasetRecord]

    enum CodingKeys: String, CodingKey {
        case folder
        case subfolderTemplate = "subfolder_template"
        case filenameTemplate = "filename_template"
        case axes
        case axisIndex = "axis_index"
        case lookup
        case records
    }
}

struct SeriesDatasetRecord: Codable {
    let label: String
    let path: String
    let position: String
    let time: String
    let channel: String
    let z: String
}

struct TrainResult: Codable {
    let modelPath: String
    let modelName: String

    enum CodingKeys: String, CodingKey {
        case modelPath = "model_path"
        case modelName = "model_name"
    }
}

struct ArrayPayload: Codable {
    let dtype: String
    let shape: [Int]
    let dataB64: String

    enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataB64 = "data_b64"
    }
}

enum SidecarError: LocalizedError {
    case invalidResponse
    case serverError(String)
    case notReady

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "Invalid response from sidecar"
        case let .serverError(message):
            message
        case .notReady:
            "Cellpose sidecar is not ready"
        }
    }
}

enum ArrayCodec {
    static func decode(_ payload: ArrayPayload) throws -> Data {
        guard let compressed = Data(base64Encoded: payload.dataB64) else {
            throw SidecarError.invalidResponse
        }
        return try decompressZlib(compressed)
    }

    static func decodeUInt8Image(_ payload: ArrayPayload) throws -> ImageData {
        let raw = try decode(payload)
        let shape = payload.shape
        let width: Int
        let height: Int
        let channels: Int
        if shape.count == 3 {
            height = shape[0]
            width = shape[1]
            channels = shape[2]
        } else if shape.count == 2 {
            height = shape[0]
            width = shape[1]
            channels = 1
        } else {
            throw SidecarError.invalidResponse
        }
        let pixels = [UInt8](raw)
        return ImageData(width: width, height: height, channels: channels, pixels: pixels)
    }

    static func decodeInt32Labels(_ payload: ArrayPayload?) throws -> ([Int32], Int, Int)? {
        guard let payload else { return nil }
        let raw = try decode(payload)
        let shape = payload.shape
        let width: Int
        let height: Int
        if shape.count == 2 {
            height = shape[0]
            width = shape[1]
        } else if shape.count == 3, shape[0] == 1 {
            height = shape[1]
            width = shape[2]
        } else {
            throw SidecarError.invalidResponse
        }
        let count = raw.count / MemoryLayout<Int32>.size
        let labels = raw.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int32.self).prefix(count))
        }
        return (labels, width, height)
    }

    static func decodeColors(_ payload: ArrayPayload?) throws -> [[UInt8]] {
        guard let payload else { return [] }
        let raw = try decode(payload)
        let shape = payload.shape
        guard shape.count == 2, shape[1] == 3 else { return [] }
        var colors: [[UInt8]] = []
        colors.reserveCapacity(shape[0])
        for row in 0 ..< shape[0] {
            let offset = row * 3
            colors.append([raw[offset], raw[offset + 1], raw[offset + 2]])
        }
        return colors
    }

    static func encodeFloatImage(_ pixels: [Float], shape: [Int]) throws -> ArrayPayload {
        var raw = Data(capacity: pixels.count * MemoryLayout<Float>.size)
        for value in pixels {
            var little = value.bitPattern.littleEndian
            raw.append(Data(bytes: &little, count: MemoryLayout<UInt32>.size))
        }
        let compressed = try compressZlib(raw)
        return ArrayPayload(dtype: "<f4", shape: shape, dataB64: compressed.base64EncodedString())
    }

    private static func decompressZlib(_ data: Data) throws -> Data {
        try decompress(data: data, algorithm: COMPRESSION_ZLIB)
    }

    private static func compressZlib(_ data: Data) throws -> Data {
        try compress(data: data, algorithm: COMPRESSION_ZLIB)
    }

    private static func decompress(data: Data, algorithm: compression_algorithm) throws -> Data {
        let destinationCapacity = max(data.count * 4, 64)
        var output = Data(count: destinationCapacity)
        let decodedSize = output.withUnsafeMutableBytes { outputBuffer in
            data.withUnsafeBytes { inputBuffer in
                compression_decode_buffer(
                    outputBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    destinationCapacity,
                    inputBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    data.count,
                    nil,
                    algorithm
                )
            }
        }
        guard decodedSize > 0 else { throw SidecarError.invalidResponse }
        output.count = decodedSize
        return output
    }

    private static func compress(data: Data, algorithm: compression_algorithm) throws -> Data {
        let destinationCapacity = max(data.count, 64)
        var output = Data(count: destinationCapacity)
        let encodedSize = output.withUnsafeMutableBytes { outputBuffer in
            data.withUnsafeBytes { inputBuffer in
                compression_encode_buffer(
                    outputBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    destinationCapacity,
                    inputBuffer.bindMemory(to: UInt8.self).baseAddress!,
                    data.count,
                    nil,
                    algorithm
                )
            }
        }
        guard encodedSize > 0 else { throw SidecarError.invalidResponse }
        output.count = encodedSize
        return output
    }
}

extension SidecarSessionResponse {
    func toSegmentationResult() throws -> SegmentationResult {
        let image = try ArrayCodec.decodeUInt8Image(displayImage)
        let maskPayload = masks
        let outlinePayload = outlines
        let maskData: MaskData?
        if let maskPayload,
           let (labels, width, height) = try ArrayCodec.decodeInt32Labels(maskPayload)
        {
            let colors = try ArrayCodec.decodeColors(colors)
            let outlines = try ArrayCodec.decodeInt32Labels(outlinePayload)?.0
            maskData = MaskData(
                width: width,
                height: height,
                labels: labels,
                colors: colors,
                outlineLabels: outlines
            )
        } else {
            maskData = nil
        }
        return SegmentationResult(
            sessionID: sessionID,
            image: image,
            masks: maskData,
            ncells: ncells,
            recomputeMasks: recomputeMasks,
            filename: filename
        )
    }
}

struct SidecarSessionResponse: Codable {
    let sessionID: String
    let filename: String?
    let shape: [Int]
    let ncells: Int
    let masks: ArrayPayload?
    let outlines: ArrayPayload?
    let displayImage: ArrayPayload
    let colors: ArrayPayload?
    let instanceClasses: ArrayPayload?
    let recomputeMasks: Bool

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case filename
        case shape
        case ncells
        case masks
        case outlines
        case displayImage = "display_image"
        case colors
        case instanceClasses = "instance_classes"
        case recomputeMasks = "recompute_masks"
    }
}
