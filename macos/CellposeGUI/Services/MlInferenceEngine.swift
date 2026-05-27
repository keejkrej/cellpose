import Compression
import Foundation

protocol MlInferenceEngine: Sendable {
    func health() async throws -> SidecarHealth
    func listModels() async throws -> SidecarModels
    func infer(
        imagePath: String,
        modelName: String?,
        customModel: Bool,
        params: SegmentationParameters
    ) async throws -> InferResult
    func recompute(flows: [ArrayPayload], params: SegmentationParameters) async throws -> RecomputeResult
    func train(params: TrainingParameters) async throws -> TrainResult
    func addModel(path: String) async throws -> String
    func removeModel(name: String) async throws
}

enum ArrayCodec {
    static func decode(_ payload: ArrayPayload) throws -> Data {
        guard let compressed = Data(base64Encoded: payload.dataB64) else {
            throw SidecarError.invalidResponse
        }
        return try decompressZlib(compressed)
    }

    static func encodeRaw(_ raw: Data, dtype: String, shape: [Int]) throws -> ArrayPayload {
        let compressed = try compressZlib(raw)
        return ArrayPayload(dtype: dtype, shape: shape, dataB64: compressed.base64EncodedString())
    }

    static func decodeMasks(_ payload: ArrayPayload?) throws -> MaskData? {
        guard let payload,
              let (labels, width, height) = try decodeInt32Labels(payload)
        else { return nil }
        let colors = MaskEditService.defaultColors(ncells: Int(labels.max() ?? 0))
        let outlines = MaskEditService.computeOutlineLabels(labels: labels, width: width, height: height)
        return MaskData(
            width: width,
            height: height,
            labels: labels,
            colors: colors,
            outlineLabels: outlines
        )
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

        if payload.dtype == "uint16" || payload.dtype == "uint32" {
            var labels = [Int32]()
            labels.reserveCapacity(raw.count / MemoryLayout<UInt16>.size)
            raw.withUnsafeBytes { buffer in
                let shorts = buffer.bindMemory(to: UInt16.self)
                for value in shorts {
                    labels.append(Int32(value))
                }
            }
            return (labels, width, height)
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

    static func toInferResult(_ response: InferResponsePayload) throws -> InferResult {
        InferResult(
            masks: try decodeMasks(response.masks),
            flows: response.flows,
            ncells: response.ncells,
            recomputeMasks: response.recomputeMasks
        )
    }

    static func toRecomputeResult(_ response: RecomputeResponsePayload) throws -> RecomputeResult {
        RecomputeResult(
            masks: try decodeMasks(response.masks),
            ncells: response.ncells
        )
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

struct InferResponsePayload: Decodable {
    let masks: ArrayPayload?
    let flows: [ArrayPayload]
    let ncells: Int
    let recomputeMasks: Bool

    enum CodingKeys: String, CodingKey {
        case masks
        case flows
        case ncells
        case recomputeMasks = "recompute_masks"
    }
}

struct RecomputeResponsePayload: Decodable {
    let masks: ArrayPayload?
    let ncells: Int
}
