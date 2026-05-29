import Compression
import Foundation

public enum ArrayCodec {
    public static func decode(_ payload: ArrayPayload) throws -> Data {
        guard let compressed = Data(base64Encoded: payload.dataB64) else {
            throw SidecarError.invalidResponse
        }
        return try decompressZlib(compressed)
    }

    public static func encodeRaw(_ raw: Data, dtype: String, shape: [Int]) throws -> ArrayPayload {
        let compressed = try compressZlib(raw)
        return ArrayPayload(
            dtype: dtype,
            shape: shape,
            dataB64: compressed.base64EncodedString()
        )
    }

    private static func decompressZlib(_ data: Data) throws -> Data {
        try decompress(data: data, algorithm: COMPRESSION_ZLIB)
    }

    private static func compressZlib(_ data: Data) throws -> Data {
        try compress(data: data, algorithm: COMPRESSION_ZLIB)
    }

    private static func decompress(data: Data, algorithm: compression_algorithm) throws -> Data {
        guard !data.isEmpty else { return Data() }
        let destinationCapacity = max(data.count * 4, 64)
        var output = Data(count: destinationCapacity)
        let decodedSize = output.withUnsafeMutableBytes { outputBuffer in
            data.withUnsafeBytes { inputBuffer in
                guard let out = outputBuffer.bindMemory(to: UInt8.self).baseAddress,
                      let input = inputBuffer.bindMemory(to: UInt8.self).baseAddress
                else { return 0 }
                return compression_decode_buffer(
                    out,
                    destinationCapacity,
                    input,
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
        guard !data.isEmpty else { return Data() }
        let destinationCapacity = max(data.count + 64, 64)
        var output = Data(count: destinationCapacity)
        let encodedSize = output.withUnsafeMutableBytes { outputBuffer in
            data.withUnsafeBytes { inputBuffer in
                guard let out = outputBuffer.bindMemory(to: UInt8.self).baseAddress,
                      let input = inputBuffer.bindMemory(to: UInt8.self).baseAddress
                else { return 0 }
                return compression_encode_buffer(
                    out,
                    destinationCapacity,
                    input,
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
