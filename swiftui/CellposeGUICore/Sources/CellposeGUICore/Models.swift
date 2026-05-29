import Foundation

public struct ArrayPayload: Codable, Sendable {
    public let dtype: String
    public let shape: [Int]
    public let dataB64: String

    public init(dtype: String, shape: [Int], dataB64: String) {
        self.dtype = dtype
        self.shape = shape
        self.dataB64 = dataB64
    }

    enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case dataB64 = "data_b64"
    }
}

public enum SidecarError: LocalizedError, Sendable {
    case invalidResponse
    case serverError(String)
    case notReady

    public var errorDescription: String? {
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
