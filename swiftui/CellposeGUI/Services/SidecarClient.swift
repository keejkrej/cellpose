import CellposeGUICore
import Foundation

final class SidecarClient: Sendable {
    let baseURL: URL
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(baseURL: URL) {
        self.baseURL = baseURL
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 600
        config.timeoutIntervalForResource = 3600
        session = URLSession(configuration: config)
        encoder = JSONEncoder()
        decoder = JSONDecoder()
    }

    private func url(for path: String) -> URL {
        var trimmed = path
        if trimmed.hasPrefix("/") {
            trimmed.removeFirst()
        }
        return baseURL.appending(path: trimmed)
    }

    func get<T: Decodable>(_ path: String, as type: T.Type = T.self) async throws -> T {
        let (data, response) = try await session.data(from: url(for: path))
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    func post<T: Decodable, Body: Encodable>(
        _ path: String,
        body: Body,
        as type: T.Type = T.self
    ) async throws -> T {
        var request = URLRequest(url: url(for: path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(T.self, from: data)
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw SidecarError.invalidResponse
        }
        guard (200 ... 299).contains(http.statusCode) else {
            if let detail = try? decoder.decode(ErrorResponse.self, from: data) {
                throw SidecarError.serverError(detail.detail)
            }
            let message = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw SidecarError.serverError(message)
        }
    }

    private struct ErrorResponse: Decodable {
        let detail: String
    }
}

struct SidecarMlEngine: MlInferenceEngine {
    let client: SidecarClient

    func health() async throws -> SidecarHealth {
        try await client.get("/health")
    }

    func listModels() async throws -> SidecarModels {
        try await client.get("/models")
    }

    func infer(
        imagePath: String,
        modelName: String?,
        customModel: Bool,
        params: SegmentationParameters
    ) async throws -> InferResult {
        let response: InferResponsePayload = try await client.post(
            "/infer",
            body: InferRequest(
                path: imagePath,
                modelName: modelName,
                customModel: customModel,
                params: params
            )
        )
        return try ArrayCodec.toInferResult(response)
    }

    func recompute(flows: [ArrayPayload], params: SegmentationParameters) async throws -> RecomputeResult {
        let response: RecomputeResponsePayload = try await client.post(
            "/recompute",
            body: RecomputeRequest(flows: flows, params: params)
        )
        return try ArrayCodec.toRecomputeResult(response)
    }

    func train(params: TrainingParameters) async throws -> TrainResult {
        try await client.post("/train", body: params)
    }

    func addModel(path: String) async throws -> String {
        struct Response: Decodable {
            let modelName: String
            enum CodingKeys: String, CodingKey { case modelName = "model_name" }
        }
        let response: Response = try await client.post("/models/add", body: AddModelRequest(path: path))
        return response.modelName
    }

    func removeModel(name: String) async throws {
        struct Response: Decodable { let removed: String }
        _ = try await client.post("/models/remove", body: RemoveModelRequest(modelName: name)) as Response
    }
}

private struct InferRequest: Encodable {
    let path: String
    let modelName: String?
    let customModel: Bool
    let params: SegmentationParameters

    enum CodingKeys: String, CodingKey {
        case path
        case modelName = "model_name"
        case customModel = "custom_model"
        case params
    }
}

private struct RecomputeRequest: Encodable {
    let flows: [ArrayPayload]
    let params: SegmentationParameters
}

private struct AddModelRequest: Encodable {
    let path: String
}

private struct RemoveModelRequest: Encodable {
    let modelName: String

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
    }
}
