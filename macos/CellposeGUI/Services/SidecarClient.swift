import Foundation

final class SidecarClient: Sendable {
    let baseURL: URL
    private let session: URLSession

    init(baseURL: URL) {
        self.baseURL = baseURL
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 600
        config.timeoutIntervalForResource = 3600
        session = URLSession(configuration: config)
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
        return try JSONDecoder().decode(T.self, from: data)
    }

    func post<T: Decodable, Body: Encodable>(
        _ path: String,
        body: Body,
        as type: T.Type = T.self
    ) async throws -> T {
        var request = URLRequest(url: url(for: path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw SidecarError.invalidResponse
        }
        guard (200 ... 299).contains(http.statusCode) else {
            if let detail = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
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

struct SidecarSegmentationEngine: SegmentationEngine {
    let client: SidecarClient

    func health() async throws -> SidecarHealth {
        try await client.get("/health")
    }

    func listModels() async throws -> SidecarModels {
        try await client.get("/models")
    }

    func loadImage(path: String, load3D: Bool) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/io/load-image",
            body: LoadImageRequest(path: path, load3D: load3D)
        )
        return try response.toSegmentationResult()
    }

    func loadSeg(path: String, load3D: Bool) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/io/load-seg",
            body: LoadSegRequest(path: path, load3D: load3D)
        )
        return try response.toSegmentationResult()
    }

    func saveSeg(sessionID: String, path: String?) async throws -> String {
        struct SaveResponse: Decodable { let path: String }
        let response: SaveResponse = try await client.post(
            "/io/save-seg",
            body: SaveSegRequest(sessionID: sessionID, path: path)
        )
        return response.path
    }

    func exportMasks(sessionID: String, path: String, format: String) async throws -> String {
        struct ExportResponse: Decodable { let path: String }
        let response: ExportResponse = try await client.post(
            "/export/masks",
            body: ExportMasksRequest(sessionID: sessionID, path: path, format: format)
        )
        return response.path
    }

    func segment(
        sessionID: String?,
        imagePayload: ArrayPayload?,
        filename: String?,
        modelName: String?,
        customModel: Bool,
        params: SegmentationParameters,
        preprocess: PreprocessingParameters
    ) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/segment",
            body: SegmentRequest(
                sessionID: sessionID,
                image: imagePayload,
                filename: filename,
                modelName: modelName,
                customModel: customModel,
                params: params,
                preprocess: preprocess
            )
        )
        return try response.toSegmentationResult()
    }

    func recomputeMasks(sessionID: String, params: SegmentationParameters) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/recompute-masks",
            body: RecomputeRequest(sessionID: sessionID, params: params)
        )
        return try response.toSegmentationResult()
    }

    func preprocess(sessionID: String, params: PreprocessingParameters) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/preprocess/\(sessionID)",
            body: params
        )
        return try response.toSegmentationResult()
    }

    func removeCells(sessionID: String, indices: [Int]) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/masks/remove",
            body: RemoveCellsRequest(sessionID: sessionID, cellIndices: indices)
        )
        return try response.toSegmentationResult()
    }

    func addMask(sessionID: String, strokes: [[[Double]]], classID: Int32) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/masks/add",
            body: AddMaskRequest(sessionID: sessionID, strokes: strokes, classID: Int(classID))
        )
        return try response.toSegmentationResult()
    }

    func mergeCells(sessionID: String, source: Int, target: Int) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/masks/merge",
            body: MergeCellsRequest(sessionID: sessionID, sourceIndex: source, targetIndex: target)
        )
        return try response.toSegmentationResult()
    }

    func discoverSeries(
        folder: String,
        subfolderTemplate: String,
        filenameTemplate: String
    ) async throws -> SeriesDiscovery {
        try await client.post(
            "/series/discover",
            body: SeriesDiscoverRequest(
                folder: folder,
                subfolderTemplate: subfolderTemplate,
                filenameTemplate: filenameTemplate
            )
        )
    }

    func suggestSeriesTemplates(folder: String) async throws -> SeriesTemplateSuggestion {
        try await client.post("/series/suggest", body: SeriesSuggestRequest(folder: folder))
    }

    func exportOutlines(sessionID: String, path: String) async throws -> String {
        struct ExportResponse: Decodable { let path: String }
        let response: ExportResponse = try await client.post(
            "/export/outlines",
            body: ExportPathRequest(sessionID: sessionID, path: path, format: "txt")
        )
        return response.path
    }

    func exportFlows(sessionID: String, path: String) async throws -> String {
        struct ExportResponse: Decodable { let path: String }
        let response: ExportResponse = try await client.post(
            "/export/flows",
            body: ExportPathRequest(sessionID: sessionID, path: path, format: "tif")
        )
        return response.path
    }

    func exportROIs(sessionID: String, path: String) async throws -> String {
        struct ExportResponse: Decodable { let path: String }
        let response: ExportResponse = try await client.post(
            "/export/rois",
            body: ExportPathRequest(sessionID: sessionID, path: path, format: "zip")
        )
        return response.path
    }

    func navigateSeries(sessionID: String, recordIndex: Int) async throws -> SegmentationResult {
        let response: SidecarSessionResponse = try await client.post(
            "/series/navigate",
            body: SeriesNavigateRequest(sessionID: sessionID, recordIndex: recordIndex)
        )
        return try response.toSegmentationResult()
    }

    func train(params: TrainingParameters) async throws -> TrainResult {
        try await client.post("/train", body: params)
    }

    func addModel(path: String) async throws -> String {
        struct Response: Decodable { let modelName: String; enum CodingKeys: String, CodingKey { case modelName = "model_name" } }
        let response: Response = try await client.post("/models/add", body: AddModelRequest(path: path))
        return response.modelName
    }

    func removeModel(name: String) async throws {
        struct Response: Decodable { let removed: String }
        _ = try await client.post("/models/remove", body: RemoveModelRequest(modelName: name)) as Response
    }
}

private struct LoadImageRequest: Encodable {
    let path: String
    let load3D: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case load3D = "load_3D"
    }
}

private struct LoadSegRequest: Encodable {
    let path: String
    let load3D: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case load3D = "load_3D"
    }
}

private struct SaveSegRequest: Encodable {
    let sessionID: String
    let path: String?

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case path
    }
}

private struct ExportMasksRequest: Encodable {
    let sessionID: String
    let path: String
    let format: String

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case path
        case format
    }
}

private struct SegmentRequest: Encodable {
    let sessionID: String?
    let image: ArrayPayload?
    let filename: String?
    let modelName: String?
    let customModel: Bool
    let params: SegmentationParameters
    let preprocess: PreprocessingParameters

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case image
        case filename
        case modelName = "model_name"
        case customModel = "custom_model"
        case params
        case preprocess
    }
}

private struct RecomputeRequest: Encodable {
    let sessionID: String
    let params: SegmentationParameters

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case params
    }
}

private struct RemoveCellsRequest: Encodable {
    let sessionID: String
    let cellIndices: [Int]

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case cellIndices = "cell_indices"
    }
}

private struct AddMaskRequest: Encodable {
    let sessionID: String
    let strokes: [[[Double]]]
    let classID: Int

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case strokes
        case classID = "class_id"
    }
}

private struct MergeCellsRequest: Encodable {
    let sessionID: String
    let sourceIndex: Int
    let targetIndex: Int

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case sourceIndex = "source_index"
        case targetIndex = "target_index"
    }
}

private struct SeriesSuggestRequest: Encodable {
    let folder: String
}

private struct ExportPathRequest: Encodable {
    let sessionID: String
    let path: String
    let format: String

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case path
        case format
    }
}

private struct SeriesDiscoverRequest: Encodable {
    let folder: String
    let subfolderTemplate: String
    let filenameTemplate: String

    enum CodingKeys: String, CodingKey {
        case folder
        case subfolderTemplate = "subfolder_template"
        case filenameTemplate = "filename_template"
    }
}

private struct SeriesNavigateRequest: Encodable {
    let sessionID: String
    let recordIndex: Int

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case recordIndex = "record_index"
    }
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
