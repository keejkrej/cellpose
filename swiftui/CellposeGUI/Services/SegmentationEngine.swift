import CellposeGUICore
import Foundation

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

struct SeriesTemplateSuggestion {
    let subfolderTemplate: String
    let filenameTemplate: String
}

struct SeriesDiscovery {
    let folder: String
    let recordCount: Int
    let records: [SeriesDiscoveryRecord]
    let axes: [String: [String]]
    let dataset: SeriesDatasetPayload
}

struct SeriesDiscoveryRecord {
    let index: Int
    let label: String
    let path: String
}

struct SeriesDatasetPayload {
    let folder: String
    let subfolderTemplate: String
    let filenameTemplate: String
    let axes: [String: [String]]
    let axisIndex: [String: [String: Int]]?
    let lookup: [String: Int]?
    let records: [SeriesDatasetRecord]
}

struct SeriesDatasetRecord {
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
