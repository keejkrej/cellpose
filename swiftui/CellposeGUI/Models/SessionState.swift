import CellposeGUICore
import Foundation

final class SessionState {
    var imagePath: String?
    var image: ImageData?
    var masks: MaskData?
    var flows: [ArrayPayload] = []
    var recomputeMasks = false
    var model = "cpsam"
    var segmentation = SegmentationParameters()
    var seriesDataset: SeriesDatasetPayload?

    var ncells: Int {
        guard let labels = masks?.labels, !labels.isEmpty else { return 0 }
        return Int(labels.max() ?? 0)
    }

    var hasImage: Bool { image != nil && imagePath != nil }
    var hasMasks: Bool { ncells > 0 }
}

struct LoadedSession {
    let imagePath: String
    let image: ImageData
    let masks: MaskData
    let flows: [ArrayPayload]
    let recomputeMasks: Bool
    let model: String
    let segmentation: SegmentationParameters
}

struct InferResult {
    let masks: MaskData?
    let flows: [ArrayPayload]
    let ncells: Int
    let recomputeMasks: Bool
}

struct RecomputeResult {
    let masks: MaskData?
    let ncells: Int
}
