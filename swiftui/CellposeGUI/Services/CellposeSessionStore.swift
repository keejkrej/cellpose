import CellposeGUICore
import Foundation

final class CellposeSessionStore {
    func defaultPath(imagePath: String) -> String {
        URL(fileURLWithPath: imagePath).deletingPathExtension().path + "_seg.npy"
    }

    func save(path: String, session: SessionState) throws {
        guard let imagePath = session.imagePath, let masks = session.masks else {
            throw SidecarError.serverError("Nothing to save")
        }

        let directory = URL(fileURLWithPath: path).deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let colors = MaskEditService.ensureColors(masks: masks, existingColors: masks.colors)
        let outlines = masks.outlineLabels ?? MaskEditService.computeOutlineLabels(
            labels: masks.labels,
            width: masks.width,
            height: masks.height
        )

        var payload: [String: Any] = [
            "outlines": SegNpyIO.fromLabels(outlines, width: masks.width, height: masks.height),
            "masks": SegNpyIO.fromLabels(masks.labels, width: masks.width, height: masks.height),
            "colors": SegNpyIO.fromColors(colors),
            "filename": imagePath,
            "flows": try session.flows.map { try SegNpyIO.fromArrayPayload($0) },
            "flow_threshold": session.segmentation.flowThreshold,
            "cellprob_threshold": session.segmentation.cellprobThreshold,
            "diameter": session.segmentation.segmentationDiameter as Any,
            "model_path": session.model == "cpsam" ? 0 : session.model,
            "restore": NSNull(),
            "ratio": 1.0,
        ]
        try SegNpyIO.write(path: path, payload: payload)
    }

    func load(path: String, imageLoader: ImageLoaderService) throws -> LoadedSession {
        let payload = try SegNpyIO.read(path: path)
        let stored = payload["filename"] as? String ?? ""
        let sourceImage = resolveSourceImage(stored: stored, sessionPath: path)
        guard FileManager.default.fileExists(atPath: sourceImage) else {
            throw SidecarError.serverError("Source image not found: \(sourceImage)")
        }
        return try buildLoadedSession(
            payload: payload,
            imagePath: sourceImage,
            image: try imageLoader.load(path: sourceImage)
        )
    }

    private func resolveSourceImage(stored: String, sessionPath: String) -> String {
        let sessionURL = URL(fileURLWithPath: sessionPath)
        let segDir = sessionURL.deletingLastPathComponent()
        let stem = sessionStem(sessionURL)
        let trimmed = stored.trimmingCharacters(in: .whitespacesAndNewlines)
        let extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

        if !trimmed.isEmpty {
            let basename = URL(fileURLWithPath: trimmed).lastPathComponent
            let candidate = segDir.appendingPathComponent(basename)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate.path
            }
        }

        for ext in extensions {
            let candidate = segDir.appendingPathComponent(stem + ext)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate.path
            }
        }

        if !trimmed.isEmpty {
            return segDir.appendingPathComponent(URL(fileURLWithPath: trimmed).lastPathComponent).path
        }
        return segDir.appendingPathComponent(stem).path
    }

    private func sessionStem(_ sessionURL: URL) -> String {
        let name = sessionURL.lastPathComponent
        let suffix = "_seg.npy"
        if name.hasSuffix(suffix) {
            return String(name.dropLast(suffix.count))
        }
        return sessionURL.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_seg", with: "")
    }

    func loadCompanion(sessionPath: String, imagePath: String, image: ImageData) throws -> LoadedSession {
        let payload = try SegNpyIO.read(path: sessionPath)
        let loaded = try buildLoadedSession(payload: payload, imagePath: imagePath, image: image)
        if loaded.masks.width != image.width || loaded.masks.height != image.height {
            throw SidecarError.serverError(
                "Session mask size \(loaded.masks.width)x\(loaded.masks.height) does not match image \(image.width)x\(image.height)"
            )
        }
        return loaded
    }

    private func buildLoadedSession(payload: [String: Any], imagePath: String, image: ImageData) throws -> LoadedSession {
        guard let masksArray = payload["masks"] as? SegNpyArray else {
            throw SidecarError.serverError("Invalid _seg.npy file: missing masks")
        }

        let labels = try SegNpyIO.readLabels(masksArray)
        let width = masksArray.shape.count == 2 ? masksArray.shape[1] : masksArray.shape[2]
        let height = masksArray.shape.count == 2 ? masksArray.shape[0] : masksArray.shape[1]

        let colors: [[UInt8]]
        if let colorsArray = payload["colors"] as? SegNpyArray {
            colors = SegNpyIO.readColors(colorsArray)
        } else {
            colors = MaskEditService.defaultColors(ncells: Int(labels.max() ?? 0))
        }

        let outlines: [Int32]
        if let outlinesArray = payload["outlines"] as? SegNpyArray {
            outlines = try SegNpyIO.readLabels(outlinesArray)
        } else {
            outlines = MaskEditService.computeOutlineLabels(labels: labels, width: width, height: height)
        }

        var flows: [ArrayPayload] = []
        if let flowItems = payload["flows"] as? [Any] {
            for item in flowItems {
                if let flowArray = item as? SegNpyArray {
                    flows.append(try SegNpyIO.toArrayPayload(flowArray))
                }
            }
        }

        let diameter = payload["diameter"] as? Double
        return LoadedSession(
            imagePath: imagePath,
            image: image,
            masks: MaskData(
                width: width,
                height: height,
                labels: labels,
                colors: colors,
                outlineLabels: outlines
            ),
            flows: flows,
            recomputeMasks: !flows.isEmpty,
            model: formatModel(payload["model_path"]),
            segmentation: SegmentationParameters(
                diameter: diameter ?? 0,
                flowThreshold: payload["flow_threshold"] as? Double ?? 0.4,
                cellprobThreshold: payload["cellprob_threshold"] as? Double ?? 0.0,
                percentileLow: 1,
                percentileHigh: 99,
                niter: 200,
                minSize: 15
            )
        )
    }

    private func formatModel(_ value: Any?) -> String {
        switch value {
        case nil, is NSNull:
            return "cpsam"
        case let intValue as Int where intValue == 0:
            return "cpsam"
        case let intValue as Int64 where intValue == 0:
            return "cpsam"
        case let string as String where string == "0" || string.isEmpty:
            return "cpsam"
        case let string as String:
            return string
        default:
            return String(describing: value ?? "cpsam")
        }
    }
}
