import Foundation

final class CellposeSessionStore {
    func defaultPath(imagePath: String) -> String {
        URL(fileURLWithPath: imagePath).deletingPathExtension().path + "_seg.cellpose"
    }

    func save(path: String, session: SessionState) throws {
        guard let imagePath = session.imagePath, let masks = session.masks else {
            throw SidecarError.serverError("Nothing to save")
        }

        let directory = URL(fileURLWithPath: path).deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let staging = FileManager.default.temporaryDirectory
            .appendingPathComponent("cellpose-save-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }

        let arraysDir = staging.appendingPathComponent("arrays", isDirectory: true)
        try FileManager.default.createDirectory(at: arraysDir, withIntermediateDirectories: true)

        var manifestArrays: [String: ManifestArray] = [:]
        try writeUInt16Array(to: arraysDir, arrays: &manifestArrays, name: "masks", masks: masks)

        for (index, flow) in session.flows.enumerated() {
            try writeRawArray(to: arraysDir, arrays: &manifestArrays, name: "flows_\(index)", payload: flow)
        }

        let colors = MaskEditService.ensureColors(masks: masks, existingColors: masks.colors)
        try writeColorArray(to: arraysDir, arrays: &manifestArrays, colors: colors)

        let manifest = SessionManifest(
            version: 1,
            sourceImage: imagePath,
            segmentation: session.segmentation,
            model: session.model,
            recomputeMasks: session.recomputeMasks,
            arrays: manifestArrays
        )
        let manifestData = try JSONEncoder.snakeCase.encode(manifest)
        try manifestData.write(to: staging.appendingPathComponent("manifest.json"))

        try ZipArchiveHelper.writeArchive(at: URL(fileURLWithPath: path), from: staging)
    }

    func load(path: String, imageLoader: ImageLoaderService) throws -> LoadedSession {
        var loaded: LoadedSession?
        try ZipArchiveHelper.withExtractedArchive(from: URL(fileURLWithPath: path)) { root in
            let manifestURL = root.appendingPathComponent("manifest.json")
            let manifestData = try Data(contentsOf: manifestURL)
            let manifest = try JSONDecoder.snakeCase.decode(SessionManifest.self, from: manifestData)
            guard manifest.version == 1 else {
                throw SidecarError.serverError("Unsupported session version: \(manifest.version)")
            }

            var sourceImage = manifest.sourceImage
            if !sourceImage.hasPrefix("/") {
                sourceImage = URL(fileURLWithPath: path).deletingLastPathComponent()
                    .appendingPathComponent(sourceImage).path
            }
            guard FileManager.default.fileExists(atPath: sourceImage) else {
                throw SidecarError.serverError("Source image not found: \(sourceImage)")
            }

            guard let masksEntry = manifest.arrays["masks"] else {
                throw SidecarError.serverError("Invalid session file: missing masks")
            }
            let labels = try readMaskLabels(root: root, entry: masksEntry)
            let colors: [[UInt8]]
            if let colorsEntry = manifest.arrays["colors"] {
                colors = try readColors(root: root, entry: colorsEntry)
            } else {
                colors = MaskEditService.defaultColors(ncells: Int(labels.max() ?? 0))
            }

            var flows: [ArrayPayload] = []
            var index = 0
            while let flowEntry = manifest.arrays["flows_\(index)"] {
                flows.append(try readPayload(root: root, entry: flowEntry))
                index += 1
            }

            let width = masksEntry.shape[masksEntry.shape.count - 1]
            let height = masksEntry.shape[masksEntry.shape.count - 2]
            let maskData = MaskData(
                width: width,
                height: height,
                labels: labels,
                colors: colors,
                outlineLabels: MaskEditService.computeOutlineLabels(labels: labels, width: width, height: height)
            )

            loaded = LoadedSession(
                imagePath: sourceImage,
                image: try imageLoader.load(path: sourceImage),
                masks: maskData,
                flows: flows,
                recomputeMasks: manifest.recomputeMasks,
                model: manifest.model,
                segmentation: manifest.segmentation
            )
        }
        guard let loaded else { throw SidecarError.invalidResponse }
        return loaded
    }

    private func writeUInt16Array(
        to directory: URL,
        arrays: inout [String: ManifestArray],
        name: String,
        masks: MaskData
    ) throws {
        var raw = Data(capacity: masks.labels.count * MemoryLayout<UInt16>.size)
        for label in masks.labels {
            var value = UInt16(clamping: Int(label)).littleEndian
            raw.append(Data(bytes: &value, count: MemoryLayout<UInt16>.size))
        }
        let relPath = "arrays/\(name).uint16"
        try raw.write(to: directory.appendingPathComponent("\(name).uint16"))
        arrays[name] = ManifestArray(file: relPath, dtype: "uint16", shape: [masks.height, masks.width])
    }

    private func writeRawArray(
        to directory: URL,
        arrays: inout [String: ManifestArray],
        name: String,
        payload: ArrayPayload
    ) throws {
        let ext = payloadExtension(payload.dtype)
        let relPath = "arrays/\(name).\(ext)"
        let raw = try ArrayCodec.decode(payload)
        try raw.write(to: directory.appendingPathComponent("\(name).\(ext)"))
        arrays[name] = ManifestArray(file: relPath, dtype: payload.dtype, shape: payload.shape)
    }

    private func writeColorArray(
        to directory: URL,
        arrays: inout [String: ManifestArray],
        colors: [[UInt8]]
    ) throws {
        var raw = Data(capacity: colors.count * 3)
        for color in colors {
            raw.append(contentsOf: color.prefix(3))
        }
        let relPath = "arrays/colors.uint8"
        try raw.write(to: directory.appendingPathComponent("colors.uint8"))
        arrays["colors"] = ManifestArray(file: relPath, dtype: "uint8", shape: [colors.count, 3])
    }

    private func readMaskLabels(root: URL, entry: ManifestArray) throws -> [Int32] {
        let raw = try readRaw(root: root, entry: entry)
        return raw.withUnsafeBytes { buffer in
            buffer.bindMemory(to: UInt16.self).map { Int32(UInt16(littleEndian: $0)) }
        }
    }

    private func readColors(root: URL, entry: ManifestArray) throws -> [[UInt8]] {
        let raw = try readRaw(root: root, entry: entry)
        let count = entry.shape[0]
        return (0 ..< count).map { row in
            let offset = row * 3
            return [raw[offset], raw[offset + 1], raw[offset + 2]]
        }
    }

    private func readPayload(root: URL, entry: ManifestArray) throws -> ArrayPayload {
        let raw = try readRaw(root: root, entry: entry)
        return try ArrayCodec.encodeRaw(raw, dtype: entry.dtype, shape: entry.shape)
    }

    private func readRaw(root: URL, entry: ManifestArray) throws -> Data {
        let fileURL = root.appendingPathComponent(entry.file)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw SidecarError.serverError("Missing array entry: \(entry.file)")
        }
        return try Data(contentsOf: fileURL)
    }

    private func payloadExtension(_ dtype: String) -> String {
        switch dtype {
        case "uint8": "uint8"
        case "float32": "float32"
        case "float64": "float64"
        default: dtype
        }
    }

    private struct SessionManifest: Codable {
        let version: Int
        let sourceImage: String
        let segmentation: SegmentationParameters
        let model: String
        let recomputeMasks: Bool
        let arrays: [String: ManifestArray]

        enum CodingKeys: String, CodingKey {
            case version
            case sourceImage = "source_image"
            case segmentation
            case model
            case recomputeMasks = "recompute_masks"
            case arrays
        }
    }

    private struct ManifestArray: Codable {
        let file: String
        let dtype: String
        let shape: [Int]
    }
}

private extension JSONEncoder {
    static var snakeCase: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

private extension JSONDecoder {
    static var snakeCase: JSONDecoder {
        JSONDecoder()
    }
}
