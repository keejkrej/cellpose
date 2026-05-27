import Foundation

final class SeriesDiscoveryService {
    private let supportedExtensions = ["jpg", "jpeg", "png", "tif", "tiff"]
    private let supportedSubfolderTemplates = ["Pos{p}", "Position{p}", "Pos_{p}", "Position_{p}"]
    private let supportedFilenameTemplates = ["img_{t}_{c}_{z}.jpg", "img_channel{c}_position{p}_time{t}_z{z}.tif"]

    func suggestTemplates(folder: String) throws -> SeriesTemplateSuggestion {
        guard FileManager.default.fileExists(atPath: folder) else {
            throw SidecarError.serverError("Folder not found: \(folder)")
        }

        var best: (String, String, Int)?
        for subfolder in supportedSubfolderTemplates {
            for filename in supportedFilenameTemplates {
                let matches = collectMatches(folder: folder, subfolderTemplate: subfolder, filenameTemplate: filename)
                guard !matches.isEmpty else { continue }
                if best == nil || matches.count > best!.2 {
                    best = (subfolder, filename, matches.count)
                }
            }
        }

        if let best {
            return SeriesTemplateSuggestion(subfolderTemplate: best.0, filenameTemplate: best.1)
        }
        return SeriesTemplateSuggestion(subfolderTemplate: "", filenameTemplate: "img_{t}_{c}_{z}.jpg")
    }

    func discover(folder: String, subfolderTemplate: String, filenameTemplate: String) throws -> SeriesDiscovery {
        var matches = collectMatches(folder: folder, subfolderTemplate: subfolderTemplate, filenameTemplate: filenameTemplate)
        guard !matches.isEmpty else {
            throw SidecarError.serverError("No files matched template in \(folder)")
        }

        matches.sort { lhs, rhs in
            let left = (
                sortRank(lhs.position),
                sortRank(lhs.time),
                sortRank(lhs.channel),
                sortRank(lhs.z),
                lhs.relativePath
            )
            let right = (
                sortRank(rhs.position),
                sortRank(rhs.time),
                sortRank(rhs.channel),
                sortRank(rhs.z),
                rhs.relativePath
            )
            return left < right
        }

        var lookup: [String: Int] = [:]
        var records: [SeriesDiscoveryRecord] = []
        var datasetRecords: [SeriesDatasetRecord] = []
        for (index, match) in matches.enumerated() {
            let key = "\(match.position)_\(match.time)_\(match.channel)_\(match.z)"
            guard lookup[key] == nil else {
                throw SidecarError.serverError("Duplicate series record for \(key)")
            }
            lookup[key] = index
            records.append(SeriesDiscoveryRecord(index: index, label: match.relativePath, path: match.path))
            datasetRecords.append(
                SeriesDatasetRecord(
                    label: match.relativePath,
                    path: match.path,
                    position: match.position,
                    time: match.time,
                    channel: match.channel,
                    z: match.z
                )
            )
        }

        let axes = [
            "position": uniqueSorted(matches.map(\.position)),
            "time": uniqueSorted(matches.map(\.time)),
            "channel": uniqueSorted(matches.map(\.channel)),
            "z": uniqueSorted(matches.map(\.z)),
        ]
        var axisIndex: [String: [String: Int]] = [:]
        for (axis, values) in axes {
            axisIndex[axis] = Dictionary(uniqueKeysWithValues: values.enumerated().map { ($1, $0) })
        }

        return SeriesDiscovery(
            folder: folder,
            recordCount: records.count,
            axes: axes,
            records: records,
            dataset: SeriesDatasetPayload(
                folder: folder,
                subfolderTemplate: subfolderTemplate,
                filenameTemplate: filenameTemplate,
                axes: axes,
                axisIndex: axisIndex,
                lookup: lookup,
                records: datasetRecords
            )
        )
    }

    private func uniqueSorted(_ values: [String]) -> [String] {
        Array(Set(values)).sorted { sortRank($0) < sortRank($1) }
    }

    private func sortRank(_ value: String) -> (Int, String) {
        if let number = Int(value) {
            return (0, String(format: "%010d", number))
        }
        return (1, value)
    }

    private func collectMatches(folder: String, subfolderTemplate: String, filenameTemplate: String) -> [SeriesMatch] {
        let subfolderPattern = compileTemplate(subfolderTemplate, allowEmpty: true)
        let filenamePattern = compileTemplate(filenameTemplate)
        var matches: [SeriesMatch] = []
        let folderURL = URL(fileURLWithPath: folder)
        let candidateFolders: [URL]
        if subfolderTemplate.isEmpty {
            candidateFolders = [folderURL]
        } else {
            candidateFolders = (try? FileManager.default.contentsOfDirectory(
                at: folderURL,
                includingPropertiesForKeys: nil
            ).filter(\.hasDirectoryPath).sorted(by: { $0.lastPathComponent < $1.lastPathComponent })) ?? []
        }

        for candidate in candidateFolders {
            let subfolderName = subfolderTemplate.isEmpty ? "" : candidate.lastPathComponent
            if !subfolderTemplate.isEmpty {
                guard subfolderPattern.firstMatch(in: subfolderName, range: NSRange(subfolderName.startIndex..., in: subfolderName)) != nil else {
                    continue
                }
            }

            let subfolderValues = namedGroups(from: subfolderName, pattern: subfolderPattern)
            let files = (try? FileManager.default.contentsOfDirectory(at: candidate, includingPropertiesForKeys: nil)
                .filter { !$0.hasDirectoryPath && isSupportedSeriesFile($0.lastPathComponent) }
                .sorted(by: { $0.lastPathComponent < $1.lastPathComponent })) ?? []

            for file in files {
                let name = file.lastPathComponent
                guard filenamePattern.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)) != nil else {
                    continue
                }
                var values = subfolderValues
                for (key, value) in namedGroups(from: name, pattern: filenamePattern) {
                    values[key] = value
                }
                let relativePath = subfolderName.isEmpty ? name : "\(subfolderName)/\(name)"
                matches.append(
                    SeriesMatch(
                        path: file.path,
                        relativePath: relativePath,
                        position: values["position", default: "0"],
                        time: values["time", default: "0"],
                        channel: values["channel", default: "0"],
                        z: values["z", default: "0"]
                    )
                )
            }
        }
        return matches
    }

    private func compileTemplate(_ template: String, allowEmpty: Bool = false) -> NSRegularExpression {
        if template.isEmpty {
            guard allowEmpty else {
                fatalError("Filename template cannot be empty.")
            }
            return try! NSRegularExpression(pattern: "^$", options: [.caseInsensitive])
        }
        return try! NSRegularExpression(pattern: compileTemplatePattern(template), options: [.caseInsensitive])
    }

    private func compileTemplatePattern(_ template: String) -> String {
        var parts = ["^"]
        let nsTemplate = template as NSString
        let placeholderRegex = try! NSRegularExpression(pattern: #"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"#)
        var searchStart = 0
        while searchStart < nsTemplate.length {
            let range = NSRange(location: searchStart, length: nsTemplate.length - searchStart)
            guard let match = placeholderRegex.firstMatch(in: template, options: [], range: range) else {
                parts.append(NSRegularExpression.escapedPattern(for: nsTemplate.substring(from: searchStart)))
                break
            }
            if match.range.location > searchStart {
                parts.append(NSRegularExpression.escapedPattern(for: nsTemplate.substring(with: NSRange(location: searchStart, length: match.range.location - searchStart))))
            }
            let name = nsTemplate.substring(with: match.range(at: 1))
            switch name {
            case "t", "time": parts.append("(?<time>.+?)")
            case "p", "position": parts.append("(?<position>.+?)")
            case "c", "channel": parts.append("(?<channel>.+?)")
            case "z": parts.append("(?<z>.+?)")
            default: fatalError("Unsupported placeholder {\(name)}")
            }
            searchStart = match.range.location + match.range.length
        }
        parts.append("$")
        return parts.joined()
    }

    private func namedGroups(from text: String, pattern: NSRegularExpression) -> [String: String] {
        guard let match = pattern.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else { return [:] }
        var result: [String: String] = [:]
        for name in ["position", "time", "channel", "z"] {
            let range = match.range(withName: name)
            guard range.location != NSNotFound, let swiftRange = Range(range, in: text) else { continue }
            result[name] = String(text[swiftRange])
        }
        return result
    }

    private func isSupportedSeriesFile(_ name: String) -> Bool {
        if name.lowercased().hasSuffix("_seg.cellpose") { return false }
        let ext = URL(fileURLWithPath: name).pathExtension.lowercased()
        return supportedExtensions.contains(ext)
    }

    private struct SeriesMatch {
        let path: String
        let relativePath: String
        let position: String
        let time: String
        let channel: String
        let z: String
    }
}
