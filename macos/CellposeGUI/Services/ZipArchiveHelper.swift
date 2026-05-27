import Foundation

enum ZipArchiveHelper {
    static func withExtractedArchive(from archiveURL: URL, _ body: (URL) throws -> Void) throws {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("cellpose-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        try unzip(archive: archiveURL, to: temp)
        try body(temp)
    }

    static func writeArchive(at destination: URL, from stagingDirectory: URL) throws {
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try zip(sourceDirectory: stagingDirectory, to: destination)
    }

    private static func unzip(archive: URL, to destination: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-qq", archive.path, "-d", destination.path]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw SidecarError.serverError("Failed to extract session archive")
        }
    }

    private static func zip(sourceDirectory: URL, to destination: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        process.currentDirectoryURL = sourceDirectory
        process.arguments = ["-qr", destination.path, "."]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw SidecarError.serverError("Failed to write session archive")
        }
    }
}
