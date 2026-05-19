import Foundation
import Observation

@MainActor
@Observable
final class SidecarProcessManager {
    private(set) var isReady = false
    private(set) var statusMessage = "Starting sidecar…"
    private(set) var baseURL: URL

    private var process: Process?
    private var repoRoot: URL

    init(baseURL: URL? = nil) {
        if let baseURL {
            self.baseURL = baseURL
            self.repoRoot = SidecarProcessManager.defaultRepoRoot()
        } else if let envURL = ProcessInfo.processInfo.environment["SIDECAR_URL"],
                  let url = URL(string: envURL)
        {
            self.baseURL = url
            self.repoRoot = SidecarProcessManager.defaultRepoRoot()
        } else {
            self.baseURL = URL(string: "http://127.0.0.1:8787")!
            self.repoRoot = SidecarProcessManager.defaultRepoRoot()
        }
    }

    static func defaultRepoRoot() -> URL {
        if let envRoot = ProcessInfo.processInfo.environment["CELLPOSE_ROOT"],
           !envRoot.isEmpty
        {
            return URL(fileURLWithPath: envRoot, isDirectory: true)
        }
        let bundleRoot = Bundle.main.bundleURL
        for candidate in [
            bundleRoot.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),
            bundleRoot.deletingLastPathComponent().deletingLastPathComponent(),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true),
        ] {
            if FileManager.default.fileExists(atPath: candidate.appendingPathComponent("cellpose").path) {
                return candidate
            }
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    }

    func startIfNeeded() async {
        if await probeHealth() {
            isReady = true
            statusMessage = "Sidecar ready"
            return
        }

        guard process == nil else { return }
        statusMessage = "Launching Python sidecar…"

        let proc = Process()
        proc.currentDirectoryURL = repoRoot
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [
            "uv", "run", "python", "-m", "cellpose.gui.sidecar",
            "--host", "127.0.0.1",
            "--port", "\(baseURL.port ?? 8787)",
        ]

        let stderrPipe = Pipe()
        proc.standardOutput = Pipe()
        proc.standardError = stderrPipe

        do {
            try proc.run()
            process = proc
        } catch {
            statusMessage = "Failed to launch sidecar: \(error.localizedDescription)"
            return
        }

        for attempt in 0 ..< 60 {
            if await probeHealth() {
                isReady = true
                statusMessage = "Sidecar ready"
                return
            }
            statusMessage = "Waiting for sidecar (\(attempt + 1)/60)…"
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        statusMessage = "Sidecar failed to start"
    }

    func stop() {
        process?.terminate()
        process = nil
        isReady = false
    }

    private func probeHealth() async -> Bool {
        let client = SidecarClient(baseURL: baseURL)
        do {
            let health = try await client.get("/health", as: SidecarHealth.self)
            return health.status == "ok"
        } catch {
            return false
        }
    }
}
