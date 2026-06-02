import Foundation
import Observation

@MainActor
@Observable
final class SidecarProcessManager {
    private(set) var isReady = false
    private(set) var hasFailed = false
    private(set) var statusMessage = "Starting sidecar…"
    private(set) var baseURL: URL

    private var process: Process?
    private var repoRoot: URL
    private var stderrBuffer = ""

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

        for start in [
            Bundle.main.bundleURL,
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true),
        ] {
            if let root = findRepoRoot(startingAt: start) {
                return root
            }
        }

        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    }

    private static func findRepoRoot(startingAt url: URL) -> URL? {
        var current = url.standardizedFileURL
        let fileManager = FileManager.default

        while !current.path.isEmpty, current.path != "/" {
            let cellposeDir = current.appendingPathComponent("cellpose")
            let pyproject = current.appendingPathComponent("pyproject.toml")
            if fileManager.fileExists(atPath: cellposeDir.path),
               fileManager.fileExists(atPath: pyproject.path)
            {
                return current
            }
            current = current.deletingLastPathComponent()
        }

        return nil
    }

    private static func findUvExecutable() -> String {
        let pathEntries = (ProcessInfo.processInfo.environment["PATH"] ?? "")
            .split(separator: ":", omittingEmptySubsequences: true)
            .map(String.init)

        for entry in pathEntries {
            let candidate = URL(fileURLWithPath: entry).appendingPathComponent("uv").path
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }

        let home = FileManager.default.homeDirectoryForCurrentUser.path
        for candidate in [
            "/opt/homebrew/bin/uv",
            "/usr/local/bin/uv",
            "\(home)/.local/bin/uv",
            "\(home)/.cargo/bin/uv",
        ] {
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }

        return "uv"
    }

    func startIfNeeded() async {
        if RuntimeEnvironment.skipSidecar {
            isReady = true
            statusMessage = "Sidecar skipped"
            return
        }

        if await probeHealth() {
            isReady = true
            statusMessage = "Sidecar ready"
            return
        }

        guard process == nil else { return }
        statusMessage = "Launching Python sidecar…"

        let uvPath = Self.findUvExecutable()
        let proc = Process()
        proc.currentDirectoryURL = repoRoot
        proc.executableURL = URL(fileURLWithPath: uvPath)
        proc.arguments = [
            "run", "python", "-m", "cellpose.api",
            "--host", "127.0.0.1",
            "--port", "\(baseURL.port ?? 8787)",
        ]

        let stderrPipe = Pipe()
        proc.standardOutput = Pipe()
        proc.standardError = stderrPipe

        stderrBuffer = ""
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                self?.stderrBuffer += text
            }
        }

        do {
            try proc.run()
            process = proc
        } catch {
            hasFailed = true
            statusMessage = "Failed to launch sidecar: \(error.localizedDescription)"
            return
        }

        // PyTorch + first import can take well over 30s on cold start.
        for attempt in 0 ..< 180 {
            if let process, !process.isRunning {
                let exitCode = process.terminationStatus
                stderrPipe.fileHandleForReading.readabilityHandler = nil
                let detail = stderrBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
                hasFailed = true
                statusMessage = detail.isEmpty
                    ? "Sidecar exited with code \(exitCode)."
                    : "Sidecar failed: \(detail)"
                self.process = nil
                return
            }

            if await probeHealth() {
                stderrPipe.fileHandleForReading.readabilityHandler = nil
                isReady = true
                statusMessage = "Sidecar ready"
                return
            }

            statusMessage = "Waiting for sidecar (\(attempt + 1)/180)…"
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }

        stderrPipe.fileHandleForReading.readabilityHandler = nil
        let detail = stderrBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        hasFailed = true
        statusMessage = detail.isEmpty
            ? "Sidecar failed to start. Run `uv run python -m cellpose.api` manually to see errors."
            : "Sidecar failed to start: \(detail)"
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
