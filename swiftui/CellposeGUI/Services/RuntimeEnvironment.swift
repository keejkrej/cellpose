import Foundation

enum RuntimeEnvironment {
    static var isRunningUnitTests: Bool {
        if ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil {
            return true
        }
        if let plugIns = Bundle.main.builtInPlugInsURL,
           let contents = try? FileManager.default.contentsOfDirectory(
               at: plugIns,
               includingPropertiesForKeys: nil
           ),
           contents.contains(where: { $0.pathExtension == "xctest" })
        {
            return true
        }
        return ProcessInfo.processInfo.arguments.contains { $0.hasSuffix(".xctest") || $0.contains("xctest") }
    }

    static var skipSidecar: Bool {
        isRunningUnitTests || ProcessInfo.processInfo.environment["CELLPOSE_SKIP_SIDECAR"] == "1"
    }
}
