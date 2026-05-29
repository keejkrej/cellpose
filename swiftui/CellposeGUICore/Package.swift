// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "CellposeGUICore",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "CellposeGUICore", targets: ["CellposeGUICore"]),
    ],
    targets: [
        .target(name: "CellposeGUICore"),
        .testTarget(
            name: "CellposeGUICoreTests",
            dependencies: ["CellposeGUICore"]
        ),
    ]
)
