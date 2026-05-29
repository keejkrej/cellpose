import CellposeGUICore
import XCTest

final class SegNpyIOTests: XCTestCase {
    private func fixturePath(_ name: String) -> String {
        var url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while url.path != "/" {
            let candidate = url.appendingPathComponent("tests/fixtures/seg_npy/\(name)")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate.path
            }
            url.deleteLastPathComponent()
        }
        fatalError("missing fixture: \(name)")
    }

    func testMinimalFixture() throws {
        let payload = try SegNpyIO.read(path: fixturePath("minimal_seg.npy"))
        guard let masks = payload["masks"] as? SegNpyArray else {
            XCTFail("missing masks")
            return
        }
        XCTAssertEqual(2, try SegNpyIO.readLabels(masks).max())
    }

    func testLegacyFixture() throws {
        let payload = try SegNpyIO.read(path: fixturePath("legacy_gui_seg.npy"))
        guard let masks = payload["masks"] as? SegNpyArray else {
            XCTFail("missing masks")
            return
        }
        XCTAssertEqual(2, try SegNpyIO.readLabels(masks).max())
    }

    func testInstanceClassesFixture() throws {
        let payload = try SegNpyIO.read(path: fixturePath("with_instance_classes_seg.npy"))
        guard let classesArray = payload["instance_classes"] as? SegNpyArray else {
            XCTFail("missing instance_classes")
            return
        }
        XCTAssertEqual([1, 2], try SegNpyIO.readLabels(classesArray))
    }
}
