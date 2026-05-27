import Foundation

enum MaskEditService {
    static func applyMasks(
        labels: [Int32],
        width: Int,
        height: Int,
        existingColors: [[UInt8]]? = nil
    ) -> MaskData {
        var mutable = labels
        renumber(&mutable)
        let ncells = mutable.isEmpty ? 0 : Int(mutable.max() ?? 0)
        let colors = ensureColors(ncells: ncells, existingColors: existingColors)
        let outlines = computeOutlineLabels(labels: mutable, width: width, height: height)
        return MaskData(
            width: width,
            height: height,
            labels: mutable,
            colors: colors,
            outlineLabels: outlines
        )
    }

    static func ensureColors(ncells: Int, existingColors: [[UInt8]]?) -> [[UInt8]] {
        if let existingColors, existingColors.count >= ncells {
            return existingColors
        }
        return defaultColors(ncells: ncells)
    }

    static func ensureColors(masks: MaskData, existingColors: [[UInt8]]?) -> [[UInt8]] {
        ensureColors(ncells: Int(masks.labels.max() ?? 0), existingColors: existingColors)
    }

    static func defaultColors(ncells: Int) -> [[UInt8]] {
        (0 ..< max(ncells, 0)).map { colorForClass(classID: Int32($0)) }
    }

    static func colorForClass(classID: Int32) -> [UInt8] {
        var generator = SeededRandomNumberGenerator(seed: UInt64(42 + Int(classID) * 7919))
        return [
            UInt8.random(in: 50 ... 255, using: &generator),
            UInt8.random(in: 50 ... 255, using: &generator),
            UInt8.random(in: 50 ... 255, using: &generator),
        ]
    }

    static func withClassColors(masks: MaskData, classIDs: [Int32]) -> MaskData {
        let ncells = masks.labels.isEmpty ? 0 : Int(masks.labels.max() ?? 0)
        guard ncells > 0 else { return masks }
        let colors = (0 ..< ncells).map { row in
            let classID = row < classIDs.count ? classIDs[row] : Int32(0)
            return colorForClass(classID: classID)
        }
        return MaskData(
            width: masks.width,
            height: masks.height,
            labels: masks.labels,
            colors: colors,
            outlineLabels: masks.outlineLabels
        )
    }

    static func computeOutlineLabels(labels: [Int32], width: Int, height: Int) -> [Int32] {
        var outlines = [Int32](repeating: 0, count: labels.count)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let label = labels[y * width + x]
                guard label > 0 else { continue }
                if x > 0, labels[y * width + x - 1] != label {
                    outlines[y * width + x] = label
                    continue
                }
                if x + 1 < width, labels[y * width + x + 1] != label {
                    outlines[y * width + x] = label
                    continue
                }
                if y > 0, labels[(y - 1) * width + x] != label {
                    outlines[y * width + x] = label
                    continue
                }
                if y + 1 < height, labels[(y + 1) * width + x] != label {
                    outlines[y * width + x] = label
                }
            }
        }
        return outlines
    }

    static func removeCells(masks: MaskData, indices: [Int]) -> MaskData {
        var labels = masks.labels
        for idx in indices.sorted(by: >) {
            guard idx > 0, idx <= Int(labels.max() ?? 0) else { continue }
            for i in labels.indices {
                if labels[i] == Int32(idx) {
                    labels[i] = 0
                } else if labels[i] > Int32(idx) {
                    labels[i] -= 1
                }
            }
        }
        var colors = masks.colors
        for idx in indices.sorted(by: >) {
            guard idx > 0, idx <= colors.count else { continue }
            colors.remove(at: idx - 1)
        }
        return applyMasks(labels: labels, width: masks.width, height: masks.height, existingColors: colors)
    }

    static func mergeCells(masks: MaskData, source: Int, target: Int) -> MaskData {
        guard source != target else { return masks }
        var labels = masks.labels
        for i in labels.indices where labels[i] == Int32(source) {
            labels[i] = Int32(target)
        }
        let merged = applyMasks(labels: labels, width: masks.width, height: masks.height, existingColors: masks.colors)
        return removeCells(masks: merged, indices: [source])
    }

    static func addMaskFromStrokes(
        masks: MaskData?,
        imageWidth: Int,
        imageHeight: Int,
        strokes: [[[Double]]],
        classID: Int32,
        color: [UInt8]? = nil
    ) -> MaskData? {
        var labels = masks?.labels ?? [Int32](repeating: 0, count: imageWidth * imageHeight)
        let width = masks?.width ?? imageWidth
        let height = masks?.height ?? imageHeight
        if labels.count != width * height {
            labels = [Int32](repeating: 0, count: width * height)
        }

        var colors = masks?.colors ?? []
        let fillColor = color ?? colorForClass(classID: classID)
        var filledRows: [Int] = []
        var filledCols: [Int] = []
        var outlineRows: [Int] = []
        var outlineCols: [Int] = []

        for stroke in strokes where stroke.count >= 3 {
            let points = stroke.map { Point(x: Int($0[2]), y: Int($0[1])) }
            let minX = points.map(\.x).min() ?? 0
            let maxX = points.map(\.x).max() ?? 0
            let minY = points.map(\.y).min() ?? 0
            let maxY = points.map(\.y).max() ?? 0
            let localWidth = maxX - minX + 5
            let localHeight = maxY - minY + 5
            var bitmap = [Bool](repeating: false, count: localWidth * localHeight)
            fillPolygon(&bitmap, width: localWidth, height: localHeight, points: points, offsetX: minX - 2, offsetY: minY - 2)

            var areaRows: [Int] = []
            var areaCols: [Int] = []
            for y in 0 ..< localHeight {
                for x in 0 ..< localWidth where bitmap[y * localWidth + x] {
                    let yr = y + minY - 2
                    let xc = x + minX - 2
                    guard yr >= 0, xc >= 0, yr < height, xc < width else { continue }
                    areaRows.append(yr)
                    areaCols.append(xc)
                }
            }
            guard areaRows.count >= 10 else { return nil }
            let overlap = zip(areaRows, areaCols).filter { labels[$0 * width + $1] > 0 }.count
            if overlap > 0, areaRows.count - overlap < 10 { return nil }
        }

        let nextLabel = (labels.max() ?? 0) + 1
        for stroke in strokes where stroke.count >= 3 {
            let points = stroke.map { Point(x: Int($0[2]), y: Int($0[1])) }
            let minX = points.map(\.x).min() ?? 0
            let maxX = points.map(\.x).max() ?? 0
            let minY = points.map(\.y).min() ?? 0
            let maxY = points.map(\.y).max() ?? 0
            let localWidth = maxX - minX + 5
            let localHeight = maxY - minY + 5
            var bitmap = [Bool](repeating: false, count: localWidth * localHeight)
            fillPolygon(&bitmap, width: localWidth, height: localHeight, points: points, offsetX: minX - 2, offsetY: minY - 2)
            traceBoundary(bitmap, width: localWidth, height: localHeight, offsetX: minX - 2, offsetY: minY - 2, rows: &outlineRows, cols: &outlineCols)
            for y in 0 ..< localHeight {
                for x in 0 ..< localWidth where bitmap[y * localWidth + x] {
                    let yr = y + minY - 2
                    let xc = x + minX - 2
                    guard yr >= 0, xc >= 0, yr < height, xc < width else { continue }
                    guard labels[yr * width + xc] == 0 else { continue }
                    labels[yr * width + xc] = nextLabel
                    filledRows.append(yr)
                    filledCols.append(xc)
                }
            }
        }

        guard !filledRows.isEmpty else { return nil }
        for (row, col) in zip(outlineRows, outlineCols) where row >= 0 && col >= 0 && row < height && col < width {
            labels[row * width + col] = nextLabel
        }
        colors.append(fillColor)
        return applyMasks(labels: labels, width: width, height: height, existingColors: colors)
    }

    private static func renumber(_ labels: inout [Int32]) {
        let max = Int(labels.max() ?? 0)
        guard max > 0 else { return }
        var present = [Bool](repeating: false, count: max + 1)
        for label in labels where label > 0 {
            present[Int(label)] = true
        }
        var next = 1
        var remap = [Int](repeating: 0, count: max + 1)
        for i in 1 ... max where present[i] {
            remap[i] = next
            next += 1
        }
        for i in labels.indices where labels[i] > 0 {
            labels[i] = Int32(remap[Int(labels[i])])
        }
    }

    private struct Point {
        let x: Int
        let y: Int
    }

    private struct SeededRandomNumberGenerator: RandomNumberGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1
            return state
        }
    }

    private static func fillPolygon(
        _ bitmap: inout [Bool],
        width: Int,
        height: Int,
        points: [Point],
        offsetX: Int,
        offsetY: Int
    ) {
        let minY = points.map(\.y).min() ?? 0
        let maxY = points.map(\.y).max() ?? 0
        for y in minY ... maxY {
            var intersections: [Int] = []
            for i in points.indices {
                let p1 = points[i]
                let p2 = points[(i + 1) % points.count]
                if p1.y == p2.y { continue }
                if y < min(p1.y, p2.y) || y >= max(p1.y, p2.y) { continue }
                let x = p1.x + Int((Double(y - p1.y) / Double(p2.y - p1.y)) * Double(p2.x - p1.x))
                intersections.append(x)
            }
            intersections.sort()
            for i in stride(from: 0, to: max(intersections.count - 1, 0), by: 2) {
                let xStart = max(intersections[i], points.map(\.x).min() ?? 0)
                let xEnd = min(intersections[i + 1], points.map(\.x).max() ?? 0)
                for x in xStart ... xEnd {
                    let lx = x - offsetX
                    let ly = y - offsetY
                    guard lx >= 0, ly >= 0, lx < width, ly < height else { continue }
                    bitmap[ly * width + lx] = true
                }
            }
        }
    }

    private static func traceBoundary(
        _ bitmap: [Bool],
        width: Int,
        height: Int,
        offsetX: Int,
        offsetY: Int,
        rows: inout [Int],
        cols: inout [Int]
    ) {
        for y in 0 ..< height {
            for x in 0 ..< width where bitmap[y * width + x] {
                let onEdge = x == 0 || y == 0 || x + 1 == width || y + 1 == height
                    || !bitmap[y * width + x - 1]
                    || !bitmap[y * width + x + 1]
                    || !bitmap[(y - 1) * width + x]
                    || !bitmap[(y + 1) * width + x]
                guard onEdge else { continue }
                rows.append(y + offsetY)
                cols.append(x + offsetX)
            }
        }
    }

    static func cellBounds(
        labels: [Int32],
        width: Int,
        height: Int,
        label: Int32,
        margin: Int = 1
    ) -> (x0: Int, y0: Int, x1: Int, y1: Int)? {
        guard label > 0 else { return nil }

        var minY = height
        var minX = width
        var maxY = -1
        var maxX = -1
        for y in 0 ..< height {
            for x in 0 ..< width where labels[y * width + x] == label {
                minY = min(minY, y)
                minX = min(minX, x)
                maxY = max(maxY, y)
                maxX = max(maxX, x)
            }
        }

        guard maxY >= 0 else { return nil }
        return (
            max(0, minX - margin),
            max(0, minY - margin),
            min(width - 1, maxX + margin),
            min(height - 1, maxY + margin)
        )
    }

    static func cellsFullyInRect(
        labels: [Int32],
        width: Int,
        height: Int,
        x0: Int,
        y0: Int,
        x1: Int,
        y1: Int,
        filterClassID: Int32? = nil,
        classIDs: [Int32]? = nil
    ) -> [Int32] {
        let left = max(0, min(min(x0, x1), width - 1))
        let right = min(width, max(x0, x1))
        let top = max(0, min(min(y0, y1), height - 1))
        let bottom = min(height, max(y0, y1))
        guard right > left, bottom > top else { return [] }

        var candidates = Set<Int32>()
        for y in top ..< bottom {
            for x in left ..< right {
                let label = labels[y * width + x]
                if label > 0 {
                    candidates.insert(label)
                }
            }
        }

        var fullyCovered: [Int32] = []
        for label in candidates.sorted() {
            var inside = true
            for y in 0 ..< height {
                for x in 0 ..< width where labels[y * width + x] == label {
                    if y < top || y >= bottom || x < left || x >= right {
                        inside = false
                        break
                    }
                }
                if !inside { break }
            }
            guard inside else { continue }

            if let filterClassID, let classIDs {
                let row = Int(label) - 1
                if row < 0 || row >= classIDs.count || classIDs[row] != filterClassID {
                    continue
                }
            }

            fullyCovered.append(label)
        }

        return fullyCovered
    }
}
