import Foundation

struct SegNpyArray {
    let descr: String
    let shape: [Int]
    let data: Data
}

enum SegNpyIO {
    static func read(path: String) throws -> [String: Any] {
        let pickle = try extractPicklePayload(Data(contentsOf: URL(fileURLWithPath: path)))
        return try PickleReader.readDict(pickle)
    }

    static func write(path: String, payload: [String: Any]) throws {
        let pickle = PickleWriter.writeDict(payload)
        try writeNpyObjectFile(path: path, pickle: pickle)
    }

    static func readLabels(_ array: SegNpyArray) throws -> [Int32] {
        let count = elementCount(array.shape)
        var labels = [Int32]()
        labels.reserveCapacity(count)
        let descr = normalizeDescr(array.descr)
        if descr == "u2" {
            array.data.withUnsafeBytes { buffer in
                let values = buffer.bindMemory(to: UInt16.self)
                for value in values {
                    labels.append(Int32(value))
                }
            }
            return labels
        }
        if descr == "u4" {
            array.data.withUnsafeBytes { buffer in
                let values = buffer.bindMemory(to: UInt32.self)
                for value in values {
                    labels.append(Int32(value))
                }
            }
            return labels
        }
        throw SidecarError.serverError("Unsupported mask dtype: \(array.descr)")
    }

    static func readColors(_ array: SegNpyArray) -> [[UInt8]] {
        guard array.shape.count == 2, array.shape[1] == 3 else { return [] }
        let count = array.shape[0]
        return (0 ..< count).map { row in
            let offset = row * 3
            return [array.data[offset], array.data[offset + 1], array.data[offset + 2]]
        }
    }

    static func toArrayPayload(_ array: SegNpyArray) throws -> ArrayPayload {
        try ArrayCodec.encodeRaw(array.data, dtype: descrToPayloadDtype(array.descr), shape: array.shape)
    }

    static func fromLabels(_ labels: [Int32], width: Int, height: Int) -> SegNpyArray {
        var raw = Data(capacity: labels.count * MemoryLayout<UInt16>.size)
        for label in labels {
            var value = UInt16(clamping: Int(label)).littleEndian
            raw.append(Data(bytes: &value, count: MemoryLayout<UInt16>.size))
        }
        return SegNpyArray(descr: "u2", shape: [height, width], data: raw)
    }

    static func fromColors(_ colors: [[UInt8]]) -> SegNpyArray {
        var raw = Data(capacity: colors.count * 3)
        for color in colors {
            raw.append(contentsOf: color.prefix(3))
        }
        return SegNpyArray(descr: "u1", shape: [colors.count, 3], data: raw)
    }

    static func fromArrayPayload(_ payload: ArrayPayload) throws -> SegNpyArray {
        SegNpyArray(
            descr: payloadDtypeToDescr(payload.dtype),
            shape: payload.shape,
            data: try ArrayCodec.decode(payload)
        )
    }

    private static func extractPicklePayload(_ file: Data) throws -> Data {
        guard file.count >= 10, file[0] == 0x93 else {
            throw SidecarError.serverError("Invalid _seg.npy file: missing NumPy magic")
        }
        guard let headerEnd = file.firstIndex(of: 0x0A), headerEnd >= 8 else {
            throw SidecarError.serverError("Invalid _seg.npy file: malformed header")
        }
        let header = String(decoding: file[..<headerEnd], as: ASCII.self)
        guard header.contains("|O") else {
            throw SidecarError.serverError("Invalid _seg.npy file: expected pickled object array")
        }
        return file[(headerEnd + 1)...]
    }

    private static func writeNpyObjectFile(path: String, pickle: Data) throws {
        let headerText = "{'descr': '|O', 'fortran_order': False, 'shape': (), }"
        let pad = (16 - ((10 + headerText.count + 1) % 16)) % 16
        let header = headerText + String(repeating: " ", count: pad) + "\n"
        var file = Data([0x93])
        file.append(contentsOf: "NUMPY".utf8)
        file.append(contentsOf: [1, 0])
        var headerLength = UInt16(header.utf8.count).littleEndian
        file.append(Data(bytes: &headerLength, count: MemoryLayout<UInt16>.size))
        file.append(contentsOf: header.utf8)
        file.append(pickle)
        try file.write(to: URL(fileURLWithPath: path))
    }

    private static func elementCount(_ shape: [Int]) -> Int {
        shape.reduce(1, *)
    }

    private static func normalizeDescr(_ descr: String) -> String {
        descr.trimmingCharacters(in: CharacterSet(charactersIn: "<>|"))
    }

    private static func descrToPayloadDtype(_ descr: String) -> String {
        switch normalizeDescr(descr) {
        case "u1": "uint8"
        case "u2": "uint16"
        case "u4": "uint32"
        case "i4": "int32"
        case "f4": "float32"
        case "f8": "float64"
        case "b1": "uint8"
        default: descr
        }
    }

    private static func payloadDtypeToDescr(_ dtype: String) -> String {
        switch dtype {
        case "uint8": "u1"
        case "uint16": "u2"
        case "uint32": "u4"
        case "int32": "i4"
        case "float32": "f4"
        case "float64": "f8"
        case "bool": "b1"
        default: dtype
        }
    }

    private final class NumpyArrayBuilder {
        var shape: [Int] = []
        var descr: String = ""
        var data: Data = Data()

        func finish() -> SegNpyArray {
            SegNpyArray(descr: descr, shape: shape, data: data)
        }
    }

    private enum PickleReader {
        static func readDict(_ data: Data) throws -> [String: Any] {
            var reader = Reader(data: data)
            try reader.parse()
            guard let dict = reader.segDict else {
                throw SidecarError.serverError("Invalid _seg.npy file: missing segmentation dict")
            }
            return dict
        }

        private struct Reader {
            let data: Data
            var index: Int = 0
            var stack: [Any] = []
            var memo: [Any] = []
            var marks: [Int] = []
            var segDict: [String: Any]?

            init(data: Data) {
                self.data = data
            }

            mutating func parse() throws {
                while index < data.count {
                    let op = readByte()
                    if op == 0x2E { break }
                    try dispatch(op)
                }
            }

            mutating func dispatch(_ op: UInt8) throws {
                switch op {
                case 0x80:
                    index += 1
                case 0x95:
                    index += 8
                case 0x8C:
                    stack.append(readShortUnicode())
                case 0x94:
                    memo.append(stack.last as Any)
                case 0x93:
                    let name = stack.removeLast() as! String
                    let module = stack.removeLast() as! String
                    stack.append("\(module).\(name)")
                case UInt8(ascii: "c"):
                    let module = readLine()
                    let name = readLine()
                    stack.append("\(module).\(name)")
                case UInt8(ascii: "h"):
                    stack.append(memo[Int(readByte())])
                case UInt8(ascii: "K"):
                    stack.append(Int64(readByte()))
                case UInt8(ascii: "J"):
                    stack.append(readInt32())
                case UInt8(ascii: "G"):
                    stack.append(readFloat64())
                case UInt8(ascii: "N"):
                    stack.append(Optional<Any>.none as Any)
                case 0x89:
                    stack.append(false)
                case 0x88:
                    stack.append(true)
                case 0x85:
                    stack.append([stack.removeLast()])
                case 0x86:
                    let second = stack.removeLast()
                    let first = stack.removeLast()
                    stack.append([first, second])
                case 0x87:
                    let third = stack.removeLast()
                    let second = stack.removeLast()
                    let first = stack.removeLast()
                    stack.append([first, second, third])
                case UInt8(ascii: "("):
                    marks.append(stack.count)
                case UInt8(ascii: "t"):
                    stack.append(popMark())
                case UInt8(ascii: "R"):
                    let args = stack.removeLast()
                    let func = stack.removeLast() as! String
                    if func.contains("ndarray") || func.contains("_reconstruct") {
                        stack.append(NumpyArrayBuilder())
                    } else {
                        stack.append(args)
                    }
                case UInt8(ascii: "b"):
                    let state = stack.removeLast()
                    let target = stack.removeLast()
                    stack.append(applyBuild(target: target, state: state))
                case UInt8(ascii: "}"):
                    stack.append([String: Any]())
                case UInt8(ascii: "]"):
                    stack.append([Any]())
                case UInt8(ascii: "u"):
                    let items = popMark()
                    guard var dict = stack.last as? [String: Any] else {
                        throw SidecarError.serverError("Invalid _seg.npy pickle dict")
                    }
                    var index = 0
                    while index + 1 < items.count {
                        dict[items[index] as! String] = items[index + 1]
                        index += 2
                    }
                    stack[stack.count - 1] = dict
                    if dict["masks"] != nil {
                        segDict = dict
                    }
                case UInt8(ascii: "e"):
                    let items = popMark()
                    guard var list = stack.last as? [Any] else {
                        throw SidecarError.serverError("Invalid _seg.npy pickle list")
                    }
                    list.append(contentsOf: items)
                    stack[stack.count - 1] = list
                case UInt8(ascii: "C"):
                    stack.append(readShortBytes())
                default:
                    throw SidecarError.serverError(String(format: "Unsupported pickle opcode 0x%02X", op))
                }
            }

            mutating func applyBuild(target: Any, state: Any) -> Any {
                if var builder = target as? NumpyArrayBuilder {
                    if let bytes = state as? Data {
                        builder.data = bytes
                        return builder.finish()
                    }
                    if let tuple = state as? [Any] {
                        if let raw = tuple.last as? Data {
                            builder.data = raw
                            return builder.finish()
                        }
                        for item in tuple {
                            if let shape = item as? [Any], shape.allSatisfy({ $0 is Int64 || $0 is Int }) {
                                builder.shape = shape.map { value in
                                    if let intValue = value as? Int { return intValue }
                                    if let int64Value = value as? Int64 { return Int(int64Value) }
                                    return 0
                                }
                            } else if let descr = item as? String, ["u1", "u2", "u4", "i4", "f4", "f8", "b1"].contains(descr) {
                                builder.descr = descr
                            }
                        }
                    }
                    return builder
                }
                return state
            }

            mutating func popMark() -> [Any] {
                let start = marks.removeLast()
                let items = Array(stack[start...])
                stack.removeSubrange(start...)
                return items
            }

            mutating func readByte() -> UInt8 {
                defer { index += 1 }
                return data[index]
            }

            mutating func readBytes(_ count: Int) -> Data {
                defer { index += count }
                return data[index ..< index + count]
            }

            mutating func readShortUnicode() -> String {
                let length = Int(readByte())
                return String(decoding: readBytes(length), as: UTF8.self)
            }

            mutating func readShortBytes() -> Data {
                readBytes(Int(readByte()))
            }

            mutating func readInt32() -> Int32 {
                readBytes(4).withUnsafeBytes { $0.load(as: Int32.self) }
            }

            mutating func readFloat64() -> Double {
                readBytes(8).withUnsafeBytes { $0.load(as: Double.self) }
            }

            mutating func readLine() -> String {
                let start = index
                while index < data.count, data[index] != 0x0A {
                    index += 1
                }
                let text = String(decoding: data[start ..< index], as: ASCII.self)
                if index < data.count { index += 1 }
                return text
            }
        }
    }

    private enum PickleWriter {
        static func writeDict(_ payload: [String: Any]) -> Data {
            var writer = Writer()
            writer.writePayload(payload)
            return writer.data
        }

        private struct Writer {
            var data = Data()

            mutating func writePayload(_ payload: [String: Any]) {
                writeRaw([0x80, 4, 0x95])
                let frameLengthPosition = data.count
                data.append(Data(count: 8))
                let frameStart = data.count
                writeOuterWrapper(payload)
                writeRaw([0x2E])
                let frameLength = UInt64(data.count - frameStart).littleEndian
                withUnsafeBytes(of: frameLength) { bytes in
                    data.replaceSubrange(frameLengthPosition ..< frameLengthPosition + 8, with: bytes)
                }
            }

            mutating func writeOuterWrapper(_ payload: [String: Any]) {
                writeArrayConstructor()
                writeUnicode("dtype")
                writeMemoGet(10)
                writeUnicode("O8")
                writeRaw([0x89, 0x88, 0x87, 0x52])
                memoize()
                writeRaw([0x28, 0x4B, 3])
                writeUnicode("|")
                writeRaw([0x4E, 0x4E, 0x4E, 0x4A])
                writeInt32(-1)
                writeRaw([0x4A])
                writeInt32(-1)
                writeRaw([0x4B, 63, 0x74, 0x62, 0x89, 0x5D])
                memoize()
                writeRaw([0x7D])
                memoize()
                writeRaw([0x28])
                for (key, value) in payload {
                    writeUnicode(key)
                    writeValue(value)
                }
                writeRaw([0x75, 0x61, 0x74, 0x62])
            }

            mutating func writeValue(_ value: Any) {
                switch value {
                case is NSNull:
                    writeRaw([0x4E])
                case let string as String:
                    writeUnicode(string)
                case let boolean as Bool:
                    writeRaw([boolean ? 0x88 : 0x89])
                case let number as Int:
                    writeRaw([0x4B, UInt8(number)])
                case let number as Double:
                    writeRaw([0x47])
                    var bits = number.bitPattern.littleEndian
                    data.append(Data(bytes: &bits, count: MemoryLayout<UInt64>.size))
                case let array as SegNpyArray:
                    writeArray(array)
                case let list as [Any]:
                    writeRaw([0x5D])
                    memoize()
                    writeRaw([0x28])
                    for item in list {
                        writeValue(item)
                    }
                    writeRaw([0x65])
                default:
                    break
                }
            }

            mutating func writeArray(_ array: SegNpyArray) {
                writeArrayConstructor()
                writeRaw([0x4B, 1])
                writeShape(array.shape)
                writeMemoGet(11)
                writeUnicode(payloadDtypeToDescr(descrToPayloadDtype(array.descr)))
                writeRaw([0x89, 0x88, 0x87, 0x52])
                memoize()
                writeRaw([0x28, 0x4B, 3])
                writeUnicode("<")
                writeRaw([0x4E, 0x4E, 0x4E, 0x4A])
                writeInt32(-1)
                writeRaw([0x4A])
                writeInt32(-1)
                writeRaw([0x4B, 0, 0x74, 0x62, 0x89, 0x43, UInt8(array.data.count)])
                data.append(array.data)
                memoize()
                writeRaw([0x74, 0x62])
            }

            mutating func writeShape(_ shape: [Int]) {
                if shape.count == 1 {
                    writeRaw([0x4B, UInt8(shape[0]), 0x85])
                } else if shape.count == 2 {
                    writeRaw([0x4B, UInt8(shape[0]), 0x4B, UInt8(shape[1]), 0x86])
                } else {
                    writeRaw([0x28])
                    for dim in shape {
                        writeRaw([0x4B, UInt8(dim)])
                    }
                    writeRaw([0x74])
                }
            }

            mutating func writeArrayConstructor() {
                writeUnicode("numpy._core.multiarray")
                memoize()
                writeUnicode("_reconstruct")
                memoize()
                writeMemoGet(0)
                writeMemoGet(1)
                writeRaw([0x93])
                memoize()
                writeUnicode("numpy")
                memoize()
                writeUnicode("ndarray")
                memoize()
                writeMemoGet(3)
                writeMemoGet(4)
                writeRaw([0x93])
                memoize()
                writeRaw([0x85, 0x4B, 0, 0x85])
                memoize()
                writeRaw([0x43, 1, 0x62])
                memoize()
                writeMemoGet(5)
                writeMemoGet(6)
                writeMemoGet(7)
                writeRaw([0x87, 0x52])
                memoize()
                writeRaw([0x28, 0x4B, 1])
            }

            mutating func writeUnicode(_ text: String) {
                let bytes = Data(text.utf8)
                data.append(0x8C)
                data.append(UInt8(bytes.count))
                data.append(bytes)
                memoize()
            }

            mutating func memoize() {
                data.append(0x94)
            }

            mutating func writeMemoGet(_ index: Int) {
                data.append(contentsOf: [0x68, UInt8(index)])
            }

            mutating func writeInt32(_ value: Int32) {
                var little = value.littleEndian
                data.append(Data(bytes: &little, count: 4))
            }

            mutating func writeRaw(_ bytes: [UInt8]) {
                data.append(contentsOf: bytes)
            }
        }
    }
}

private extension Array where Element == Any {
    func allSatisfy(_ predicate: (Any) -> Bool) -> Bool {
        !contains { !predicate($0) }
    }
}
