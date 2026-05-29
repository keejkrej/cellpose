using System.Buffers.Binary;
using System.Text;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class SegNpyArray
{
    public string Descr { get; init; } = "";
    public int[] Shape { get; init; } = [];
    public byte[] Data { get; init; } = [];
}

public static class SegNpyIO
{
    public static Dictionary<string, object?> Read(string path)
    {
        var pickle = ExtractPicklePayload(File.ReadAllBytes(path));
        return PickleReader.ReadDict(pickle);
    }

    public static void Write(string path, Dictionary<string, object?> payload)
    {
        var pickle = PickleWriter.WriteDict(payload);
        WriteNpyObjectFile(path, pickle);
    }

    public static int[] ReadLabels(SegNpyArray array)
    {
        var count = ElementCount(array.Shape);
        var labels = new int[count];
        var descr = NormalizeDescr(array.Descr);
        if (descr is "u2")
        {
            for (var i = 0; i < count; i++)
                labels[i] = BinaryPrimitives.ReadUInt16LittleEndian(array.Data.AsSpan(i * 2));
            return labels;
        }

        if (descr is "u4")
        {
            for (var i = 0; i < count; i++)
                labels[i] = (int)BinaryPrimitives.ReadUInt32LittleEndian(array.Data.AsSpan(i * 4));
            return labels;
        }

        if (descr is "i4")
        {
            Buffer.BlockCopy(array.Data, 0, labels, 0, Math.Min(array.Data.Length, labels.Length * 4));
            return labels;
        }

        throw new SidecarException($"Unsupported mask dtype: {array.Descr}");
    }

    public static byte[][] ReadColors(SegNpyArray array)
    {
        if (array.Shape.Length != 2 || array.Shape[1] != 3)
            return [];
        var count = array.Shape[0];
        var colors = new byte[count][];
        for (var row = 0; row < count; row++)
            colors[row] = [array.Data[row * 3], array.Data[row * 3 + 1], array.Data[row * 3 + 2]];
        return colors;
    }

    public static ArrayPayload ToArrayPayload(SegNpyArray array)
    {
        var dtype = DescrToPayloadDtype(array.Descr);
        return ArrayCodec.EncodeRaw(array.Data, dtype, array.Shape);
    }

    public static SegNpyArray FromLabels(int[] labels, int width, int height)
    {
        var raw = new byte[labels.Length * sizeof(ushort)];
        for (var i = 0; i < labels.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(i * sizeof(ushort)), (ushort)Math.Clamp(labels[i], 0, ushort.MaxValue));
        return new SegNpyArray { Descr = "u2", Shape = [height, width], Data = raw };
    }

    public static SegNpyArray FromColors(byte[][] colors)
    {
        var raw = new byte[colors.Length * 3];
        for (var i = 0; i < colors.Length; i++)
        {
            raw[i * 3] = colors[i][0];
            raw[i * 3 + 1] = colors[i][1];
            raw[i * 3 + 2] = colors[i][2];
        }
        return new SegNpyArray { Descr = "u1", Shape = [colors.Length, 3], Data = raw };
    }

    public static SegNpyArray FromArrayPayload(ArrayPayload payload) =>
        new()
        {
            Descr = PayloadDtypeToDescr(payload.Dtype),
            Shape = payload.Shape,
            Data = ArrayCodec.Decode(payload),
        };

    private static byte[] ExtractPicklePayload(byte[] file)
    {
        if (file.Length < 10 || file[0] != 0x93)
            throw new SidecarException("Invalid _seg.npy file: missing NumPy magic");

        var headerEnd = Array.IndexOf(file, (byte)'\n', 8);
        if (headerEnd < 0)
            throw new SidecarException("Invalid _seg.npy file: malformed header");

        var header = Encoding.ASCII.GetString(file, 0, headerEnd + 1);
        if (!header.Contains("|O", StringComparison.Ordinal))
            throw new SidecarException("Invalid _seg.npy file: expected pickled object array");

        return file[(headerEnd + 1)..];
    }

    private static void WriteNpyObjectFile(string path, byte[] pickle)
    {
        var headerText = "{'descr': '|O', 'fortran_order': False, 'shape': (), }";
        var pad = (16 - ((10 + headerText.Length + 1) % 16)) % 16;
        var headerBytes = Encoding.ASCII.GetBytes(headerText + new string(' ', pad) + "\n");

        using var stream = File.Create(path);
        stream.WriteByte(0x93);
        stream.Write(Encoding.ASCII.GetBytes("NUMPY"));
        stream.WriteByte(1);
        stream.WriteByte(0);
        Span<byte> len = stackalloc byte[2];
        BinaryPrimitives.WriteUInt16LittleEndian(len, (ushort)headerBytes.Length);
        stream.Write(len);
        stream.Write(headerBytes);
        stream.Write(pickle);
    }

    private static int ElementCount(int[] shape) =>
        shape.Aggregate(1, (left, right) => left * right);

    private static string NormalizeDescr(string descr) =>
        descr.TrimStart('<', '|', '>');

    private static string DescrToPayloadDtype(string descr) => NormalizeDescr(descr) switch
    {
        "u1" => "uint8",
        "u2" => "uint16",
        "u4" => "uint32",
        "i4" => "int32",
        "f4" => "float32",
        "f8" => "float64",
        "b1" => "uint8",
        _ => throw new SidecarException($"Unsupported dtype: {descr}"),
    };

    private static string PayloadDtypeToDescr(string dtype) => dtype switch
    {
        "uint8" => "u1",
        "uint16" => "u2",
        "uint32" => "u4",
        "int32" => "i4",
        "float32" => "f4",
        "float64" => "f8",
        "bool" => "b1",
        _ => dtype,
    };

    private sealed class NumpyArrayBuilder
    {
        public int[] Shape = [];
        public string Descr = "";
        public byte[] Data = [];

        public SegNpyArray Finish() =>
            new() { Shape = Shape, Descr = Descr, Data = Data };
    }

    private sealed class PickleReader
    {
        private readonly byte[] _data;
        private int _index;
        private readonly Stack<object?> _stack = new();
        private readonly List<object?> _memo = [];
        private readonly Stack<int> _marks = new();
        private Dictionary<string, object?>? _segDict;

        private PickleReader(byte[] data) => _data = data;

        public static Dictionary<string, object?> ReadDict(byte[] data)
        {
            var reader = new PickleReader(data);
            reader.Parse();
            if (reader._segDict == null)
                throw new SidecarException("Invalid _seg.npy file: missing segmentation dict");
            return reader._segDict;
        }

        private void Parse()
        {
            while (_index < _data.Length)
            {
                var op = ReadByte();
                if (op == (byte)'.')
                    break;
                Dispatch(op);
            }
        }

        private void Dispatch(byte op)
        {
            switch (op)
            {
                case 0x80:
                    _index++;
                    break;
                case 0x95:
                    _index += 8;
                    break;
                case 0x8c:
                    _stack.Push(ReadShortUnicode());
                    break;
                case 0x94:
                    _memo.Add(_stack.Peek());
                    break;
                case 0x93:
                {
                    var name = (string)_stack.Pop()!;
                    var module = (string)_stack.Pop()!;
                    _stack.Push($"{module}.{name}");
                    break;
                }
                case (byte)'c':
                {
                    var module = ReadLine();
                    var name = ReadLine();
                    _stack.Push($"{module}.{name}");
                    break;
                }
                case (byte)'c':
                {
                    var module = ReadLine();
                    var name = ReadLine();
                    _stack.Push($"{module}.{name}");
                    break;
                }
                case (byte)'h':
                    _stack.Push(_memo[ReadByte()]);
                    break;
                case (byte)'j':
                    _stack.Push(_memo[BinaryPrimitives.ReadInt32LittleEndian(ReadBytes(4))]);
                    break;
                case (byte)'K':
                    _stack.Push((long)ReadByte());
                    break;
                case (byte)'J':
                    _stack.Push(BinaryPrimitives.ReadInt32LittleEndian(ReadBytes(4)));
                    break;
                case (byte)'G':
                    _stack.Push(BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(ReadBytes(8))));
                    break;
                case (byte)'N':
                    _stack.Push(null);
                    break;
                case 0x89:
                    _stack.Push(false);
                    break;
                case 0x88:
                    _stack.Push(true);
                    break;
                case 0x85:
                    _stack.Push(new object?[] { _stack.Pop() });
                    break;
                case 0x86:
                {
                    var second = _stack.Pop();
                    var first = _stack.Pop();
                    _stack.Push(new object?[] { first, second });
                    break;
                }
                case 0x87:
                {
                    var third = _stack.Pop();
                    var second = _stack.Pop();
                    var first = _stack.Pop();
                    _stack.Push(new object?[] { first, second, third });
                    break;
                }
                case (byte)'(':
                    _marks.Push(_stack.Count);
                    break;
                case (byte)'t':
                    _stack.Push(PopMark());
                    break;
                case (byte)'R':
                {
                    var args = _stack.Pop();
                    var func = _stack.Pop()?.ToString() ?? "";
                    if (func.Contains("ndarray", StringComparison.Ordinal) ||
                        func.Contains("_reconstruct", StringComparison.Ordinal))
                        _stack.Push(new NumpyArrayBuilder());
                    else if (func.Contains("dtype", StringComparison.Ordinal))
                        _stack.Push(args);
                    else
                        _stack.Push(new ReduceFrame(func, args));
                    break;
                }
                case (byte)'b':
                {
                    var state = _stack.Pop();
                    var target = _stack.Pop();
                    _stack.Push(ApplyBuild(target, state));
                    break;
                }
                case (byte)'}':
                    _stack.Push(new Dictionary<string, object?>());
                    break;
                case (byte)']':
                    _stack.Push(new List<object?>());
                    break;
                case (byte)'u':
                {
                    var items = PopMark();
                    if (_stack.Peek() is not Dictionary<string, object?> dict)
                        throw new SidecarException("Invalid _seg.npy pickle dict");
                    for (var i = 0; i + 1 < items.Length; i += 2)
                        dict[(string)items[i]!] = items[i + 1];
                    if (dict.ContainsKey("masks"))
                        _segDict = dict;
                    break;
                }
                case (byte)'e':
                {
                    var items = PopMark();
                    if (_stack.Peek() is not List<object?> list)
                        throw new SidecarException("Invalid _seg.npy pickle list");
                    list.AddRange(items);
                    break;
                }
                case (byte)'a':
                {
                    var value = _stack.Pop();
                    if (_stack.Peek() is List<object?> list)
                        list.Add(value);
                    break;
                }
                case (byte)'C':
                    _stack.Push(ReadShortBytes());
                    break;
                case (byte)'B':
                    _stack.Push(ReadBinBytes());
                    break;
                case (byte')':
                    _stack.Push(Array.Empty<object?>());
                    break;
                default:
                    throw new SidecarException($"Unsupported pickle opcode 0x{op:X2} at {_index - 1}");
            }
        }

        private object? ApplyBuild(object? target, object? state)
        {
            if (target is NumpyArrayBuilder builder)
            {
                if (state is byte[] bytes)
                {
                    builder.Data = bytes;
                    return builder.Finish();
                }

                if (state is object?[] tuple)
                {
                    if (tuple.LastOrDefault() is byte[] raw)
                    {
                        builder.Data = raw;
                        return builder.Finish();
                    }

                    foreach (var item in tuple)
                    {
                        if (item is object?[] shape && shape.All(x => x is long or int))
                            builder.Shape = shape.Select(x => Convert.ToInt32(x)).ToArray();
                        else if (item is string descr && descr is "u1" or "u2" or "u4" or "i4" or "f4" or "f8" or "b1")
                            builder.Descr = descr;
                        else if (item is object?[] { Length: >= 3 } dtypeTuple &&
                                 dtypeTuple.Any(x => x is string s && s is "u1" or "u2" or "u4" or "i4" or "f4" or "f8" or "b1"))
                        {
                            foreach (var x in dtypeTuple)
                            {
                                if (x is string s && s is "u1" or "u2" or "u4" or "i4" or "f4" or "f8" or "b1")
                                    builder.Descr = s;
                            }
                        }
                    }
                }

                return builder;
            }

            if (state is Dictionary<string, object?> dict)
                return dict;
            return state ?? target;
        }

        private object?[] PopMark()
        {
            var start = _marks.Pop();
            var count = _stack.Count - start;
            var items = new object?[count];
            for (var i = count - 1; i >= 0; i--)
                items[i] = _stack.Pop();
            return items;
        }

        private byte ReadByte() => _data[_index++];
        private byte[] ReadBytes(int count) => _data[_index..(_index += count)];
        private string ReadShortUnicode()
        {
            var length = ReadByte();
            return Encoding.UTF8.GetString(ReadBytes(length));
        }

        private byte[] ReadShortBytes()
        {
            var length = ReadByte();
            return ReadBytes(length);
        }

        private byte[] ReadBinBytes()
        {
            var length = BinaryPrimitives.ReadInt32LittleEndian(ReadBytes(4));
            return ReadBytes(length);
        }

        private string ReadLine()
        {
            var start = _index;
            while (_index < _data.Length && _data[_index] != (byte)'\n')
                _index++;
            var text = Encoding.ASCII.GetString(_data, start, _index - start);
            _index++;
            return text;
        }

        private sealed record ReduceFrame(string Function, object? Args);
    }

    private sealed class PickleWriter
    {
        private readonly MemoryStream _stream = new();
        private readonly List<object?> _memo = [];

        public static byte[] WriteDict(Dictionary<string, object?> payload)
        {
            var writer = new PickleWriter();
            writer.WritePayload(payload);
            return writer._stream.ToArray();
        }

        private void WritePayload(Dictionary<string, object?> payload)
        {
            WriteRaw([0x80, 4]);
            WriteFrameStart();
            WriteOuterWrapper(payload);
            WriteRaw([(byte)'.']);
            PatchFrame();
        }

        private long _frameLenPos = -1;
        private long _frameStart = -1;

        private void WriteFrameStart()
        {
            WriteRaw([0x95]);
            _frameLenPos = _stream.Position;
            _stream.Write(new byte[8]);
            _frameStart = _stream.Position;
        }

        private void PatchFrame()
        {
            if (_frameLenPos < 0)
                return;
            var bytes = _stream.ToArray();
            BinaryPrimitives.WriteUInt64LittleEndian(
                bytes.AsSpan((int)_frameLenPos, 8),
                (ulong)(bytes.Length - _frameStart));
            _stream.SetLength(0);
            _stream.Write(bytes);
        }

        private void WriteOuterWrapper(Dictionary<string, object?> payload)
        {
            WriteArrayConstructor();
            WriteUnicode("dtype");
            WriteMemoGet(10);
            WriteUnicode("O8");
            WriteRaw([0x89, 0x88, 0x87, (byte)'R']);
            Memoize();
            WriteRaw([(byte)'(', (byte)'K', 3]);
            WriteUnicode("|");
            WriteRaw([(byte)'N', (byte)'N', (byte)'N', (byte)'J']);
            WriteInt32(-1);
            WriteRaw([(byte)'J']);
            WriteInt32(-1);
            WriteRaw([(byte)'K', 63, (byte)'t', (byte)'b', 0x89]);
            WriteRaw([(byte)']']);
            Memoize();
            WriteRaw([(byte)'}']);
            Memoize();
            WriteRaw([(byte)'(']);
            foreach (var pair in payload)
            {
                WriteUnicode(pair.Key);
                WriteValue(pair.Value);
            }
            WriteRaw([(byte)'u', (byte)'a', (byte)'t', (byte)'b']);
        }

        private void WriteArrayConstructor()
        {
            WriteUnicode("numpy._core.multiarray");
            Memoize();
            WriteUnicode("_reconstruct");
            Memoize();
            WriteMemoGet(0);
            WriteMemoGet(1);
            WriteRaw([0x93]);
            Memoize();
            WriteUnicode("numpy");
            Memoize();
            WriteUnicode("ndarray");
            Memoize();
            WriteMemoGet(3);
            WriteMemoGet(4);
            WriteRaw([0x93]);
            Memoize();
            WriteRaw([0x85, (byte)'K', 0, 0x85]);
            Memoize();
            WriteRaw([(byte)'C', 1, (byte)'b']);
            Memoize();
            WriteMemoGet(5);
            WriteMemoGet(6);
            WriteMemoGet(7);
            WriteRaw([0x87, (byte)'R']);
            Memoize();
            WriteRaw([(byte)'(', (byte)'K', 1]);
        }

        private void WriteValue(object? value)
        {
            switch (value)
            {
                case null:
                    WriteRaw([(byte)'N']);
                    break;
                case string s:
                    WriteUnicode(s);
                    break;
                case bool b:
                    WriteRaw([b ? (byte)0x88 : (byte)0x89]);
                    break;
                case int i:
                    WriteRaw([(byte)'K', (byte)i]);
                    break;
                case long l when l is >= 0 and <= 255:
                    WriteRaw([(byte)'K', (byte)l]);
                    break;
                case double d:
                    WriteRaw([(byte)'G']);
                    Span<byte> buf = stackalloc byte[8];
                    BinaryPrimitives.WriteInt64LittleEndian(buf, BitConverter.DoubleToInt64Bits(d));
                    _stream.Write(buf);
                    break;
                case SegNpyArray array:
                    WriteArray(array);
                    break;
                case List<object?> list:
                    WriteRaw([(byte)']']);
                    Memoize();
                    WriteRaw([(byte)'(']);
                    foreach (var item in list)
                        WriteValue(item);
                    WriteRaw([(byte)'e']);
                    break;
                default:
                    throw new SidecarException($"Unsupported _seg.npy value type: {value.GetType().Name}");
            }
        }

        private void WriteArray(SegNpyArray array)
        {
            WriteArrayConstructor();
            WriteRaw([(byte)'K', 1]);
            WriteShape(array.Shape);
            WriteMemoGet(11);
            WriteUnicode(NormalizeDescr(PayloadDtypeToDescr(DescrToPayloadDtype(array.Descr))));
            WriteRaw([0x89, 0x88, 0x87, (byte)'R']);
            Memoize();
            WriteRaw([(byte)'(', (byte)'K', 3]);
            WriteUnicode("<");
            WriteRaw([(byte)'N', (byte)'N', (byte)'N', (byte)'J']);
            WriteInt32(-1);
            WriteRaw([(byte)'J']);
            WriteInt32(-1);
            WriteRaw([(byte)'K', 0, (byte)'t', (byte)'b', 0x89]);
            WriteRaw([(byte)'C', (byte)array.Data.Length]);
            _stream.Write(array.Data);
            Memoize();
            WriteRaw([(byte)'t', (byte)'b']);
        }

        private void WriteShape(int[] shape)
        {
            if (shape.Length == 1)
            {
                WriteRaw([(byte)'K', (byte)shape[0], 0x85]);
                return;
            }

            if (shape.Length == 2)
            {
                WriteRaw([(byte)'K', (byte)shape[0], (byte)'K', (byte)shape[1], 0x86]);
                return;
            }

            WriteRaw([(byte)'(']);
            foreach (var dim in shape)
                WriteRaw([(byte)'K', (byte)dim]);
            WriteRaw([(byte)'t']);
        }

        private void WriteUnicode(string text)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            WriteRaw([0x8c, (byte)bytes.Length]);
            _stream.Write(bytes);
            Memoize();
        }

        private void Memoize() => WriteRaw([0x94]);

        private void WriteMemoGet(int index) => WriteRaw([(byte)'h', (byte)index]);

        private void WriteInt32(int value)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(buf, value);
            _stream.Write(buf);
        }

        private void WriteRaw(byte[] bytes) => _stream.Write(bytes);
    }
}
