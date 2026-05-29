using CellposeGUI.Services;
using Xunit;

namespace CellposeGUI.Tests;

public sealed class SegNpyIOTests
{
    private static string FixturePath(string name)
    {
        var projectDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        return Path.GetFullPath(Path.Combine(projectDir, "..", "..", "tests", "fixtures", "seg_npy", name));
    }

    public static IEnumerable<object[]> FixtureCases =>
    [
        ["minimal_seg.npy", 2, "image.tif"],
        ["legacy_gui_seg.npy", 2, "legacy_image.tif"],
        ["with_instance_classes_seg.npy", 2, "image.tif"],
    ];

    [Theory]
    [MemberData(nameof(FixtureCases))]
    public void ReadCommittedFixtures(string fixtureName, int expectedNcells, string expectedSourceSuffix)
    {
        var path = FixturePath(fixtureName);
        Assert.True(File.Exists(path), $"missing fixture: {path}");

        var payload = SegNpyIO.Read(path);
        var filename = payload.GetValueOrDefault("filename")?.ToString() ?? "";
        Assert.EndsWith(expectedSourceSuffix, filename);

        Assert.IsType<SegNpyArray>(payload.GetValueOrDefault("masks"));
        var masks = (SegNpyArray)payload["masks"]!;
        Assert.Equal(expectedNcells, MaxLabel(masks));
    }

    [Fact]
    public void MinimalFixtureHasFlows()
    {
        var payload = SegNpyIO.Read(FixturePath("minimal_seg.npy"));
        Assert.IsType<List<object?>>(payload.GetValueOrDefault("flows"));
        var flows = (List<object?>)payload["flows"]!;
        Assert.Equal(2, flows.Count);
        Assert.All(flows, item => Assert.IsType<SegNpyArray>(item));
    }

    [Fact]
    public void InstanceClassesFixture()
    {
        var payload = SegNpyIO.Read(FixturePath("with_instance_classes_seg.npy"));
        Assert.IsType<SegNpyArray>(payload.GetValueOrDefault("instance_classes"));
        var classes = SegNpyIO.ReadLabels((SegNpyArray)payload["instance_classes"]!);
        Assert.Equal([1, 2], classes);
    }

    [Fact]
    public void LegacyFixtureHasColors()
    {
        var payload = SegNpyIO.Read(FixturePath("legacy_gui_seg.npy"));
        Assert.IsType<SegNpyArray>(payload.GetValueOrDefault("colors"));
        var colors = SegNpyIO.ReadColors((SegNpyArray)payload["colors"]!);
        Assert.Equal(2, colors.Length);
        Assert.Equal([100, 150, 200], colors[0]);
    }

    private static int MaxLabel(SegNpyArray masks)
    {
        var labels = SegNpyIO.ReadLabels(masks);
        return labels.Length == 0 ? 0 : labels.Max();
    }
}
