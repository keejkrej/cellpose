using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class SeriesDiscoveryService
{
    private static readonly string[] SupportedExtensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"];
    private static readonly string[] SupportedSubfolderTemplates =
        ["Pos{p}", "Position{p}", "Pos_{p}", "Position_{p}"];
    private static readonly string[] SupportedFilenameTemplates =
        ["img_{t}_{c}_{z}.jpg", "img_channel{c}_position{p}_time{t}_z{z}.tif"];

    public SeriesTemplateSuggestion SuggestTemplates(string folder)
    {
        if (!Directory.Exists(folder))
            throw new SidecarException($"Folder not found: {folder}");

        (string Subfolder, string Filename, int Count)? best = null;
        foreach (var subfolder in SupportedSubfolderTemplates)
        {
            foreach (var filename in SupportedFilenameTemplates)
            {
                var matches = CollectMatches(folder, subfolder, filename);
                if (matches.Count == 0)
                    continue;
                var score = matches.Count;
                if (best == null || score > best.Value.Count)
                    best = (subfolder, filename, score);
            }
        }

        if (best != null)
            return new SeriesTemplateSuggestion { SubfolderTemplate = best.Value.Subfolder, FilenameTemplate = best.Value.Filename };

        return new SeriesTemplateSuggestion
        {
            SubfolderTemplate = "",
            FilenameTemplate = "img_{t}_{c}_{z}.jpg",
        };
    }

    public SeriesDiscovery Discover(string folder, string subfolderTemplate, string filenameTemplate)
    {
        var matches = CollectMatches(folder, subfolderTemplate, filenameTemplate);
        if (matches.Count == 0)
            throw new SidecarException($"No files matched template in {folder}");

        matches = matches
            .OrderBy(m => SortKey(m.Position))
            .ThenBy(m => SortKey(m.Time))
            .ThenBy(m => SortKey(m.Channel))
            .ThenBy(m => SortKey(m.Z))
            .ThenBy(m => m.RelativePath)
            .ToList();

        var lookup = new Dictionary<string, int>();
        var records = new List<SeriesDiscoveryRecord>();
        var datasetRecords = new List<SeriesDatasetRecord>();
        for (var i = 0; i < matches.Count; i++)
        {
            var match = matches[i];
            var key = $"{match.Position}_{match.Time}_{match.Channel}_{match.Z}";
            if (lookup.ContainsKey(key))
                throw new SidecarException($"Duplicate series record for {key}");
            lookup[key] = i;
            records.Add(new SeriesDiscoveryRecord { Index = i, Label = match.RelativePath, Path = match.Path });
            datasetRecords.Add(new SeriesDatasetRecord
            {
                Label = match.RelativePath,
                Path = match.Path,
                Position = match.Position,
                Time = match.Time,
                Channel = match.Channel,
                Z = match.Z,
            });
        }

        var axes = new Dictionary<string, List<string>>
        {
            ["position"] = matches.Select(m => m.Position).Distinct().OrderBy(SortKey).ToList(),
            ["time"] = matches.Select(m => m.Time).Distinct().OrderBy(SortKey).ToList(),
            ["channel"] = matches.Select(m => m.Channel).Distinct().OrderBy(SortKey).ToList(),
            ["z"] = matches.Select(m => m.Z).Distinct().OrderBy(SortKey).ToList(),
        };
        var axisIndex = axes.ToDictionary(
            pair => pair.Key,
            pair => pair.Value.Select((value, index) => (value, index)).ToDictionary(v => v.value, v => v.index));

        return new SeriesDiscovery
        {
            Folder = folder,
            RecordCount = records.Count,
            Records = records,
            Axes = axes,
            Dataset = new SeriesDatasetPayload
            {
                Folder = folder,
                SubfolderTemplate = subfolderTemplate,
                FilenameTemplate = filenameTemplate,
                Axes = axes,
                AxisIndex = axisIndex,
                Lookup = lookup,
                Records = datasetRecords,
            },
        };
    }

    private static List<SeriesMatch> CollectMatches(string folder, string subfolderTemplate, string filenameTemplate)
    {
        var subfolderPattern = CompileTemplate(subfolderTemplate, allowEmpty: true);
        var filenamePattern = CompileTemplate(filenameTemplate);
        var matches = new List<SeriesMatch>();
        var candidateFolders = string.IsNullOrEmpty(subfolderTemplate)
            ? [new DirectoryInfo(folder)]
            : new DirectoryInfo(folder).GetDirectories().OrderBy(d => d.Name).ToArray();

        foreach (var candidate in candidateFolders)
        {
            var subfolderName = string.IsNullOrEmpty(subfolderTemplate) ? "" : candidate.Name;
            if (!string.IsNullOrEmpty(subfolderTemplate))
            {
                if (!subfolderPattern.IsMatch(subfolderName))
                    continue;
            }
            else if (!candidate.Exists)
            {
                continue;
            }

            var subfolderValues = subfolderPattern.Match(subfolderName).Groups
                .Cast<Group>()
                .Where(g => g.Name is "position" or "time" or "channel" or "z")
                .ToDictionary(g => g.Name, g => g.Value);

            foreach (var file in candidate.GetFiles().Where(f => IsSupportedSeriesFile(f.Name)).OrderBy(f => f.Name))
            {
                var filenameMatch = filenamePattern.Match(file.Name);
                if (!filenameMatch.Success)
                    continue;

                var filenameValues = filenameMatch.Groups
                    .Cast<Group>()
                    .Where(g => g.Name is "position" or "time" or "channel" or "z")
                    .ToDictionary(g => g.Name, g => g.Value);

                var values = new Dictionary<string, string>(subfolderValues);
                foreach (var pair in filenameValues)
                    values[pair.Key] = pair.Value;

                var relativePath = string.IsNullOrEmpty(subfolderName)
                    ? file.Name
                    : $"{subfolderName}/{file.Name}";

                matches.Add(new SeriesMatch
                {
                    Path = file.FullName,
                    RelativePath = relativePath,
                    Position = values.GetValueOrDefault("position", "0"),
                    Time = values.GetValueOrDefault("time", "0"),
                    Channel = values.GetValueOrDefault("channel", "0"),
                    Z = values.GetValueOrDefault("z", "0"),
                });
            }
        }

        return matches;
    }

    private static Regex CompileTemplate(string template, bool allowEmpty = false)
    {
        if (string.IsNullOrEmpty(template))
        {
            if (allowEmpty)
                return new Regex("^$", RegexOptions.IgnoreCase);
            throw new SidecarException("Filename template cannot be empty.");
        }

        var parts = new List<string> { "^" };
        var index = 0;
        var matches = Regex.Matches(template, @"\{([a-zA-Z_][a-zA-Z0-9_]*)\}");
        foreach (Match match in matches)
        {
            parts.Add(Regex.Escape(template[index..match.Index]));
            parts.Add(match.Groups[1].Value switch
            {
                "t" or "time" => "(?<time>.+?)",
                "p" or "position" => "(?<position>.+?)",
                "c" or "channel" => "(?<channel>.+?)",
                "z" => "(?<z>.+?)",
                _ => throw new SidecarException($"Unsupported placeholder {{{match.Groups[1].Value}}}"),
            });
            index = match.Index + match.Length;
        }

        parts.Add(Regex.Escape(template[index..]));
        parts.Add("$");
        return new Regex(string.Concat(parts), RegexOptions.IgnoreCase);
    }

    private static bool IsSupportedSeriesFile(string name)
    {
        if (name.EndsWith("_seg.cellpose", StringComparison.OrdinalIgnoreCase))
            return false;
        return SupportedExtensions.Contains(Path.GetExtension(name).ToLowerInvariant());
    }

    private static IComparable SortKey(string value) =>
        int.TryParse(value, out var number) ? number : value;

    private sealed class SeriesMatch
    {
        public string Path { get; init; } = "";
        public string RelativePath { get; init; } = "";
        public string Position { get; init; } = "0";
        public string Time { get; init; } = "0";
        public string Channel { get; init; } = "0";
        public string Z { get; init; } = "0";
    }
}
