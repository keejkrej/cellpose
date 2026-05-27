using CellposeGUI.Models;

namespace CellposeGUI.Services;

public interface ISegmentationEngine
{
    Task<SidecarHealth> HealthAsync(CancellationToken cancellationToken = default);
    Task<SidecarModels> ListModelsAsync(CancellationToken cancellationToken = default);
    Task<SegmentationResult> LoadImageAsync(string path, bool load3D, CancellationToken cancellationToken = default);
    Task<SegmentationResult> LoadSegAsync(string path, bool load3D, CancellationToken cancellationToken = default);
    Task<string> SaveSegAsync(string sessionId, string? path, CancellationToken cancellationToken = default);
    Task<string> ExportMasksAsync(string sessionId, string path, string format, CancellationToken cancellationToken = default);
    Task<SegmentationResult> SegmentAsync(
        string? sessionId,
        ArrayPayload? imagePayload,
        string? filename,
        string? modelName,
        bool customModel,
        SegmentationParameters parameters,
        PreprocessingParameters preprocess,
        CancellationToken cancellationToken = default);
    Task<SegmentationResult> RecomputeMasksAsync(string sessionId, SegmentationParameters parameters, CancellationToken cancellationToken = default);
    Task<SegmentationResult> PreprocessAsync(string sessionId, PreprocessingParameters parameters, CancellationToken cancellationToken = default);
    Task<SegmentationResult> RemoveCellsAsync(string sessionId, IReadOnlyList<int> indices, CancellationToken cancellationToken = default);
    Task<SegmentationResult> AddMaskAsync(string sessionId, double[][][] strokes, int classId, CancellationToken cancellationToken = default);
    Task<SegmentationResult> MergeCellsAsync(string sessionId, int source, int target, CancellationToken cancellationToken = default);
    Task<SeriesDiscovery> DiscoverSeriesAsync(string folder, string subfolderTemplate, string filenameTemplate, CancellationToken cancellationToken = default);
    Task<SeriesTemplateSuggestion> SuggestSeriesTemplatesAsync(string folder, CancellationToken cancellationToken = default);
    Task<string> ExportOutlinesAsync(string sessionId, string path, CancellationToken cancellationToken = default);
    Task<string> ExportFlowsAsync(string sessionId, string path, CancellationToken cancellationToken = default);
    Task<string> ExportROIsAsync(string sessionId, string path, CancellationToken cancellationToken = default);
    Task<SegmentationResult> NavigateSeriesAsync(string sessionId, int recordIndex, CancellationToken cancellationToken = default);
    Task<TrainResult> TrainAsync(TrainingParameters parameters, CancellationToken cancellationToken = default);
    Task<string> AddModelAsync(string path, CancellationToken cancellationToken = default);
    Task RemoveModelAsync(string name, CancellationToken cancellationToken = default);
}

public sealed class SidecarSegmentationEngine(SidecarClient client) : ISegmentationEngine
{
    public Task<SidecarHealth> HealthAsync(CancellationToken cancellationToken = default) =>
        client.GetAsync<SidecarHealth>("/health", cancellationToken);

    public Task<SidecarModels> ListModelsAsync(CancellationToken cancellationToken = default) =>
        client.GetAsync<SidecarModels>("/models", cancellationToken);

    public async Task<SegmentationResult> LoadImageAsync(string path, bool load3D, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/io/load-image",
            new { path, load_3D = load3D },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public async Task<SegmentationResult> LoadSegAsync(string path, bool load3D, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/io/load-seg",
            new { path, load_3D = load3D },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public async Task<string> SaveSegAsync(string sessionId, string? path, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SavePathResponse>(
            "/io/save-seg",
            new { session_id = sessionId, path },
            cancellationToken);
        return response.Path;
    }

    public async Task<string> ExportMasksAsync(string sessionId, string path, string format, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SavePathResponse>(
            "/export/masks",
            new { session_id = sessionId, path, format },
            cancellationToken);
        return response.Path;
    }

    public async Task<SegmentationResult> SegmentAsync(
        string? sessionId,
        ArrayPayload? imagePayload,
        string? filename,
        string? modelName,
        bool customModel,
        SegmentationParameters parameters,
        PreprocessingParameters preprocess,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/segment",
            new
            {
                session_id = sessionId,
                image = imagePayload,
                filename,
                model_name = modelName,
                custom_model = customModel,
                @params = parameters,
                preprocess,
            },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public async Task<SegmentationResult> RecomputeMasksAsync(
        string sessionId,
        SegmentationParameters parameters,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/recompute-masks",
            new { session_id = sessionId, @params = parameters },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public async Task<SegmentationResult> PreprocessAsync(
        string sessionId,
        PreprocessingParameters parameters,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            $"/preprocess/{sessionId}",
            parameters,
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public async Task<SegmentationResult> RemoveCellsAsync(
        string sessionId,
        IReadOnlyList<int> indices,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/masks/remove",
            new { session_id = sessionId, cell_indices = indices },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public async Task<SegmentationResult> AddMaskAsync(
        string sessionId,
        double[][][] strokes,
        int classId,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/masks/add",
            new { session_id = sessionId, strokes, class_id = classId },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public async Task<SegmentationResult> MergeCellsAsync(
        string sessionId,
        int source,
        int target,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/masks/merge",
            new { session_id = sessionId, source_index = source, target_index = target },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public Task<SeriesDiscovery> DiscoverSeriesAsync(
        string folder,
        string subfolderTemplate,
        string filenameTemplate,
        CancellationToken cancellationToken = default) =>
        client.PostAsync<SeriesDiscovery>(
            "/series/discover",
            new
            {
                folder,
                subfolder_template = subfolderTemplate,
                filename_template = filenameTemplate,
            },
            cancellationToken);

    public Task<SeriesTemplateSuggestion> SuggestSeriesTemplatesAsync(string folder, CancellationToken cancellationToken = default) =>
        client.PostAsync<SeriesTemplateSuggestion>("/series/suggest", new { folder }, cancellationToken);

    public async Task<string> ExportOutlinesAsync(string sessionId, string path, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SavePathResponse>(
            "/export/outlines",
            new { session_id = sessionId, path, format = "txt" },
            cancellationToken);
        return response.Path;
    }

    public async Task<string> ExportFlowsAsync(string sessionId, string path, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SavePathResponse>(
            "/export/flows",
            new { session_id = sessionId, path, format = "tif" },
            cancellationToken);
        return response.Path;
    }

    public async Task<string> ExportROIsAsync(string sessionId, string path, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SavePathResponse>(
            "/export/rois",
            new { session_id = sessionId, path, format = "zip" },
            cancellationToken);
        return response.Path;
    }

    public async Task<SegmentationResult> NavigateSeriesAsync(
        string sessionId,
        int recordIndex,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<SidecarSessionResponse>(
            "/series/navigate",
            new { session_id = sessionId, record_index = recordIndex },
            cancellationToken);
        return ArrayCodec.ToSegmentationResult(response);
    }

    public Task<TrainResult> TrainAsync(TrainingParameters parameters, CancellationToken cancellationToken = default) =>
        client.PostAsync<TrainResult>("/train", parameters, cancellationToken);

    public async Task<string> AddModelAsync(string path, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<AddModelResponse>("/models/add", new { path }, cancellationToken);
        return response.ModelName;
    }

    public Task RemoveModelAsync(string name, CancellationToken cancellationToken = default) =>
        client.PostAsync<RemoveModelResponse>("/models/remove", new { model_name = name }, cancellationToken);

    private sealed class SavePathResponse
    {
        public string Path { get; set; } = "";
    }

    private sealed class AddModelResponse
    {
        public string ModelName { get; set; } = "";
    }

    private sealed class RemoveModelResponse
    {
        public string Removed { get; set; } = "";
    }
}
