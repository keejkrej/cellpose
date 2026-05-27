using CellposeGUI.Models;

namespace CellposeGUI.Services;

public interface IMlInferenceEngine
{
    Task<SidecarHealth> HealthAsync(CancellationToken cancellationToken = default);
    Task<SidecarModels> ListModelsAsync(CancellationToken cancellationToken = default);
    Task<InferResult> InferAsync(
        string imagePath,
        string? modelName,
        bool customModel,
        SegmentationParameters parameters,
        CancellationToken cancellationToken = default);
    Task<RecomputeResult> RecomputeAsync(
        IReadOnlyList<ArrayPayload> flows,
        SegmentationParameters parameters,
        CancellationToken cancellationToken = default);
    Task<TrainResult> TrainAsync(TrainingParameters parameters, CancellationToken cancellationToken = default);
    Task<string> AddModelAsync(string path, CancellationToken cancellationToken = default);
    Task RemoveModelAsync(string name, CancellationToken cancellationToken = default);
}

public sealed class InferResult
{
    public MaskData? Masks { get; init; }
    public List<ArrayPayload> Flows { get; init; } = [];
    public int Ncells { get; init; }
    public bool RecomputeMasks { get; init; }
}

public sealed class RecomputeResult
{
    public MaskData? Masks { get; init; }
    public int Ncells { get; init; }
}
