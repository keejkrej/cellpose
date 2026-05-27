using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class SidecarMlEngine(SidecarClient client) : IMlInferenceEngine
{
    public Task<SidecarHealth> HealthAsync(CancellationToken cancellationToken = default) =>
        client.GetAsync<SidecarHealth>("/health", cancellationToken);

    public Task<SidecarModels> ListModelsAsync(CancellationToken cancellationToken = default) =>
        client.GetAsync<SidecarModels>("/models", cancellationToken);

    public async Task<InferResult> InferAsync(
        string imagePath,
        string? modelName,
        bool customModel,
        SegmentationParameters parameters,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<InferResponsePayload>(
            "/infer",
            new
            {
                path = imagePath,
                load_3D = false,
                model_name = modelName,
                custom_model = customModel,
                @params = parameters,
            },
            cancellationToken);
        return ArrayCodec.ToInferResult(response);
    }

    public async Task<RecomputeResult> RecomputeAsync(
        IReadOnlyList<ArrayPayload> flows,
        SegmentationParameters parameters,
        CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<RecomputeResponsePayload>(
            "/recompute",
            new
            {
                flows,
                @params = parameters,
            },
            cancellationToken);
        return ArrayCodec.ToRecomputeResult(response);
    }

    public Task<TrainResult> TrainAsync(TrainingParameters parameters, CancellationToken cancellationToken = default) =>
        client.PostAsync<TrainResult>("/train", parameters, cancellationToken);

    public async Task<string> AddModelAsync(string path, CancellationToken cancellationToken = default)
    {
        var response = await client.PostAsync<AddModelResponse>(
            "/models/add",
            new { path },
            cancellationToken);
        return response.ModelName;
    }

    public Task RemoveModelAsync(string name, CancellationToken cancellationToken = default) =>
        client.PostAsync<RemoveModelResponse>("/models/remove", new { model_name = name }, cancellationToken);

    private sealed class AddModelResponse
    {
        [JsonPropertyName("model_name")]
        public string ModelName { get; set; } = "";
    }

    private sealed class RemoveModelResponse
    {
        public string Removed { get; set; } = "";
    }
}

public sealed class SidecarClient
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
    };

    private readonly HttpClient _http;
    private readonly Uri _baseUri;

    public SidecarClient(Uri baseUri)
    {
        _baseUri = baseUri;
        _http = new HttpClient
        {
            Timeout = TimeSpan.FromHours(1),
        };
    }

    public async Task<T> GetAsync<T>(string path, CancellationToken cancellationToken = default)
    {
        var response = await _http.GetAsync(new Uri(_baseUri, path.TrimStart('/')), cancellationToken);
        return await ReadResponseAsync<T>(response, cancellationToken);
    }

    public async Task<T> PostAsync<T>(string path, object body, CancellationToken cancellationToken = default)
    {
        var json = JsonSerializer.Serialize(body, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await _http.PostAsync(new Uri(_baseUri, path.TrimStart('/')), content, cancellationToken);
        return await ReadResponseAsync<T>(response, cancellationToken);
    }

    private static async Task<T> ReadResponseAsync<T>(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        var data = await response.Content.ReadAsStringAsync(cancellationToken);
        if (response.IsSuccessStatusCode)
            return JsonSerializer.Deserialize<T>(data, JsonOptions)
                   ?? throw new SidecarException("Invalid response from sidecar");

        var detail = JsonSerializer.Deserialize<SidecarErrorResponse>(data, JsonOptions)?.Detail;
        throw new SidecarException(detail ?? data);
    }
}
