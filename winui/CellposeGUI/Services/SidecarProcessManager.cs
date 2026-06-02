using System.Diagnostics;
using System.Text;
using CellposeGUI.Models;

namespace CellposeGUI.Services;

public sealed class SidecarProcessManager
{
    private Process? _process;
    private readonly StringBuilder _stderr = new();

    public Uri BaseUri { get; }
    public string RepoRoot { get; }
    public bool IsReady { get; private set; }
    public string StatusMessage { get; private set; } = "Starting sidecar…";

    public event Action<string>? StatusChanged;

    public SidecarProcessManager(Uri? baseUri = null)
    {
        var envUrl = Environment.GetEnvironmentVariable("SIDECAR_URL");
        if (baseUri != null)
            BaseUri = baseUri;
        else if (!string.IsNullOrWhiteSpace(envUrl) && Uri.TryCreate(envUrl, UriKind.Absolute, out var parsed))
            BaseUri = parsed;
        else
            BaseUri = new Uri("http://127.0.0.1:8787");

        RepoRoot = ResolveRepoRoot();
    }

    public async Task StartIfNeededAsync(CancellationToken cancellationToken = default)
    {
        SetStatus("Checking sidecar…");
        if (await ProbeHealthAsync(cancellationToken))
        {
            IsReady = true;
            SetStatus("Sidecar ready");
            return;
        }

        if (_process != null)
            return;

        SetStatus("Launching Python sidecar…");
        var port = BaseUri.IsDefaultPort ? 8787 : BaseUri.Port;
        var uvPath = FindUvExecutable();

        _process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = uvPath,
                Arguments = $"run python -m cellpose.api --host 127.0.0.1 --port {port}",
                WorkingDirectory = RepoRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            },
            EnableRaisingEvents = true,
        };

        _process.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
                _stderr.AppendLine(e.Data);
        };

        try
        {
            if (!_process.Start())
            {
                SetStatus("Failed to launch sidecar process.");
                _process = null;
                return;
            }

            _process.BeginErrorReadLine();
        }
        catch (Exception ex)
        {
            SetStatus($"Failed to launch sidecar: {ex.Message}");
            _process = null;
            return;
        }

        // PyTorch + first import can take well over 30s on cold start.
        for (var attempt = 0; attempt < 180; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_process.HasExited && _process.ExitCode != 0)
            {
                var detail = _stderr.ToString().Trim();
                SetStatus(string.IsNullOrWhiteSpace(detail)
                    ? $"Sidecar exited with code {_process.ExitCode}."
                    : $"Sidecar failed: {detail}");
                _process = null;
                return;
            }

            if (await ProbeHealthAsync(cancellationToken))
            {
                IsReady = true;
                SetStatus("Sidecar ready");
                return;
            }

            SetStatus($"Waiting for sidecar ({attempt + 1}/180)…");
            await Task.Delay(1000, cancellationToken);
        }

        var stderrTail = _stderr.ToString().Trim();
        SetStatus(string.IsNullOrWhiteSpace(stderrTail)
            ? "Sidecar failed to start. Run `uv run python -m cellpose.api` manually to see errors."
            : $"Sidecar failed to start: {stderrTail}");
    }

    public void Stop()
    {
        if (_process == null)
            return;

        try
        {
            if (!_process.HasExited)
                _process.Kill(entireProcessTree: true);
        }
        catch
        {
            // Best effort shutdown.
        }
        finally
        {
            _process.Dispose();
            _process = null;
            IsReady = false;
        }
    }

    private void SetStatus(string message)
    {
        StatusMessage = message;
        StatusChanged?.Invoke(message);
    }

    private async Task<bool> ProbeHealthAsync(CancellationToken cancellationToken)
    {
        try
        {
            var client = new SidecarClient(BaseUri);
            var health = await client.GetAsync<SidecarHealth>("/health", cancellationToken);
            return health.Status == "ok";
        }
        catch
        {
            return false;
        }
    }

    private static string ResolveRepoRoot()
    {
        var envRoot = Environment.GetEnvironmentVariable("CELLPOSE_ROOT");
        if (!string.IsNullOrWhiteSpace(envRoot) && Directory.Exists(envRoot))
            return Path.GetFullPath(envRoot);

        foreach (var start in new[]
                 {
                     Directory.GetCurrentDirectory(),
                     AppContext.BaseDirectory,
                 })
        {
            var found = FindRepoRoot(start);
            if (found != null)
                return found;
        }

        return Directory.GetCurrentDirectory();
    }

    private static string? FindRepoRoot(string startPath)
    {
        var current = new DirectoryInfo(Path.GetFullPath(startPath));
        while (current != null)
        {
            if (Directory.Exists(Path.Combine(current.FullName, "cellpose")) &&
                File.Exists(Path.Combine(current.FullName, "pyproject.toml")))
                return current.FullName;

            current = current.Parent;
        }

        return null;
    }

    private static string FindUvExecutable()
    {
        var pathEntries = (Environment.GetEnvironmentVariable("PATH") ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries);

        foreach (var entry in pathEntries)
        {
            var candidate = Path.Combine(entry, OperatingSystem.IsWindows() ? "uv.exe" : "uv");
            if (File.Exists(candidate))
                return candidate;
        }

        return "uv";
    }
}

