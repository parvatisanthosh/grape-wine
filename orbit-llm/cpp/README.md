# orbit_run — C++ executor

`orbit_run` runs one controlled OpenVINO GenAI benchmark configuration through the
GenAI C++ API and writes one JSON line per measured run. The Python layer
(`orbit_exec.py`) chooses configurations, calls it, and collects the rows.

## Build

Uses OpenVINO from the pip package and GenAI from the local source build in
`../../openvino.genai/build` (both 2026.3).

```powershell
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 `
      -DOpenVINO_DIR="C:/Users/parva/venv/Lib/site-packages/openvino/cmake"
cmake --build cpp/build --config Release
```

The build copies `openvino_genai.dll` and `openvino_tokenizers.dll` next to the
executable. OpenVINO core DLLs and device plugins are found through `PATH`.

## Run

```powershell
$env:Path = "C:\Users\parva\venv\Lib\site-packages\openvino\libs;$env:Path"
.\cpp\build\Release\orbit_run.exe --model C:\Users\parva\tinyllama-int4 --label INT4 `
    --prompt-tokens 512 --output-tokens 32 --runs 3 --output results\example.jsonl
```

Run `orbit_run.exe --help` for all options.

| Option | Values | Controls |
|---|---|---|
| `--device` | `CPU`, `GPU` | Inference device |
| `--backend` | `pa`, `sdpa` | Paged attention (continuous batching) or SDPA |
| `--prefix-caching` | `on`, `off` | Reuse KV cache for repeated prompt prefixes (PA) |
| `--max-batched-tokens` | `0` = unlimited | Prefill chunk size (PA) |
| `--cache-size-gb` | `0` = dynamic | Pre-allocated KV-cache size (PA) |
| `--kv-precision` | `default`, `f32`, `f16`, `bf16`, `u8`, `u4` | KV-cache element type |
| `--threads` | `0` = default | CPU inference threads |
| `--cores` | `any`, `pcore`, `ecore` | CPU core type on hybrid CPUs |
| `--prompt-mode` | `cold`, `repeat` | Unique prompt per run, or the same prompt every run |

With the PA backend, `orbit_run` starts from `LLMPipeline`'s own latency-oriented
scheduler defaults (unlimited batched tokens, prefix caching on). A plain
`SchedulerConfig{}` would cap prefill at 256 tokens per step and disable prefix
caching, silently changing what is being measured.

## Output

Each row records the configuration, exact token counts, TTFT, TPOT, throughput,
sampled peak working set, and machine state just before the run (available RAM,
system CPU busy %). Load and warm-up failures (e.g. out of memory) are written as
rows with `status` `load_failed` / `warmup_failed`; `orbit_exec.py` adds a
`crashed` row if the process dies without writing anything.
