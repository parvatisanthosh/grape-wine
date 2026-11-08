# C++ programs (OpenVINO GenAI C++ API)

| Program | Purpose |
|---|---|
| `orbit_chat` | Local multi-turn chat application (the deployment) |
| `orbit_run` | One measured configuration per process; used by all experiments |
| `orbit_worker` | Long-lived process that keeps models loaded and serves requests over stdin/stdout |

All three are built by the same CMake project (see Build below).

## orbit_chat — local chat application

```powershell
$env:Path = "C:\Users\parva\venv\Lib\site-packages\openvino\libs;$env:Path"
.\cpp\build\Release\orbit_chat.exe --model C:\Users\parva\tinyllama-int4
```

- **Chat:** multi-turn with the model's chat template (`ov::genai::ChatHistory`) and
  streamed output.
- **Per-turn metrics:** prompt tokens (whole conversation), answer tokens, TTFT, tok/s,
  working set and committed memory.
- **Commands:** `/reset`, `/stats`, `/help`, `/exit`.

| Option | Values | Controls |
|---|---|---|
| `--device` | `CPU`, `GPU` | Inference device |
| `--cores` | `any`, `pcore`, `ecore` | CPU core type (CPU only) |
| `--cache-dir` | path | Compiled-model cache; use it for the GPU |
| `--prefix-caching` | `on` (default), `off` | Reuse the KV cache of earlier turns |
| `--kv-precision` | `default`, `f32`, `f16`, `u8` | KV-cache element type (CPU only; explicit precision crashes the GPU path on GenAI 2026.3) |
| `--evict` | `START:RECENT:MAX` | KV-cache eviction budget in tokens (multiples of the block size: 32 on CPU, 16 on GPU) |
| `--max-new-tokens`, `--temperature`, `--repetition-penalty`, `--system` | | Generation settings |
| `--script FILE`, `--json-out FILE` | | Replay a conversation non-interactively and save per-turn metrics |

It warns at start-up when the laptop is on battery: inference runs ~10× slower there
(FINDINGS Finding 20).

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
| `--cache-dir` | path, empty = off | Compiled-model cache; cuts GPU load time on later loads |
| `--prompt-mode` | `cold`, `repeat` | Unique prompt per run, or the same prompt every run |

With the PA backend, `orbit_run` starts from `LLMPipeline`'s own latency-oriented
scheduler defaults (unlimited batched tokens, prefix caching on). A plain
`SchedulerConfig{}` would cap prefill at 256 tokens per step and disable prefix
caching, silently changing what is being measured.

## Output

Each row records the configuration, exact token counts, TTFT, TPOT, throughput,
sampled peak working set, committed private memory (after load, warm-up and
generation, plus lifetime peak), and machine state just before the run (available
RAM, system CPU busy %), plus CPU behaviour *during* the run: effective CPU
frequency (`% Processor Performance` x base MHz), this process's CPU share, and
background CPU share. Use committed memory, not working set, for OOM analysis:
a pre-allocated KV cache is committed but never shows up in the working set
(FINDINGS.md, Finding 9). Load and warm-up failures (e.g. out of memory) are written as
rows with `status` `load_failed` / `warmup_failed`; `orbit_exec.py` adds a
`crashed` row if the process dies without writing anything.
