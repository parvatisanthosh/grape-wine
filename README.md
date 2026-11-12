# Local LLM on OpenVINO — with ORBIT-LLM

A 1–3B parameter language model running entirely on a laptop through the **OpenVINO GenAI
C++ API**. Each part was measured: inference, quantization, the KV cache, hardware use and
OpenVINO's KV-cache management. This is semester project **SWFW_5S_13 — Local LLM on
OpenVINO** (Zoho).

Everything lives in **[`orbit-llm/`](orbit-llm/)**:

| Start here | What it is |
|---|---|
| [`orbit-llm/REPORT.md`](orbit-llm/REPORT.md) | The report, organized by the task's four learning outcomes |
| [`orbit-llm/cpp/orbit_chat.cpp`](orbit-llm/cpp/orbit_chat.cpp) | **The deliverable:** a local multi-turn chat app on the C++ API, with streaming, CPU or iGPU, KV-cache options and per-turn metrics |
| [`orbit-llm/FINDINGS.md`](orbit-llm/FINDINGS.md) | Lab log: every experiment with its method, raw-data path and conclusion |
| [`orbit-llm/README.md`](orbit-llm/README.md) | Component map, build and demo commands |

## Results

Every speed and memory number comes from one laptop: Intel Core i5-1335U (2 P-cores + 8
E-cores), Iris Xe iGPU, 16 GB RAM, OpenVINO 2026.3. Models: TinyLlama-1.1B-Chat (FP16,
INT8, INT4) and Qwen2.5-Coder-3B-Instruct (INT4). The HumanEval quality scores were run
on a Colab CPU, so that all three 3B precisions ran on the same machine.

- **Quantization, speed (TinyLlama, CPU):** INT8 decodes 1.8–2.0× faster than FP16 and
  INT4 2.2–3.1× faster, using 48% and 67% less memory.
- **Quantization, quality (Qwen2.5-Coder-3B, HumanEval, 164 problems):**
  - FP16 and INT8 each pass 137; INT4 passes 135.
  - There is no detectable difference (McNemar p = 0.79), and INT4 weights are 71%
    smaller.
  - A 95% interval still allows INT4 to be up to 5.7 points worse.
  - TinyLlama-1.1B passes only 10–12 of 164 at every precision, so here model size
    limits quality far more than precision does.
- **Prefix caching:** a repeated 1,920-token prompt reaches its first token about 177×
  faster.
- **What the prefix cache keeps:**
  - OpenVINO's default KV pool keeps only the latest prompt.
  - A rule derived from the GenAI source predicts how many conversations a fixed pool
    keeps: K × ⌈tokens / block size⌉ blocks for K conversations.
  - Measured: 45 of 45 hits on the iGPU and 23 of 24 on CPU at exactly that size, and 0
    hits one block short.
- **KV-cache precision on the iGPU:**
  - Read from the compiled model, OpenVINO stores the KV cache in **4 bits by default**:
    11.5 KiB per token for the 3B model, against 36 at f16.
  - Measured GPU memory matches that layout to within 1.5 MiB, and speed does not change.
  - The 4-bit cache changes 40% of HumanEval answers (int8 only 5%). Retrieval at up to
    3.5k tokens stays perfect.
  - See [Finding 30](orbit-llm/FINDINGS.md).
- **Hardware:** the iGPU is 3–7× faster than the CPU to the first token and 1.4–2.6×
  faster at generating. On the hybrid CPU, two P-cores beat the 12-thread default at
  generating by 16–40%.
- **Two OpenVINO bugs, reproduced minimally:**
  - The GPU model cache silently ignores `KV_CACHE_PRECISION`: a later compile gets the
    first compile's KV layout. This reproduces on 2026.3 and 2026.4.1. Report text:
    [`orbit-llm/notes/issue_gpu_cache_kv_precision_TO_FILE.md`](orbit-llm/notes/issue_gpu_cache_kv_precision_TO_FILE.md).
  - Combining cache eviction with prefix caching crashes a multi-turn chat and leaks KV
    blocks.
  - A 60-line program reproduces it on GenAI 2026.3 and on the latest 2026.4.1.
  - Report text: [`orbit-llm/notes/issue_eviction_prefix_caching_TO_FILE.md`](orbit-llm/notes/issue_eviction_prefix_caching_TO_FILE.md).

## Extension: ORBIT-LLM

ORBIT is a controller that predicts how every configuration (model, precision, device,
cores) would perform for a request, picks the best one within the user's limits, and
corrects itself from measurements.
- **Prediction accuracy:** its predictor reaches within about 4 points of the
  run-to-run noise.
- **Against fixed settings:** the first evaluation, on synthetic requests, showed **no
  win** over one well-chosen fixed configuration.
- **On a real workload:** the second evaluation used 300 requests with real lengths, from
  the Azure 2023 LLM traces and OpenAssistant conversations, with its plan written down
  before any measurement.
  - ORBIT broke **84** service limits, against **96** for the best fixed configuration
    (exact McNemar p = 0.0005) and **152** for OpenVINO's default.
  - The gain comes from sending code to the 3B model, and a simple rule matches it
    (84). See [Finding 29](orbit-llm/FINDINGS.md) and
    [the plan](orbit-llm/notes/evaluation_v2_plan.md).

## Quick start

Prerequisites:
- Windows with Visual Studio 2022;
- the Python packages `openvino==2026.3.0` and `openvino-genai==2026.3.0.0` in a
  virtual environment;
- OpenVINO GenAI built from source in `openvino.genai/` (commit `bd8d6542`);
- a model in OpenVINO IR format, e.g. `OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov` from
  Hugging Face.

```powershell
git clone --recurse-submodules https://github.com/parvatisanthosh/grape-wine.git
cd grape-wine/orbit-llm
cmake -S cpp -B cpp/build -G "Visual Studio 17 2022" -A x64 `
      -DOpenVINO_DIR="<venv>/Lib/site-packages/openvino/cmake"
cmake --build cpp/build --config Release
$env:Path = "<venv>\Lib\site-packages\openvino\libs;$env:Path"
.\cpp\build\Release\orbit_chat.exe --model <model-dir> --device GPU
```

Build details and every option: [`orbit-llm/cpp/README.md`](orbit-llm/cpp/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `orbit-llm/` | The project: C++ programs, experiments, results, report |
| `openvino.genai/` | OpenVINO GenAI source at commit `bd8d6542`, the version this project is built against |
| `LOG.md`, `test*.py`, `runchat.ps1` | First exploration with the Python API, before the C++ work |
