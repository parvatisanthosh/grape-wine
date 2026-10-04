# TinyLlama on OpenVINO GenAI — Findings Summary

## Setup & Build Log

- **Model:** TinyLlama-1.1B-Chat-v1.0, converted to OpenVINO IR via `optimum-cli`, tested in FP16, INT8, and INT4
- First inference succeeded via the Python GenAI API on CPU. Output was fluent but factually incorrect on the first try — a expected limitation at 1B-parameter scale
- The C++ build (`chat_sample.exe`) compiled successfully but exited silently at runtime (`STATUS_DLL_NOT_FOUND`, exit code `-1073741515`)
  - **Root cause:** pip-installed OpenVINO packages don't co-locate their runtime DLLs the way the full installer archive does
  - **Fix:** added `openvino\libs`, `openvino_genai`, and `openvino_tokenizers\lib` to `PATH`
  - One DLL (`openvino_tokenizers.dll`) was hardcoded to load from `openvino_genai`'s own folder rather than resolved via `PATH` search — required manually copying the file there

---

## Performance Benchmarks (FP16 vs INT8 vs INT4)

| Metric | FP16 | INT8 | INT4 |
|---|---|---|---|
| Load time | 3752 ms | 2664 ms | 3309 ms |
| Output tokens generated | 20 | 20 | 15 |
| TTFT (first token) | 647.20 ± 362.83 ms | 129.04 ± 125.11 ms | 115.34 ± 20.04 ms |
| TPOT (per token) | 260.82 ± 249.62 ms | 72.79 ± 27.61 ms | 50.10 ± 10.47 ms |
| Throughput | 3.83 ± 3.67 tok/s | 13.74 ± 5.21 tok/s | 19.96 ± 4.17 tok/s |

**Observations:**

1. **Throughput scales roughly as expected, but not linearly.** FP16 → INT8 gave a ~3.6x speedup (halving precision). INT8 → INT4 only gave ~1.45x — a much smaller jump. This tracks with INT4 not being purely 4-bit: 12% of layers stayed at INT8, so the full theoretical 2x from another halving was never fully realized.

2. **Variance shrinks as precision drops.** FP16 had huge relative variance (±362 ms on a 647 ms mean — over 50%). INT8 tightened substantially, and INT4 was the most consistent of all (±20 ms on 115 ms — about 17%). Likely explanation: less memory movement per token means less exposure to system-level scheduling noise.

3. **Load time didn't follow a clean pattern.** INT4 loaded slower than INT8 (3309 ms vs 2664 ms) despite having a smaller file on disk — possibly extra setup overhead from the mixed-precision structure, possibly one-off measurement noise. Worth flagging as an open question rather than glossing over.

---

## Model Size on Disk

| Precision | Model size | Reduction vs FP16 | Ratio |
|---|---|---|---|
| FP16 | 2098.2 MB | — | 1.0x |
| INT8 | 1050.7 MB | 49.9% smaller | 2.0x |
| INT4 | 623.6 MB | 70.3% smaller | 3.4x |

INT8 hit almost exactly the theoretical 2x reduction. INT4 fell short of the theoretical 4x (only 3.4x), consistent with the mixed-precision layer distribution noted above.

---

## KV Cache: Memory Growth (FP16)

| Stage | Memory | Δ from previous |
|---|---|---|
| Before load | 54.1 MB | — |
| After model load | 125.5 MB | +71.4 MB |
| After 50 tokens | 4150.9 MB | +4025.4 MB |
| After 200 tokens | 4158.5 MB | +7.6 MB |
| After 500 tokens | 4167.3 MB | +8.8 MB |
| After 1000 tokens | 4180.3 MB | +13.0 MB |

**Key findings:**
- **Lazy loading:** weights appear to load into active memory on the *first inference call*, not at pipeline construction — explains the massive jump between "after model load" and "after 50 tokens"
- **Empirical KV cache growth rate:** ~0.027 MB/token (measured from the 200→1000 token range)
- **Disproportionate slowdown:** memory grows roughly linearly with tokens, but generation *time* grows worse than linear (50→1000 tokens = 20x more tokens, but ~46x more time) — later tokens cost more to generate as the attention context grows

---

## KV Cache: Memory Growth (INT4)

| Stage | Memory | Time |
|---|---|---|
| Before load | 53.8 MB | — |
| After model load | 272.9 MB | — |
| After 50 tokens | 1337.3 MB | 8.1 s |
| After 200 tokens | 1342.6 MB | 10.0 s |
| After 500 tokens | 1348.9 MB | 26.3 s |
| After 1000 tokens | 1360.7 MB | 55.4 s |

## FP16 vs INT4 Memory Comparison

| Stage | FP16 | INT4 | Difference |
|---|---|---|---|
| Before load | 54.1 MB | 53.8 MB | ~same |
| After model load | 125.5 MB | 272.9 MB | INT4 higher (see note) |
| After 50 tokens | 4150.9 MB | 1337.3 MB | ~68% less |
| After 200 tokens | 4158.5 MB | 1342.6 MB | ~68% less |
| After 500 tokens | 4167.3 MB | 1348.9 MB | ~68% less |
| After 1000 tokens | 4180.3 MB | 1360.7 MB | ~67% less |

**Key finding:** total memory footprint during generation is consistently ~2/3 lower for INT4 vs FP16, and this holds steady across all token counts — a proportional reduction, not a one-time discount.

**Growth rate comparison** (isolating KV cache growth from the initial load jump, 200→1000 tokens):
- FP16: +21.8 MB over 800 tokens ≈ 0.027 MB/token
- INT4: +18.1 MB over 800 tokens ≈ 0.023 MB/token

The per-token growth rates are close (0.027 vs 0.023 MB/token) — meaning the ~3x total memory gap comes almost entirely from **smaller model weights in memory**, not from the KV cache itself growing at a fundamentally different rate. This is consistent with the KV cache typically being stored in a fixed precision (often FP16) internally regardless of weight quantization, unless explicitly configured otherwise. The savings from quantization are baked in once, upfront, rather than compounding per-token.

**Anomaly:** INT4's "after model load" memory (272.9 MB) is higher than FP16's (125.5 MB) — the opposite of what you'd expect before generation even starts. Possible explanation: quantized models may require extra one-time setup work (e.g., building dequantization lookup structures) before the real memory savings show up during actual inference.