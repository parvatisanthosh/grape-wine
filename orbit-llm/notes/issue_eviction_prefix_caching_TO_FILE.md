# Ready to file at https://github.com/openvinotoolkit/openvino.genai/issues/new

Copy the title and everything under "Body" into the new issue. Attach
`cpp/repro/repro_eviction_prefix.py` (or paste it, it is short). The C++ version is
`cpp/repro/repro_eviction_prefix.cpp` if a maintainer asks for one.

---

## Title

Cache eviction with prefix caching fails in multi-turn chat: "blocks across layers must be freed simultaneously", then leaks KV blocks

## Body

### Summary

With `SchedulerConfig.use_cache_eviction = True` and `enable_prefix_caching = True`,
a multi-turn `ChatHistory` conversation fails as soon as the prompt grows past the
eviction budget. GenAI then reports leaked KV blocks. With prefix caching off, the same
conversation completes.

### Environment

- OpenVINO GenAI **2026.4.1.0** (`2026.4.1.0-3408-ab01d1771e4`, pip) — fails
- OpenVINO GenAI **2026.3.0.0** (built from source at `bd8d6542`) — fails the same way
- Windows 11, Intel Core i5-1335U, CPU device
- Model: TinyLlama-1.1B-Chat-v1.0, INT4 weights, exported with optimum-intel. On GenAI
  2026.3 it also failed with `OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov` on the GPU
  device (Intel Iris Xe); see the table below.

### Steps to reproduce

```python
import sys
import openvino_genai as og

model, device, prefix = sys.argv[1], sys.argv[2], sys.argv[3] == "on"

scheduler = og.SchedulerConfig()
scheduler.enable_prefix_caching = prefix
scheduler.use_cache_eviction = True
scheduler.cache_eviction_config = og.CacheEvictionConfig(32, 64, 256, og.AggregationMode.NORM_SUM)

pipeline = og.LLMPipeline(model, device, scheduler_config=scheduler)
config = og.GenerationConfig()
config.max_new_tokens = 80

history = og.ChatHistory()
message = ("Tell me one more fact about the history of computing, in two sentences. "
           "Remember that my project is called ORBIT and my lucky number is 47.")
for turn in range(1, 9):
    history.append({"role": "user", "content": message})
    result = pipeline.generate(history, config)
    history.append({"role": "assistant", "content": result.texts[0]})
    print(f"turn {turn} OK, prompt tokens {result.perf_metrics.get_num_input_tokens()}")
```

Run it as `python repro.py <model_dir> CPU on`.

### Actual result (prefix caching on)

```
turn 1 OK, prompt tokens 49
turn 2 OK, prompt tokens 180
turn 3 OK, prompt tokens 311
RuntimeError: Check 'is_all_free' failed at src/cpp/src/continuous_batching/cache/block_manager.hpp:371:
blocks across layers must be freed simultaneously
[ERROR] BlockManager leaked sequence block tables: 1, first leaked sequence id: 3
[ERROR] BlockAllocator leaked blocks. Expected num free blocks: 17, actual: 9
```

The failure comes at turn 4, the first prompt (442 tokens) longer than the eviction
budget (`max_cache_size` = 256).

A longer 12-turn conversation, with 80-token answers, gave this:

| Model, device | `CacheEvictionConfig(32, 64, 256)` | `CacheEvictionConfig(32, 128, 512)` | No eviction |
|---|---|---|---|
| TinyLlama INT4, CPU | fails after 3 turns | fails after 5 turns | all 12 turns |
| Qwen2.5-Coder-3B INT4, GPU | fails after 4 turns | fails after 6 turns | all 12 turns |

### With prefix caching off

All 8 turns complete, with prompts of up to 782 tokens.

### Expected result

Either the two features work together, or the pipeline rejects the combination when it
is constructed, with a clear message. The documentation does not mention that they are
incompatible.

### Notes

- Eviction appears to make block tables per decoder layer (`need_per_layer_kv_cache_control`
  in `pipeline_impl.cpp`), while freeing a prefix-cached block asserts that the block is
  free in all layers at once (`OverwritableBlocksHashStore::add`). This is an inference from
  reading the source, not a confirmed root cause. Draft PR #4680 also mentions that
  eviction switches the block table to per-decoder-layer indices.
- No existing issue was found for this error.
