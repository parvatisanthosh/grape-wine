# Draft issue for openvinotoolkit/openvino.genai (not filed)

**Title:** Enabling cache eviction together with prefix caching crashes on the next request: "blocks across layers must be freed simultaneously"

## Environment

- OpenVINO 2026.3.0 (pip), OpenVINO GenAI 2026.3.0.0 built from source at `bd8d6542`
- Windows 11, Intel Core i5-1335U, Intel Iris Xe Graphics
- Models: TinyLlama-1.1B INT4 IR on CPU; Qwen2.5-Coder-3B INT4 IR on GPU

## Steps to reproduce (C++ API)

Construct `ov::genai::LLMPipeline(models_path, device, ov::genai::scheduler_config(config))` with:

- `config.enable_prefix_caching = true`
- `config.use_cache_eviction = true`
- `config.cache_eviction_config = CacheEvictionConfig(32, 64, 256, AggregationMode::NORM_SUM)`

Then run a multi-turn chat through `ov::genai::ChatHistory`, so that each prompt includes
the earlier turns. It was found with `cpp/orbit_chat.cpp`, replaying
`conversations/long_recall.txt` with `--max-new-tokens 80 --evict 32:64:256`. The current
`orbit_chat` turns prefix caching off when eviction is requested, so reproducing it needs
that guard removed, or a minimal program. Failed rows are in
`results/chat/eviction_with_prefix_caching_CRASHED.jsonl`.

## Actual result

Generation fails once the conversation passes the eviction budget: after 3 turns with
`32:64:256`, and after 5 turns with `32:128:512` (TinyLlama, CPU).

```
Check 'is_all_free' failed at src/cpp/src/continuous_batching/cache/block_manager.hpp:371:
blocks across layers must be freed simultaneously
```

## Expected result

Either the two features work together, or pipeline construction rejects the combination
with a clear message.

## Notes

- With `enable_prefix_caching = false`, the same conversation completes all 12 turns.
- Eviction appears to remove blocks per layer (`need_per_layer_kv_cache_control` in
  `pipeline_impl.cpp`), while `BlockAllocator::free` asserts that a block group is freed
  across all layers at once. That would leave layers with different remaining blocks,
  which the prefix-cache path does not expect. This is an inference from reading the
  source, not a confirmed root cause.

## Before filing

- Check for an existing issue: search "freed simultaneously" and "cache eviction prefix".
- Reproduce with a minimal standalone program, and attach it.
- Try the latest nightly to see if it is already fixed.
