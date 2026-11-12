"""The exact KV-cache layout OpenVINO chooses, per device and KV precision.

GenAI's continuous-batching pipeline (LLMPipeline with paged attention) does
three things before it allocates the KV cache:
  1. runs ov::pass::SDPAToPagedAttention on the model
     (pipeline_impl.cpp: prepare_model_for_paged_attention);
  2. compiles it on the device with the user's properties;
  3. reads every "key_cache.N" / "value_cache.N" input of the compiled model and
     sums shape x element size into the bytes of one KV block
     (kv_cache_manager.hpp).
This script does steps 1 and 2 with the same OpenVINO runtime the C++ worker
loads, and reports step 3 exactly: the element type, the block shape (including
any rows the plugin adds for quantization scales) and the bytes per token.

It measures layout, not speed or quality: nothing is generated.

Run from orbit-llm/ with the OpenVINO venv:
    C:/Users/parva/venv/Scripts/python.exe kv_layout.py
Output: results/kv_layout.json
"""

import argparse
import gc
import json
import time
from pathlib import Path

import openvino as ov
from openvino._offline_transformations import paged_attention_transformation

MODELS = {
    "TinyLlama-1.1B INT4": "C:/Users/parva/tinyllama-int4",
    "Qwen2.5-Coder-3B INT4": "C:/Users/parva/qwen2.5-coder-3b-int4",
}
PRECISIONS = ["default", "f32", "f16", "u8", "i8", "u4", "i4"]
BLOCK_TOKENS = {"CPU": 32, "GPU": 16}  # kv_cache_manager.hpp
OUT = Path("results/kv_layout.json")


def cache_inputs(compiled):
    """(layer kind, shape, element type) of every KV cache input."""
    found = []
    for port in compiled.inputs:
        for name in port.get_names():
            if name.startswith(("key_cache.", "value_cache.")):
                # Dimension 0 is the number of blocks, set at allocation; GenAI
                # sizes a block from dimensions 1-3, which are static.
                shape = port.get_partial_shape()
                dims = [shape[i].get_length() if shape[i].is_static else None for i in range(1, 4)]
                found.append((name, [None] + dims, port.get_element_type()))
                break
    return found


def layout(core, model_path, device, precision, cache_dir=None):
    model = core.read_model(Path(model_path) / "openvino_model.xml")
    # GenAI's call for a plain LLMPipeline (no eviction, no xattention):
    # SDPAToPagedAttention(false, false, true, false, false, false).
    paged_attention_transformation(model, False, False, True, False, False, False)
    properties = {} if precision == "default" else {"KV_CACHE_PRECISION": precision}
    # No model cache by default: on the GPU a cache hit returns the blob compiled
    # first, whatever KV precision is asked for (Finding 30).
    if cache_dir:
        properties["CACHE_DIR"] = cache_dir
    start = time.perf_counter()
    compiled = core.compile_model(model, device, properties)
    compile_s = time.perf_counter() - start

    inputs = cache_inputs(compiled)
    block_bytes = 0
    for _, shape, element in inputs:
        if None in shape[1:]:
            raise RuntimeError(f"cache block shape not static: {shape}")
        values = shape[1] * shape[2] * shape[3]
        # GenAI multiplies by element size in bytes (u4/i4 report 0.5).
        block_bytes += values * element.size if element.bitwidth >= 8 else values * element.bitwidth / 8
    keys = [entry for entry in inputs if entry[0].startswith("key_cache.")]
    values = [entry for entry in inputs if entry[0].startswith("value_cache.")]
    result = {
        "device": device,
        "requested": precision,
        "reported_kv_cache_precision": str(compiled.get_property("KV_CACHE_PRECISION")),
        "key_type": str(keys[0][2]),
        "value_type": str(values[0][2]),
        "key_block_shape": keys[0][1][1:],
        "value_block_shape": values[0][1][1:],
        "layers": len(keys),
        "block_tokens": BLOCK_TOKENS[device],
        "block_bytes": block_bytes,
        "kib_per_token": block_bytes / BLOCK_TOKENS[device] / 1024,
        "compile_s": round(compile_s, 1),
    }
    del compiled, model
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", nargs="+", default=["GPU", "CPU"])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--cache-dir", default=None,
                        help="Compile through this model cache (shows the cache problem).")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out

    core = ov.Core()
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["model"], r["device"], r["requested"]) for r in results}
    for model_name in args.models:
        for device in args.devices:
            for precision in PRECISIONS:
                if (model_name, device, precision) in done:
                    continue
                try:
                    row = {"model": model_name,
                           **layout(core, MODELS[model_name], device, precision, args.cache_dir),
                           "cache_dir": args.cache_dir, "status": "success"}
                except Exception as error:  # an unsupported precision is a result too
                    row = {"model": model_name, "device": device, "requested": precision,
                           "status": "failed", "error": str(error).strip().splitlines()[0][:300]}
                results.append(row)
                out.write_text(json.dumps(results, indent=1), encoding="utf-8")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
