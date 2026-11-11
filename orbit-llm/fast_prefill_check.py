"""Why do a few 4,096-token prompts report a ~50 ms TTFT on the 3B model?

Evaluation v2 found five requests (test 247 and 286, tuning 184, 239 and 288)
whose prefill took almost no time, every time they ran, while other prompts of
the same length took about 23 s. This script runs them again on fresh
pipelines, with and without prefix caching, next to a normal prompt of the
same length, to separate a measurement artefact from real work skipped.

Run from orbit-llm/ on mains power:  python fast_prefill_check.py
Output: results/fast_prefill_check.jsonl
"""

import json
from pathlib import Path

import orbit
from evaluate_controller import load_options
from orbit_worker_client import OrbitWorker
from run_experiment import prevent_sleep, require_ac_power

OUT = Path("results/fast_prefill_check.jsonl")
MODEL = "Q3B-INT4"
# (prompt id, prompt tokens, output tokens): two fast ones, one normal control.
PROMPTS = [("run_3#r286", 4096, 41), ("run_2#r247", 4093, 62), ("run_3#r282", 4096, 12)]
KEEP = ["prompt_id", "actual_input_tokens", "actual_output_tokens", "ttft_ms",
        "tpot_ms_per_token", "generation_ms", "wall_time_ms", "status", "error"]


def main():
    require_ac_power(False)
    prevent_sleep()
    path = json.loads(orbit.MODELS_JSON.read_text(encoding="utf-8"))[MODEL]["path"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for prefix_caching in ("on", "off"):
            for prompt_id, prompt_tokens, output_tokens in PROMPTS:
                # A fresh worker and pipeline for every prompt: nothing can be cached.
                worker = OrbitWorker()
                try:
                    options = load_options(MODEL, "gpu")
                    options.update(kv_blocks=506, prefix_caching=prefix_caching)
                    loaded = worker.load("s", path, **options)
                    if loaded["status"] != "success":
                        raise RuntimeError(loaded)
                    for attempt in (1, 2):  # the second run of the same prompt may hit
                        row = worker.generate("s", prompt_tokens, output_tokens, prompt_id=prompt_id)
                        record = {"prefix_caching": prefix_caching, "attempt": attempt,
                                  **{key: row.get(key) for key in KEEP}}
                        print(json.dumps(record), flush=True)
                        out.write(json.dumps(record) + "\n")
                        out.flush()
                finally:
                    worker.close()


if __name__ == "__main__":
    main()
