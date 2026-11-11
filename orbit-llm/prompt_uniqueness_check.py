"""Are two prompts with different tags really different token sequences?

orbit_run/orbit_worker give every request its own prompt by writing a tag
("Request r286. ") at the start of a long repeated text, then keeping the first
N token ids. Both models' tokenizer IRs contain a Truncate step on the LEFT
(TinyLlama: 2,047 tokens, Qwen2.5: 32,768). If GenAI applies it, a long text
loses its tag, and prompts with different tags become identical, so they hit
the prefix cache when they should not.

The check: on a fresh pipeline with prefix caching on, send prompt X, then a
prompt Y with a different tag and the same length. A miss on Y means the tags
survived; a hit (TTFT far below the cold one) means they were cut off.

Run from orbit-llm/ on mains power:  python prompt_uniqueness_check.py
Output: results/prompt_uniqueness_check.jsonl
"""

import json
from pathlib import Path

import orbit
from evaluate_controller import load_options
from orbit_worker_client import OrbitWorker
from run_experiment import prevent_sleep, require_ac_power

OUT = Path("results/prompt_uniqueness_check.jsonl")
# (model, placement, prompt tokens, first tag, second tag)
CASES = [
    ("INT4", "gpu", 1500, "run_1#c1", "run_1#c2"),
    ("INT4", "gpu", 1500, "run_1#c2", "run_1#c10"),
    ("Q3B-INT4", "gpu", 2000, "run_3#r282", "run_3#r286"),
    ("Q3B-INT4", "gpu", 4096, "run_3#r282", "run_3#r286"),
]
OUTPUT_TOKENS = 4


def main():
    require_ac_power(False)
    prevent_sleep()
    models = json.loads(orbit.MODELS_JSON.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as out:
        for label, placement, tokens, first, second in CASES:
            worker = OrbitWorker()  # fresh pipeline: an empty prefix cache
            try:
                options = load_options(label, placement)
                options["prefix_caching"] = "on"
                loaded = worker.load("s", models[label]["path"], **options)
                if loaded["status"] != "success":
                    raise RuntimeError(loaded)
                for step, prompt_id in (("first", first), ("other_tag", second), ("same_again", first)):
                    row = worker.generate("s", tokens, OUTPUT_TOKENS, prompt_id=prompt_id)
                    record = {"model": label, "placement": placement, "prompt_tokens": tokens,
                              "step": step, "prompt_id": prompt_id, "ttft_ms": row.get("ttft_ms"),
                              "status": row.get("status"), "error": row.get("error")}
                    print(json.dumps(record), flush=True)
                    out.write(json.dumps(record) + "\n")
                    out.flush()
            finally:
                worker.close()


if __name__ == "__main__":
    main()
