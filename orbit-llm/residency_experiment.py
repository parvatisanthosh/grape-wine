"""Measure model residency with the persistent worker.

The controller evaluation assumes a loaded model stays loaded and charges load
time only on a switch, but every run there is really a fresh orbit_run.exe that
reloads the model. This experiment measures what the assumption hides:

  fresh             a fresh process per request: load + generation, every time
  first_after_load  first request on a newly loaded worker slot
  resident_miss     model already loaded, new prompt (no prefix-cache reuse)
  resident_hit      model already loaded, prompt seen before on this slot
  switch            unload one model, load another (both measured)
  after_switch      first request after a switch (the prefix cache is gone)
  two_resident      two models loaded at once, requests alternating between them

Each round runs the fresh-process phase and the worker session back to back;
the order of the two alternates between rounds so laptop drift affects both.

    python residency_experiment.py
    python residency_experiment.py --rounds 3
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from orbit_exec import run_configuration
from orbit_worker_client import OrbitWorker

MODELS_JSON = Path("models.json")
OUTPUT = Path("results/experiments/residency.jsonl")

PROMPT_TOKENS = 1024
OUTPUT_TOKENS = 64
GPU_CACHE_DIR = "ov_cache/gpu"

# (slot name, model label) for the two models a quality-aware controller
# switches between.
SMALL = ("INT4@gpu", "INT4")
LARGE = ("Q3B@gpu", "Q3B-INT4")


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3, help="Requests per measured step.")
    return parser.parse_args()


def record(records, round_number, phase, step, reply, **extra):
    row = {
        "round": round_number,
        "phase": phase,
        "step": step,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
        **reply,
    }
    records.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")

    if reply.get("status") not in (None, "success"):
        print(f"  !! {step}: {reply.get('status')} {reply.get('error', '')}", flush=True)
    return row


def fresh_phase(records, round_number, models, repeats):
    """The old way: a new orbit_run.exe per request."""
    print(f"\n[round {round_number}] fresh process per request", flush=True)
    label = SMALL[1]
    for _ in range(repeats):
        rows = run_configuration(
            models[label]["path"],
            {
                "label": label,
                "device": "GPU",
                "cache_dir": GPU_CACHE_DIR,
                "prompt_tokens": PROMPT_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "warmup": 0,
                "runs": 1,
                "prompt_mode": "cold",
            },
        )
        row = rows[-1]
        total = None
        if row.get("status") == "success":
            total = row["pipeline_construction_ms"] + row["generation_ms"]
        record(records, round_number, "fresh", "fresh", row, service_ms=total)


def worker_phase(records, round_number, models, repeats):
    """The new way: one worker, models kept loaded."""
    print(f"\n[round {round_number}] persistent worker", flush=True)
    small_slot, small_label = SMALL
    large_slot, large_label = LARGE
    gpu = {"device": "GPU", "cache_dir": GPU_CACHE_DIR}

    def generate(phase, step, slot, prompt_id):
        reply = worker.generate(slot, PROMPT_TOKENS, OUTPUT_TOKENS, prompt_id=prompt_id)
        record(records, round_number, phase, step, reply, service_ms=reply.get("generation_ms"))

    with OrbitWorker() as worker:
        record(records, round_number, "worker", "load",
               worker.load(small_slot, models[small_label]["path"], label=small_label, **gpu))

        # First request after loading, then new prompts and repeated prompts.
        generate("worker", "first_after_load", small_slot, "run_1")
        for prompt_id in ["run_2", "run_3", "run_4"][:repeats]:
            generate("worker", "resident_miss", small_slot, prompt_id)
        for _ in range(repeats):
            generate("worker", "resident_hit", small_slot, "run_1")

        # A real switch to the larger model and back, each half measured.
        record(records, round_number, "worker", "switch_unload", worker.unload(small_slot))
        record(records, round_number, "worker", "switch_load",
               worker.load(large_slot, models[large_label]["path"], label=large_label, **gpu))
        generate("worker", "after_switch", large_slot, "run_1")

        record(records, round_number, "worker", "switch_unload", worker.unload(large_slot))
        record(records, round_number, "worker", "switch_load",
               worker.load(small_slot, models[small_label]["path"], label=small_label, **gpu))
        # run_1 was cached before the switch; unloading threw that cache away.
        generate("worker", "after_switch", small_slot, "run_1")

        # Both models loaded at once: no switching, but more memory.
        record(records, round_number, "worker", "second_load",
               worker.load(large_slot, models[large_label]["path"], label=large_label, **gpu))
        for prompt_id in ["run_4", "run_5"]:
            generate("worker", "two_resident", small_slot, prompt_id)
            generate("worker", "two_resident", large_slot, prompt_id)

        record(records, round_number, "worker", "final_unload", worker.unload(small_slot))
        record(records, round_number, "worker", "final_unload", worker.unload(large_slot))


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarize(records):
    def rows(step, slot=None):
        return [
            row for row in records
            if row["step"] == step and row.get("status") == "success"
            and (slot is None or row.get("slot") == slot)
        ]

    def show(name, selected, key):
        value = median(row.get(key) for row in selected)
        text = "-" if value is None else f"{value:,.0f} ms"
        print(f"  {name:<44}{text:>12}   (n={len(selected)})")

    print("\n" + "=" * 70)
    print(f"RESIDENCY SUMMARY ({SMALL[1]} on GPU, {PROMPT_TOKENS} -> {OUTPUT_TOKENS} tokens, medians)")
    print("=" * 70)
    fresh = rows("fresh")
    show("fresh process: model load", fresh, "pipeline_construction_ms")
    show("fresh process: TTFT", fresh, "ttft_ms")
    show("fresh process: total service time", fresh, "service_ms")
    show("worker: first request after load, TTFT", rows("first_after_load"), "ttft_ms")
    show("worker: resident, new prompt, TTFT", rows("resident_miss"), "ttft_ms")
    show("worker: resident, repeated prompt, TTFT", rows("resident_hit"), "ttft_ms")
    show("worker: resident, new prompt, total", rows("resident_miss"), "service_ms")
    show("switch: unload", rows("switch_unload"), "unload_ms")
    show(f"switch: load {LARGE[1]}", rows("switch_load", LARGE[0]), "pipeline_construction_ms")
    show(f"switch: load {SMALL[1]}", rows("switch_load", SMALL[0]), "pipeline_construction_ms")
    show(f"after switch back: repeated prompt TTFT", rows("after_switch", SMALL[0]), "ttft_ms")
    show(f"two resident: {SMALL[1]} TTFT", rows("two_resident", SMALL[0]), "ttft_ms")
    show(f"two resident: {LARGE[1]} TTFT", rows("two_resident", LARGE[0]), "ttft_ms")

    print("\nMemory")
    for step, slot, key, name in [
        ("load", SMALL[0], "commit_after_mib", f"commit with {SMALL[1]} loaded"),
        ("second_load", LARGE[0], "commit_after_mib", "commit with both loaded"),
        ("final_unload", "last", "commit_after_mib", "commit after unloading everything"),
        ("switch_unload", None, "freed_commit_mib", "commit freed per unload"),
    ]:
        if slot == "last":
            # Only the last unload of each round leaves nothing loaded.
            selected = [row for row in rows(step) if row.get("loaded_slots") == ""]
        else:
            selected = rows(step, slot)
        value = median(row.get(key) for row in selected)
        text = "-" if value is None else f"{value:,.0f} MiB"
        print(f"  {name:<44}{text:>12}   (n={len(selected)})")

    print(f"\nRows: {OUTPUT.resolve()}")


def main():
    args = get_arguments()
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    records = []

    for round_number in range(1, args.rounds + 1):
        # Alternate which phase goes first, so drift over time hits both.
        phases = [fresh_phase, worker_phase]
        if round_number % 2 == 0:
            phases.reverse()
        for phase in phases:
            phase(records, round_number, models, args.repeats)

    summarize(records)


if __name__ == "__main__":
    main()