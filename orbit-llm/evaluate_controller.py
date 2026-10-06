"""Phase 8 evaluation: baseline OpenVINO vs a static optimized configuration
vs the ORBIT-LLM controller, on the same request trace, all measured.

Systems
- baseline: out-of-the-box OpenVINO. optimum-cli exports models above 1B
  parameters with INT8 weights by default; CPU device, default settings.
- static:   the single best hand-picked configuration from the experiments
  (FINDINGS Phase 5): INT4 on the iGPU, used for every request.
- orbit:    orbit.py decides per request (Layer 2) and corrects its
  predictions after every run (Layer 3).

Every request is executed through the C++ executor. Each system is treated
as a long-lived service: model load time is charged only when the system
switches configuration, using the load time actually measured.

A request violates its SLA if it fails, or if its measured TTFT, total time
(load if switching + generation) or peak memory exceeds its constraint.
"""

import argparse
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import psutil

import orbit
from build_dataset import load_rows, normalize
from orbit_exec import run_configuration
from run_experiment import prevent_sleep
from twin_v0 import AnalyticalTwin


RESULTS_JSONL = Path("results/evaluation_runs.jsonl")
SUMMARY_CSV = Path("results/evaluation_summary.csv")

BASELINE = ("INT8", "cpu_auto")
STATIC = ("INT4", "gpu")

# (name, prompt tokens, output tokens, repeated prefix, objective, constraints)
SCENARIOS = [
    ("short_chat", 64, 64, False, "latency", {"max_total_ms": 5000}),
    ("long_document", 1920, 128, False, "latency", {"max_ttft_ms": 4000}),
    ("follow_up", 1920, 64, True, "latency", {"max_ttft_ms": 1000}),
    ("memory_capped", 512, 64, False, "latency", {"max_memory_mib": 1500}),
    ("quality_first", 256, 128, False, "quality", {"max_total_ms": 20000}),
]

REPEATS = 2


def get_arguments():
    parser = argparse.ArgumentParser(description="Run the Phase 8 evaluation.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["baseline", "static", "orbit"],
        choices=["baseline", "static", "orbit"],
    )
    return parser.parse_args()


def build_trace():
    trace = []

    for repeat in range(REPEATS):
        for name, prompt, output, repeat_prefix, objective, limits in SCENARIOS:
            trace.append(
                Namespace(
                    scenario=name,
                    repeat=repeat + 1,
                    prompt_tokens=prompt,
                    output_tokens=output,
                    repeat_prefix=repeat_prefix,
                    objective=objective,
                    max_ttft_ms=limits.get("max_ttft_ms"),
                    max_total_ms=limits.get("max_total_ms"),
                    max_memory_mib=limits.get("max_memory_mib"),
                )
            )

    return trace


def execute(label, placement, request, models, loaded):
    """Run one request on one configuration; return the measured row."""
    settings = orbit.PLACEMENTS[placement]
    identifier = orbit.candidate_id(label, placement)

    # A prefix-cache hit needs the same pipeline to still be loaded.
    cache_hit = request.repeat_prefix and loaded == identifier

    configuration = {
        "label": label,
        "prompt_tokens": request.prompt_tokens,
        "output_tokens": request.output_tokens,
        "runs": 1,
        "prompt_mode": "repeat" if cache_hit else "cold",
        "device": settings["device"],
    }

    if settings["device"] == "CPU":
        configuration["cores"] = settings["cores"]
        configuration["threads"] = settings["threads"]
    else:
        configuration["cache_dir"] = orbit.GPU_CACHE_DIR

    return run_configuration(models[label]["path"], configuration)[-1]


def orbit_decide(request, state, dataset, twin, models, quality):
    memory = psutil.virtual_memory()
    machine = {
        "sys_cpu_busy_percent": psutil.cpu_percent(interval=0.5),
        "sys_available_ram_mib": memory.available / 2**20,
    }

    candidates = orbit.observed_candidates(dataset, list(models))
    frame = orbit.build_candidate_frame(request, candidates, state, machine)
    frame = orbit.predict(frame, twin, dataset, state)
    frame = orbit.check_constraints(
        frame,
        request,
        machine["sys_available_ram_mib"],
    )

    return orbit.choose(frame, request.objective, quality)


def judge(request, row, switched, quality):
    if row["status"] != "success":
        return {
            "status": row["status"],
            "violated": True,
            "violations": "failed",
        }

    load_ms = row["pipeline_construction_ms"] if switched else 0.0
    total_ms = load_ms + row["generation_ms"]

    broken = []
    if request.max_ttft_ms and row["ttft_ms"] > request.max_ttft_ms:
        broken.append("ttft")
    if request.max_total_ms and total_ms > request.max_total_ms:
        broken.append("total")
    if request.max_memory_mib and row["sampled_peak_rss_mib"] > request.max_memory_mib:
        broken.append("memory")

    return {
        "status": "success",
        "load_ms_charged": load_ms,
        "ttft_ms": row["ttft_ms"],
        "tpot_ms": row["tpot_ms_per_token"],
        "total_ms": total_ms,
        "peak_mib": row["sampled_peak_rss_mib"],
        "bits_per_byte": quality.get(row["label"], {}).get("bits_per_byte"),
        "violated": bool(broken),
        "violations": ",".join(broken),
    }


def main():
    args = get_arguments()

    models = json.loads(orbit.MODELS_JSON.read_text(encoding="utf-8"))
    quality_path = Path("results/quality.json")
    quality = (
        json.loads(quality_path.read_text(encoding="utf-8"))
        if quality_path.exists()
        else {}
    )

    if not quality:
        raise SystemExit("Run quality_eval.py first (results/quality.json).")

    prevent_sleep()

    dataset = normalize(load_rows())
    twin = AnalyticalTwin().fit(dataset)
    trace = build_trace()

    records = []

    for system in args.systems:
        loaded = None
        # ORBIT learns within its own run only, starting from no corrections.
        state = {"loaded_candidate": None, "gpu_cache_warm": [], "corrections": {}}

        for index, request in enumerate(trace, start=1):
            if system == "baseline":
                label, placement = BASELINE
                choice = None
            elif system == "static":
                label, placement = STATIC
                choice = None
            else:
                choice = orbit_decide(request, state, dataset, twin, models, quality)

                if choice is None:
                    record = {
                        "system": system,
                        "index": index,
                        "scenario": request.scenario,
                        "chosen": None,
                        "status": "no_feasible_candidate",
                        "violated": True,
                        "violations": "no-candidate",
                    }
                    records.append(record)
                    continue

                label, placement = choice["label"], choice["placement"]

            identifier = orbit.candidate_id(label, placement)
            switched = loaded != identifier

            print(
                f"\n[{system} {index}/{len(trace)}] {request.scenario} "
                f"-> {identifier}{' (switch)' if switched else ''}",
                flush=True,
            )

            if system == "orbit":
                row = orbit.run_and_learn(choice, request, models, state)
            else:
                row = execute(label, placement, request, models, loaded)

            loaded = identifier if row["status"] == "success" else None

            record = {
                "system": system,
                "index": index,
                "scenario": request.scenario,
                "repeat": request.repeat,
                "chosen": identifier,
                "switched": switched,
                **judge(request, row, switched, quality),
            }
            if choice is not None:
                record["predicted_total_ms"] = float(choice["pred_total_ms"])

            records.append(record)

            RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
            with RESULTS_JSONL.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, default=float) + "\n")

    frame = pd.DataFrame(records)
    summary = frame.groupby("system").agg(
        requests=("index", "count"),
        sla_violation_rate=("violated", "mean"),
        mean_total_s=("total_ms", lambda values: values.mean() / 1000),
        mean_ttft_s=("ttft_ms", lambda values: values.mean() / 1000),
        mean_peak_mib=("peak_mib", "mean"),
        mean_bits_per_byte=("bits_per_byte", "mean"),
        switches=("switched", "sum"),
    )
    summary["sla_violation_rate"] *= 100
    summary.to_csv(SUMMARY_CSV)

    print("\n" + summary.round(2).to_string())

    by_scenario = frame.pivot_table(
        index="scenario",
        columns="system",
        values="violated",
        aggfunc="mean",
    )
    print("\nSLA violation rate by scenario:")
    print((100 * by_scenario).round(0).to_string())

    print(f"\nRuns: {RESULTS_JSONL.resolve()}")
    print(f"Summary: {SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
