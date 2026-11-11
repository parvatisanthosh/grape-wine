"""ORBIT evaluation v2 on real-data traces (notes/evaluation_v2_plan.md).

This file holds the request model, the service limits and the per-request
decision for every system. The dry run predicts every request on every
configuration with the Performance Twin and runs nothing:

    python evaluate_v2.py dry-run --tuning traces/tuning.jsonl --test traces/test.jsonl

It answers, before any measurement, whether ORBIT and the best fixed
configuration would choose differently at all, and roughly how long the
measured runs will take.
"""

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path

import pandas as pd

import orbit
from build_dataset import load_rows, normalize
from twin_v0 import AnalyticalTwin


# Service limits, fixed in the plan before any v2 data existed.
LIMITS = {
    "chat": {"max_ttft_ms": 2000.0, "max_tpot_ms": 100.0, "needs_code_quality": False},
    "one_shot": {"max_ttft_ms": 2000.0, "max_tpot_ms": 100.0, "needs_code_quality": False},
    "code": {"max_ttft_ms": 1500.0, "max_tpot_ms": None, "needs_code_quality": True},
}
CODE_QUALITY_FLOOR = 0.50

# HumanEval pass rates (Finding 25 and the TinyLlama run). A label with no
# measured score cannot serve code requests.
HUMANEVAL_SUMMARIES = {
    "Q3B-INT4": (Path("results/quality_humaneval/summary.json"), "int4"),
    "FP16": (Path("results/quality_humaneval_tinyllama/summary.json"), "fp16"),
    "INT8": (Path("results/quality_humaneval_tinyllama/summary.json"), "int8"),
    "INT4": (Path("results/quality_humaneval_tinyllama/summary.json"), "int4"),
}

DEFAULT_CONFIGURATION = ("INT8", "cpu_auto")  # OpenVINO's default export, on CPU
RULE_SMALL = ("INT4", "gpu")
RULE_LARGE = ("Q3B-INT4", "gpu")
RULE_CONTEXT_TOKENS = 2048

# Machine state assumed by the dry run; the measured run reads the real one.
DRY_RUN_MACHINE = {"sys_cpu_busy_percent": 5.0, "sys_available_ram_mib": 6000.0}


# ---------------------------------------------------------------------------
# Requests


def load_trace(path):
    """Read a trace written by build_real_trace.py into request objects."""
    requests = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        limits = LIMITS[row["stream"]]
        requests.append(
            Namespace(
                index=row["index"],
                stream=row["stream"],
                scenario=row["stream"],
                conversation=row.get("conversation"),
                turn=row.get("turn", 0),
                capped=row.get("capped", False),
                prompt_id=row["prompt_id"],
                prompt_tokens=int(row["prompt_tokens"]),
                output_tokens=int(row["output_tokens"]),
                prompt_key=f"{row['prompt_id']}:{row['prompt_tokens']}",
                repeat_prefix=row["stream"] == "chat" and row.get("turn", 0) > 1,
                objective="latency",
                max_ttft_ms=limits["max_ttft_ms"],
                max_tpot_ms=limits["max_tpot_ms"],
                max_total_ms=None,
                max_memory_mib=None,
                needs_code_quality=limits["needs_code_quality"],
            )
        )
    return requests


def context_limits(models):
    """Largest prompt + output each model accepts (max_position_embeddings)."""
    limits = {}
    for label, entry in models.items():
        config = json.loads((Path(entry["path"]) / "config.json").read_text(encoding="utf-8"))
        limits[label] = int(config["max_position_embeddings"])
    return limits


def code_quality(labels, summaries=HUMANEVAL_SUMMARIES):
    """HumanEval pass rate per label, or None if it was never measured."""
    scores = {}
    for label in labels:
        path, tag = summaries.get(label, (None, None))
        if path is None or not path.exists():
            scores[label] = None
            continue
        model = json.loads(path.read_text(encoding="utf-8"))["models"].get(tag)
        scores[label] = model["pass_rate"] if model else None
    return scores


# ---------------------------------------------------------------------------
# Limits a configuration cannot break, and limits it may miss


def can_serve(label, request, contexts, scores):
    """None if the label may serve the request, otherwise the reason it may not."""
    if request.prompt_tokens + request.output_tokens > contexts[label]:
        return "context"
    if request.needs_code_quality:
        score = scores.get(label)
        if score is None or score < CODE_QUALITY_FLOOR:
            return "quality"
    return None


def add_tpot_limit(frame, request):
    """check_constraints covers TTFT, total and memory; add decode speed."""
    if not request.max_tpot_ms:
        return frame
    frame = frame.copy()
    slow = frame["pred_tpot_ms"] > request.max_tpot_ms
    frame.loc[slow, "violations"] = [
        ",".join(filter(None, [reasons, "tpot"])) for reasons in frame.loc[slow, "violations"]
    ]
    frame["feasible"] = frame["violations"] == ""
    frame["point_feasible"] = frame["point_feasible"] & ~slow
    frame["severity"] = [
        max(severity, tpot / request.max_tpot_ms - 1)
        for severity, tpot in zip(frame["severity"], frame["pred_tpot_ms"])
    ]
    return frame


def predicted_violations(row, request):
    """Which limits the predictions say this configuration would miss."""
    broken = []
    if request.max_ttft_ms and row["pred_ttft_ms"] > request.max_ttft_ms:
        broken.append("ttft")
    if request.max_tpot_ms and row["pred_tpot_ms"] > request.max_tpot_ms:
        broken.append("tpot")
    return broken


# ---------------------------------------------------------------------------
# Decisions


def candidate_frame(request, candidates, state, machine, twin, dataset):
    """Every candidate's predictions and limit checks for one request."""
    frame = orbit.build_candidate_frame(request, candidates, state, machine)
    frame = orbit.predict(frame, twin, dataset, state)
    frame = orbit.check_constraints(frame, request, machine["sys_available_ram_mib"])
    return add_tpot_limit(frame, request)


def orbit_choice(frame, request, contexts, scores, quality):
    """ORBIT's pick: the fastest candidate that keeps every limit, otherwise the
    smallest predicted miss. Candidates that may not serve the request at all
    (context or code quality) are never picked. None if none may serve it."""
    allowed = frame[[can_serve(label, request, contexts, scores) is None for label in frame["label"]]]
    if allowed.empty:
        return None, None
    return orbit.select(allowed, request.objective, quality, fallback=True)


def fixed_choice(identifier, frame):
    rows = frame[frame["candidate"] == identifier]
    return rows.iloc[0] if not rows.empty else None


def rule_choice(request):
    large = request.needs_code_quality or (
        request.prompt_tokens + request.output_tokens > RULE_CONTEXT_TOKENS
    )
    return orbit.candidate_id(*(RULE_LARGE if large else RULE_SMALL))


def predicted_outcome(row, request, contexts, scores):
    """(violated, reasons) for serving the request with this candidate row."""
    if row is None:
        return True, "unservable"
    refused = can_serve(row["label"], request, contexts, scores)
    if refused:
        return True, refused
    broken = predicted_violations(row, request)
    return bool(broken), ",".join(broken)


# ---------------------------------------------------------------------------
# Dry run


def dry_run_frames(trace, candidates, twin, dataset):
    """Predictions for every request and candidate, in a steady state where
    every candidate is resident and the GPU model cache is warm."""
    state = {
        "loaded_candidate": None,
        "gpu_cache_warm": sorted({label for label, _ in candidates}),
        "corrections": {},
        "resident": [orbit.candidate_id(*candidate) for candidate in candidates],
        "cached_prompts": {},
    }
    return [
        candidate_frame(request, candidates, state, DRY_RUN_MACHINE, twin, dataset)
        for request in trace
    ]


def score_fixed(trace, frames, candidates, contexts, scores):
    """Predicted violations and time if one candidate served every request."""
    rows = []
    for label, placement in candidates:
        identifier = orbit.candidate_id(label, placement)
        violated, total = 0, 0.0
        for request, frame in zip(trace, frames):
            row = fixed_choice(identifier, frame)
            broken, _ = predicted_outcome(row, request, contexts, scores)
            violated += broken
            if row is not None and can_serve(label, request, contexts, scores) is None:
                total += row["pred_total_ms"]
        rows.append({"candidate": identifier, "violations": violated, "served_total_s": total / 1000})
    return pd.DataFrame(rows).sort_values(["violations", "served_total_s"]).reset_index(drop=True)


def compare_systems(trace, frames, best_fixed, contexts, scores, quality):
    """Each system's predicted choice and outcome for every test request."""
    rows = []
    for request, frame in zip(trace, frames):
        choices = {
            "default": fixed_choice(orbit.candidate_id(*DEFAULT_CONFIGURATION), frame),
            "best_fixed": fixed_choice(best_fixed, frame),
            "rule": fixed_choice(rule_choice(request), frame),
        }
        choices["orbit"], _ = orbit_choice(frame, request, contexts, scores, quality)
        for system, row in choices.items():
            violated, reasons = predicted_outcome(row, request, contexts, scores)
            rows.append(
                {
                    "index": request.index,
                    "stream": request.stream,
                    "system": system,
                    "candidate": None if row is None else row["candidate"],
                    "violated": violated,
                    "reasons": reasons,
                    "pred_total_s": None if row is None else row["pred_total_ms"] / 1000,
                }
            )
    return pd.DataFrame(rows)


def report_dry_run(comparison, best_fixed):
    pivot = comparison.pivot(index="index", columns="system", values="candidate")
    differs = (pivot["orbit"] != pivot["best_fixed"]).sum()
    print(f"\nORBIT picks a different configuration from {best_fixed} on "
          f"{differs} of {len(pivot)} test requests.")
    print(comparison[comparison.system == "orbit"]
          .groupby(["stream", "candidate"]).size().rename("orbit picks").to_string())

    print("\nPredicted SLA violations per system and stream:")
    print(comparison.pivot_table(index="stream", columns="system", values="violated",
                                 aggfunc="sum", margins=True).to_string())

    violated = comparison.pivot(index="index", columns="system", values="violated")
    only_orbit = int((violated["orbit"] & ~violated["best_fixed"]).sum())
    only_fixed = int((~violated["orbit"] & violated["best_fixed"]).sum())
    print(f"\nPredicted discordant pairs, ORBIT vs best fixed: {only_fixed} only best fixed "
          f"broke, {only_orbit} only ORBIT broke.")

    hours = comparison.groupby("system")["pred_total_s"].sum() / 3600
    print("\nPredicted serving time per system (hours, excluding loads):")
    print(hours.round(2).to_string())


# ---------------------------------------------------------------------------
# Measured runs


RESULTS_DIR = Path("results/evaluation_v2")
ACTIVE_CHAT_USERS = 4
SWEEP_TOP = 5

# Two systems per pass: each keeps its own models loaded, and three at once need
# more memory than this laptop has free (plan section 10).
PASSES = {
    "A": ["orbit", "best_fixed"],  # the primary test
    "B": ["rule", "best_fixed_aa"],  # rule, and best fixed again for the noise floor
    "C": ["orbit_nocorr", "default"],
}
# How many models each system keeps loaded.
RESIDENT = {"orbit": 2, "orbit_nocorr": 2, "rule": 2}


def pool_for(trace, active_users=ACTIVE_CHAT_USERS):
    """KV pool per device: the active chat users' conversations, plus room for
    the largest request being served, plus a spare block each (Finding 26)."""
    chat = [r.prompt_tokens + r.output_tokens for r in trace if r.stream == "chat"]
    longest_chat = max(chat, default=0)
    longest = max(r.prompt_tokens + r.output_tokens for r in trace)
    return {
        device: active_users * math.ceil(longest_chat / size)
        + math.ceil(longest / size)
        + active_users
        + 1
        for device, size in orbit.KV_BLOCK_TOKENS.items()
    }


def judge(request, row, load_ms):
    """The measured outcome against the request's limits."""
    if row.get("status") != "success":
        return {"status": row.get("status"), "violated": True, "violations": "failed",
                "error": row.get("error")}

    broken = []
    if request.max_ttft_ms and row["ttft_ms"] > request.max_ttft_ms:
        broken.append("ttft")
    if request.max_tpot_ms and row["tpot_ms_per_token"] > request.max_tpot_ms:
        broken.append("tpot")
    # Free RAM is recorded but not judged: both systems of a pass keep their models
    # loaded, so it measures the whole machine, not the system under test (plan
    # section 11). Running out of memory shows up as a failed run above.
    available = row.get("sys_available_ram_mib")

    return {
        "status": "success",
        "load_ms_charged": load_ms,
        "ttft_ms": row["ttft_ms"],
        "tpot_ms": row["tpot_ms_per_token"],
        "total_ms": load_ms + row["generation_ms"],
        "peak_mib": row.get("sampled_peak_rss_mib"),
        "available_ram_mib": available,
        "cache_hit": bool(row.get("cache_hit")),
        "measured_hit": row.get("measured_hit"),
        "violated": bool(broken),
        "violations": ",".join(broken),
    }


def refusal(reason):
    return {"status": "not_served", "violated": True, "violations": reason}


def live_machine():
    import psutil

    return {
        "sys_cpu_busy_percent": psutil.cpu_percent(interval=0.2),
        "sys_available_ram_mib": psutil.virtual_memory().available / 2**20,
    }


class Context:
    """Everything a system needs to serve requests, loaded once."""

    def __init__(self, models, dataset, twin, candidates, contexts, scores, quality):
        self.models = models
        self.dataset = dataset
        self.twin = twin
        self.candidates = candidates
        self.contexts = contexts
        self.scores = scores
        self.quality = quality

    @classmethod
    def load(cls):
        models = json.loads(orbit.MODELS_JSON.read_text(encoding="utf-8"))
        dataset = normalize(load_rows())
        return cls(
            models,
            dataset,
            AnalyticalTwin().fit(dataset),
            orbit.observed_candidates(dataset, list(models)),
            context_limits(models),
            code_quality(list(models)),
            json.loads(Path("results/quality.json").read_text(encoding="utf-8")),
        )


def serve(system, request, service, state, context, fixed=None):
    """One system serves one request. Returns the record fields."""
    from evaluate_controller import charged_load_ms

    if system in ("orbit", "orbit_nocorr"):
        service.share_state(state)
        frame = candidate_frame(request, context.candidates, state, live_machine(),
                                context.twin, context.dataset)
        choice, mode = orbit_choice(frame, request, context.contexts, context.scores,
                                    context.quality)
        if choice is None:
            return {"candidate": None, "mode": None, **refusal("unservable")}
        label, placement = choice["label"], choice["placement"]
        switched = orbit.candidate_id(label, placement) not in service.resident
        if system == "orbit":
            row = orbit.run_and_learn(
                choice, request, context.models, state,
                executor=lambda chosen, args: service.run(chosen["label"], chosen["placement"], args),
            )
        else:
            row = service.run(label, placement, request)
        extra = {"mode": mode, "pred_ttft_ms": float(choice["pred_ttft_ms"]),
                 "pred_tpot_ms": float(choice["pred_tpot_ms"])}
    else:
        identifier = rule_choice(request) if system == "rule" else fixed
        label, placement = identifier.split("@")
        reason = can_serve(label, request, context.contexts, context.scores)
        if reason:
            return {"candidate": identifier, **refusal(reason)}
        switched = identifier not in service.resident
        row = service.run(label, placement, request)
        extra = {}

    candidate = orbit.candidate_id(label, placement)
    return {"candidate": candidate, "switched": switched,
            **judge(request, row, charged_load_ms(row, switched, "worker")), **extra}


def request_fields(request):
    return {"index": request.index, "stream": request.stream, "turn": request.turn,
            "prompt_tokens": request.prompt_tokens, "output_tokens": request.output_tokens,
            "capped": request.capped}


def done_keys(path, *fields):
    from quality_core import load_jsonl

    return {tuple(row[field] for field in fields) for row in load_jsonl(path)}


def append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=float) + "\n")


def sweep(trace_path, top, out, worker_factory=None, context=None):
    """Measure the predicted-best fixed configurations on a (tuning) trace."""
    from evaluate_controller import ResidentService
    from orbit_worker_client import OrbitWorker

    context = context or Context.load()
    trace = load_trace(trace_path)
    ranked = score_fixed(trace, dry_run_frames(trace, context.candidates, context.twin, context.dataset),
                         context.candidates, context.contexts, context.scores)
    chosen = list(ranked["candidate"].head(top))
    pool = pool_for(trace)
    print(f"Sweeping {chosen} on {len(trace)} requests; KV pool {pool}", flush=True)

    done = done_keys(out, "candidate", "index")
    for identifier in chosen:
        service = ResidentService(context.models, 1, worker_factory or OrbitWorker, kv_blocks=pool)
        try:
            for request in trace:
                if (identifier, request.index) in done:
                    continue
                record = serve("fixed", request, service, None, context, fixed=identifier)
                append(out, {"system": "fixed", **request_fields(request), **record,
                             "candidate": identifier})
        finally:
            service.close()
        print(f"  {identifier} done", flush=True)

    results = pd.DataFrame([row for row in __import__("quality_core").load_jsonl(out)])
    table = results.groupby("candidate").agg(violations=("violated", "sum"),
                                             requests=("violated", "size"))
    table = table.sort_values("violations")
    print(table.to_string())
    best = table.index[0]
    (out.parent / "best_fixed.json").write_text(
        json.dumps({"candidate": best, "violations": int(table.iloc[0]["violations"]),
                    "trace": str(trace_path), "swept": chosen}, indent=2),
        encoding="utf-8",
    )
    print(f"Best fixed configuration (measured): {best}")
    return best


def compare(trace_path, pass_name, best_fixed, out, seed=0, worker_factory=None, context=None,
            block_size=None):
    """Run one pass on the same requests for each of its systems.

    block_size None: the systems take turns on every request, in a shuffled order,
    all loaded at once. block_size N: the trace is cut into blocks of N requests;
    each system serves a block in turn (shuffled per block), and only one system is
    loaded at a time, so peak memory is one system's (plan section 12). ORBIT's
    learned state carries over between blocks; loaded models do not.
    """
    import random

    from evaluate_controller import ResidentService, interleaved_schedule
    from orbit_worker_client import OrbitWorker

    context = context or Context.load()
    trace = load_trace(trace_path)
    systems = PASSES[pass_name]
    pool = pool_for(trace)
    mode = f"blocks of {block_size}, one system loaded at a time" if block_size else "interleaved"
    print(f"Pass {pass_name}: {systems} on {len(trace)} requests ({mode}); best fixed {best_fixed}; "
          f"KV pool {pool}", flush=True)

    def new_service(system):
        return ResidentService(context.models, RESIDENT.get(system, 1),
                               worker_factory or OrbitWorker, kv_blocks=pool)

    states = {}
    for system in systems:
        state = {"loaded_candidate": None, "gpu_cache_warm": [], "corrections": {}}
        orbit.sync_gpu_cache_state(state, context.dataset)
        states[system] = state

    done = done_keys(out, "system", "index")

    def run(system, request, service):
        if (system, request.index) in done:
            return
        fixed = best_fixed if system.startswith("best_fixed") else None
        if system == "default":
            fixed = orbit.candidate_id(*DEFAULT_CONFIGURATION)
        record = serve(system, request, service, states[system], context, fixed=fixed)
        append(out, {"system": system, "pass": pass_name, **request_fields(request), **record})
        print(f"[{pass_name} {request.index}/{len(trace)}] {system:<14} {record['candidate']} "
              f"{'VIOLATED ' + record['violations'] if record['violated'] else 'ok'}", flush=True)

    if block_size:
        rng = random.Random(seed)
        for start in range(0, len(trace), block_size):
            block = trace[start:start + block_size]
            order = list(systems)
            rng.shuffle(order)
            for system in order:
                if all((system, request.index) in done for request in block):
                    continue
                service = new_service(system)
                try:
                    for request in block:
                        run(system, request, service)
                finally:
                    service.close()
        return

    services = {system: new_service(system) for system in systems}
    try:
        for index, request, system in interleaved_schedule(trace, systems, seed):
            run(system, request, services[system])
    finally:
        for service in services.values():
            service.close()


# ---------------------------------------------------------------------------
# Analysis (plan section 4)


def paired_stats(frame, first, second):
    """McNemar on violations, and the log ratio of total time where both served."""
    import numpy as np
    from scipy.stats import wilcoxon

    from quality_core import mcnemar_exact

    pivot = frame.pivot_table(index="index", columns="system", values="violated", aggfunc="first")
    pair = pivot[[first, second]].dropna().astype(bool)
    only_first_broke = int((pair[first] & ~pair[second]).sum())
    only_second_broke = int((~pair[first] & pair[second]).sum())
    result = {
        "pair": f"{first} vs {second}",
        "requests": len(pair),
        f"only {first} broke": only_first_broke,
        f"only {second} broke": only_second_broke,
        "mcnemar_p": mcnemar_exact(only_first_broke, only_second_broke),
    }

    served = frame[frame["status"] == "success"].pivot_table(
        index="index", columns="system", values="total_ms", aggfunc="first"
    )
    if first in served and second in served:
        both = served[[first, second]].dropna()
        if len(both) >= 10:
            ratio = np.log(both[first] / both[second]).to_numpy()
            rng = np.random.default_rng(0)
            boots = [np.median(rng.choice(ratio, len(ratio))) for _ in range(10_000)]
            result.update({
                "both_served": len(both),
                "median_time_ratio": float(np.exp(np.median(ratio))),
                "ratio_ci_low": float(np.exp(np.percentile(boots, 2.5))),
                "ratio_ci_high": float(np.exp(np.percentile(boots, 97.5))),
                "wilcoxon_p": float(wilcoxon(ratio).pvalue) if np.any(ratio != 0) else 1.0,
            })
    return result


def apply_corrections(frame, corrections, file_name=None):
    """Replace rows re-measured after a measurement fault (Finding 29).

    A correction row replaces the row with the same system and index (and, for
    the sweep, candidate) in the file it names. Returns the frame and the count.
    """
    if corrections is None or not len(corrections):
        return frame, 0
    keys = ["system", "index"] + (["candidate"] if file_name == "sweep_tuning.jsonl" else [])
    chosen = corrections if file_name is None else corrections[corrections["file"] == file_name]
    if file_name is None:
        chosen = chosen[chosen["file"].str.startswith("pass_")]
    chosen = chosen.drop(columns=["file"])
    matched = frame.merge(chosen[keys], on=keys, how="left", indicator=True)["_merge"] == "both"
    kept = frame[~matched.to_numpy()]
    return pd.concat([kept, chosen], ignore_index=True), int(matched.sum())


def sweep_table(frame):
    return (frame.groupby("candidate").agg(violations=("violated", "sum"), requests=("violated", "size"))
            .sort_values("violations"))


def analyze_frame(frame):
    print(frame.pivot_table(index="stream", columns="system", values="violated",
                            aggfunc="sum", margins=True).to_string())
    pairs = [("orbit", "best_fixed"), ("best_fixed", "best_fixed_aa"), ("orbit", "rule"),
             ("orbit", "default"), ("orbit", "orbit_nocorr")]
    results = [paired_stats(frame, a, b) for a, b in pairs
               if a in set(frame["system"]) and b in set(frame["system"])]
    for result in results:
        print(json.dumps(result, indent=1, default=float))
    return results


def analyze(paths, corrections_path=None, sweep_path=None):
    """Plan section 4. With corrections, prints the results as measured, then
    with the re-measured rows in place (both are reported)."""
    from quality_core import load_jsonl

    frame = pd.DataFrame([row for path in paths for row in load_jsonl(path)])
    corrections = None
    if corrections_path and Path(corrections_path).exists():
        corrections = pd.DataFrame(load_jsonl(corrections_path))

    if sweep_path and Path(sweep_path).exists():
        sweep_rows = pd.DataFrame(load_jsonl(sweep_path))
        print("Tuning sweep, as measured:")
        print(sweep_table(sweep_rows).to_string())
        if corrections is not None:
            fixed_sweep, count = apply_corrections(sweep_rows, corrections, "sweep_tuning.jsonl")
            print(f"Tuning sweep, with {count} re-measured rows:")
            print(sweep_table(fixed_sweep).to_string())
        print()

    print("As measured:")
    results = analyze_frame(frame)
    if corrections is not None:
        corrected, count = apply_corrections(frame, corrections)
        print()
        print(f"With corrections: {count} rows re-measured with unique prompts (Finding 29):")
        results = analyze_frame(corrected)
    print("Only orbit vs best_fixed ran interleaved, in pass A. Every other pair spans two "
          "passes, so it is paired by request but not run at the same time.")
    return results


def main():
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "colab"))  # quality_core statistics

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="Predict every request; run nothing.")
    dry.add_argument("--tuning", type=Path, required=True)
    dry.add_argument("--test", type=Path, required=True)

    sweep_parser = sub.add_parser("sweep", help="Measure fixed configurations on the tuning trace.")
    sweep_parser.add_argument("--trace", type=Path, required=True)
    sweep_parser.add_argument("--top", type=int, default=SWEEP_TOP)
    sweep_parser.add_argument("--out", type=Path, default=RESULTS_DIR / "sweep_tuning.jsonl")
    sweep_parser.add_argument("--allow-battery", action="store_true")

    compare_parser = sub.add_parser("compare", help="Run pass A or B on the test trace.")
    compare_parser.add_argument("--trace", type=Path, required=True)
    compare_parser.add_argument("--pass", dest="pass_name", choices=sorted(PASSES), required=True)
    compare_parser.add_argument("--best-fixed", default=None,
                                help="Default: results/evaluation_v2/best_fixed.json")
    compare_parser.add_argument("--seed", type=int, default=0)
    compare_parser.add_argument("--block-size", type=int, default=None,
                                help="Serve blocks of N requests per system, one system loaded at a time.")
    compare_parser.add_argument("--allow-battery", action="store_true")

    analyze_parser = sub.add_parser("analyze", help="Statistics from the pass results.")
    analyze_parser.add_argument("paths", type=Path, nargs="+")
    analyze_parser.add_argument("--corrections", type=Path, default=None,
                                help="Re-measured rows, e.g. results/evaluation_v2/corrections.jsonl")
    analyze_parser.add_argument("--sweep", type=Path, default=None,
                                help="Also summarise the tuning sweep, e.g. results/evaluation_v2/sweep_tuning.jsonl")

    args = parser.parse_args()

    if args.command == "analyze":
        analyze(args.paths, args.corrections, args.sweep)
        return

    if args.command in ("sweep", "compare"):
        from run_experiment import prevent_sleep, require_ac_power

        require_ac_power(args.allow_battery)
        prevent_sleep()

    if args.command == "sweep":
        sweep(args.trace, args.top, args.out)
        return

    if args.command == "compare":
        best = args.best_fixed or json.loads(
            (RESULTS_DIR / "best_fixed.json").read_text(encoding="utf-8"))["candidate"]
        compare(args.trace, args.pass_name, best,
                RESULTS_DIR / f"pass_{args.pass_name}.jsonl", args.seed, block_size=args.block_size)
        return

    context = Context.load()
    print("Context limits:", context.contexts)
    print("HumanEval pass rates:",
          {k: (None if v is None else round(v, 3)) for k, v in context.scores.items()})
    tuning = load_trace(args.tuning)
    test = load_trace(args.test)

    fixed = score_fixed(tuning, dry_run_frames(tuning, context.candidates, context.twin, context.dataset),
                        context.candidates, context.contexts, context.scores)
    print(f"\nBest fixed configuration on the tuning trace ({len(tuning)} requests), predicted:")
    print(fixed.head(6).round(1).to_string(index=False))
    best_fixed = fixed.iloc[0]["candidate"]

    comparison = compare_systems(test, dry_run_frames(test, context.candidates, context.twin, context.dataset),
                                 best_fixed, context.contexts, context.scores, context.quality)
    report_dry_run(comparison, best_fixed)


if __name__ == "__main__":
    main()
