"""ORBIT-LLM controller: choose an inference configuration for one request.

Layer 2 (optimizer): enumerate candidate configurations that have measured
data, predict each one's load time, TTFT, decode time and peak memory with
the Performance Twin, drop candidates that break the user's constraints or
would not fit in currently free RAM, and pick the best by the objective.

Layer 3 (online correction): with --run, execute the choice through the C++
executor, compare actual with predicted, and update a per-candidate
correction factor that is applied to every later prediction.

Example:
    python orbit.py --prompt-tokens 1024 --output-tokens 128 --max-ttft-ms 3000
    python orbit.py --prompt-tokens 512 --output-tokens 64 --objective memory --run
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from build_dataset import load_rows, normalize
from orbit_exec import run_configuration
from twin_v0 import AnalyticalTwin


MODELS_JSON = Path("models.json")
STATE_JSON = Path("results/orbit_state.json")
DECISIONS_JSONL = Path("results/orbit_decisions.jsonl")

GPU_CACHE_DIR = "ov_cache/gpu"

# Hardware placements the controller can choose (FINDINGS Phase 5).
PLACEMENTS = {
    "cpu_auto": {"device": "CPU", "cores": "any", "threads": 0},
    "cpu_pcore": {"device": "CPU", "cores": "pcore", "threads": 0},
    "cpu_ecore": {"device": "CPU", "cores": "ecore", "threads": 0},
    "cpu_4t": {"device": "CPU", "cores": "any", "threads": 4},
    "cpu_8t": {"device": "CPU", "cores": "any", "threads": 8},
    # The model cache cuts GPU load from ~8 s to ~2.5 s and is never used
    # on CPU, where it makes loading slower (Finding 13).
    "gpu": {"device": "GPU", "cores": "any", "threads": 0},
}

# A candidate needs at least this many measured runs to be considered.
MIN_OBSERVED_RUNS = 3

# Keep this much RAM free beyond the predicted peak (MiB).
RAM_HEADROOM_MIB = 512

# Weight of the newest observation in the correction factors.
CORRECTION_ALPHA = 0.3

# One run's measured/predicted ratio is clipped to this range before it is
# averaged in, so a single freak run (a background update, a failed warm-up)
# cannot drag a factor far from the truth.
RATIO_MIN = 0.5
RATIO_MAX = 3.0

# Memory is corrected asymmetrically. Under-predicting memory risks an
# out-of-memory crash, so a measurement above the prediction is learned fast;
# over-predicting only wastes some headroom, so that is learned slowly.
PEAK_ALPHA_UNDER = 0.6
PEAK_ALPHA_OVER = 0.2


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Choose (and optionally run) an inference configuration."
    )

    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)

    parser.add_argument(
        "--objective",
        choices=["latency", "memory", "quality"],
        default="latency",
        help="What to optimize among feasible candidates.",
    )
    parser.add_argument("--max-ttft-ms", type=float)
    parser.add_argument("--max-total-ms", type=float)
    parser.add_argument("--max-memory-mib", type=float)

    parser.add_argument(
        "--models",
        nargs="+",
        help="Restrict to these model labels (default: all in models.json).",
    )
    parser.add_argument(
        "--repeat-prefix",
        action="store_true",
        help=(
            "The prompt repeats a prefix the currently loaded pipeline has "
            "already processed (prefix-cache hit, Finding 8)."
        ),
    )
    parser.add_argument(
        "--explore",
        type=float,
        default=0.0,
        help=(
            "Probability (0-1) of running a different feasible candidate "
            "instead of the best one, so stale corrections get re-measured."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the chosen configuration and learn from the result.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# State: what is loaded, which GPU caches are warm, correction factors.


def load_state():
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))

    return {
        "loaded_candidate": None,
        "gpu_cache_warm": [],
        "corrections": {},
    }


def save_state(state):
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Candidates and predictions


def candidate_id(label, placement):
    return f"{label}@{placement}"


def observed_candidates(dataset, labels):
    """(label, placement) pairs with enough measured default-config runs."""
    defaults = dataset[
        (dataset["backend"] == "pa")
        & (dataset["kv_precision"].isin(["u8", "gpu_default"]))
        & (dataset["cache_size_gb"] == 0)
    ]

    candidates = []

    for label in labels:
        for placement, settings in PLACEMENTS.items():
            matching = defaults[
                (defaults["precision"] == label)
                & (defaults["device"] == settings["device"])
                & (defaults["cores"] == settings["cores"])
                & (defaults["threads"] == settings["threads"])
            ]

            if len(matching) >= MIN_OBSERVED_RUNS:
                candidates.append((label, placement))

    return candidates


def predict_load_ms(dataset, label, device, cache_warm):
    rows = dataset[
        (dataset["precision"] == label)
        & (dataset["device"] == device)
        & dataset["load_ms"].notna()
    ]

    if device == "GPU":
        cached = rows[rows["model_cache"]]
        uncached = rows[~rows["model_cache"]]
        rows = cached if (cache_warm and len(cached)) else uncached

        if len(rows) == 0:
            # No per-model data: fall back to all GPU loads of that kind.
            pool = dataset[
                (dataset["device"] == "GPU")
                & (dataset["model_cache"] == cache_warm)
            ]
            return float(pool["load_ms"].median())

    return float(rows["load_ms"].median())


def build_candidate_frame(args, candidates, state, machine):
    rows = []

    for label, placement in candidates:
        settings = PLACEMENTS[placement]
        identifier = candidate_id(label, placement)

        # A prefix-cache hit is only possible on the pipeline that already
        # holds the prefix; switching configuration loses the cache.
        cache_hit = int(
            args.repeat_prefix
            and state["loaded_candidate"] == identifier
        )

        rows.append(
            {
                "candidate": identifier,
                "label": label,
                "placement": placement,
                "precision": label,
                "device": settings["device"],
                "backend": "pa",
                "cores": settings["cores"],
                "threads": settings["threads"],
                "kv_precision": (
                    "u8" if settings["device"] == "CPU" else "gpu_default"
                ),
                "prefix_caching": True,
                "cache_size_gb": 0,
                "prompt_mode": "repeat" if cache_hit else "cold",
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
                "cache_hit": cache_hit,
                **machine,
            }
        )

    return pd.DataFrame(rows)


def predict(frame, twin, dataset, state):
    raw = twin.predict(frame)

    frame = frame.copy()
    frame["raw_ttft_ms"] = raw["ttft_ms"]
    frame["raw_tpot_ms"] = raw["tpot_ms_per_token"]
    frame["raw_peak_mib"] = raw["sampled_peak_rss_mib"]

    ttft_factor = []
    tpot_factor = []
    peak_factor = []

    for identifier in frame["candidate"]:
        correction = state["corrections"].get(identifier, {})
        ttft_factor.append(correction.get("ttft_factor", 1.0))
        tpot_factor.append(correction.get("tpot_factor", 1.0))
        peak_factor.append(correction.get("peak_factor", 1.0))

    frame["pred_ttft_ms"] = frame["raw_ttft_ms"] * np.array(ttft_factor)
    frame["pred_tpot_ms"] = frame["raw_tpot_ms"] * np.array(tpot_factor)
    frame["pred_peak_mib"] = frame["raw_peak_mib"] * np.array(peak_factor)

    load = []

    for _, row in frame.iterrows():
        if state["loaded_candidate"] == row["candidate"]:
            load.append(0.0)
        else:
            load.append(
                predict_load_ms(
                    dataset,
                    row["label"],
                    row["device"],
                    row["label"] in state["gpu_cache_warm"],
                )
            )

    frame["pred_load_ms"] = load
    frame["pred_total_ms"] = (
        frame["pred_load_ms"]
        + frame["pred_ttft_ms"]
        + frame["output_tokens"] * frame["pred_tpot_ms"]
    )

    return frame


def check_constraints(frame, args, available_ram_mib):
    reasons = []

    for _, row in frame.iterrows():
        broken = []

        if args.max_ttft_ms and row["pred_ttft_ms"] > args.max_ttft_ms:
            broken.append("ttft")
        if args.max_total_ms and row["pred_total_ms"] > args.max_total_ms:
            broken.append("total")
        if args.max_memory_mib and row["pred_peak_mib"] > args.max_memory_mib:
            broken.append("memory")
        # Only the CPU path is checked against free RAM: GPU buffers are not
        # fully visible in process RSS (Finding 11), so its RSS is a floor.
        if (
            row["device"] == "CPU"
            and row["pred_peak_mib"] + RAM_HEADROOM_MIB > available_ram_mib
        ):
            broken.append("oom-risk")

        reasons.append(",".join(broken))

    frame = frame.copy()
    frame["violations"] = reasons
    frame["feasible"] = frame["violations"] == ""
    return frame


def pareto_front(frame):
    """Candidates not beaten on both total time and memory."""
    front = []

    for index, row in frame.iterrows():
        dominated = (
            (frame["pred_total_ms"] <= row["pred_total_ms"])
            & (frame["pred_peak_mib"] <= row["pred_peak_mib"])
            & (
                (frame["pred_total_ms"] < row["pred_total_ms"])
                | (frame["pred_peak_mib"] < row["pred_peak_mib"])
            )
        ).any()

        if not dominated:
            front.append(index)

    return front


def choose(frame, objective, quality):
    feasible = frame[frame["feasible"]]

    if feasible.empty:
        return None

    if objective == "latency":
        return feasible.sort_values("pred_total_ms").iloc[0]
    if objective == "memory":
        return feasible.sort_values(["pred_peak_mib", "pred_total_ms"]).iloc[0]

    if not quality:
        raise SystemExit(
            "No quality scores yet (results/quality.json). "
            "Run the quality evaluation first, or use another objective."
        )

    # Lower bits-per-byte means a better model (quality_eval.py).
    score = {
        label: -values["bits_per_byte"]
        for label, values in quality.items()
    }
    feasible = feasible.assign(
        quality=feasible["label"].map(score).fillna(-math.inf)
    )
    return feasible.sort_values(
        ["quality", "pred_total_ms"],
        ascending=[False, True],
    ).iloc[0]


# ---------------------------------------------------------------------------
# Layer 3: run and correct


def clip_ratio(actual, raw):
    """measured / predicted, limited so one freak run cannot dominate."""
    return min(max(actual / raw, RATIO_MIN), RATIO_MAX)


def update_correction(
    state,
    candidate,
    raw_ttft_ms,
    raw_tpot_ms,
    actual_ttft_ms,
    actual_tpot_ms,
    raw_peak_mib=None,
    actual_peak_mib=None,
):
    """Move the candidate's factors toward actual / raw-twin-prediction.

    Time factors: an exponentially weighted average. One noisy run moves a
    factor by CORRECTION_ALPHA of its error, a persistent bias is learned
    within a few requests, and each run's ratio is clipped first.

    Peak-memory factor: same idea, but asymmetric. Measuring more memory than
    predicted is learned fast (PEAK_ALPHA_UNDER) because that is the OOM
    direction; measuring less is learned slowly (PEAK_ALPHA_OVER).
    """
    correction = state["corrections"].setdefault(
        candidate,
        {"ttft_factor": 1.0, "tpot_factor": 1.0, "observations": 0},
    )
    # State files written before the memory factor existed lack this key.
    correction.setdefault("peak_factor", 1.0)

    for factor_key, actual, raw in [
        ("ttft_factor", actual_ttft_ms, raw_ttft_ms),
        ("tpot_factor", actual_tpot_ms, raw_tpot_ms),
    ]:
        correction[factor_key] = (
            (1 - CORRECTION_ALPHA) * correction[factor_key]
            + CORRECTION_ALPHA * clip_ratio(actual, raw)
        )

    if raw_peak_mib and actual_peak_mib:
        ratio = clip_ratio(actual_peak_mib, raw_peak_mib)
        alpha = (
            PEAK_ALPHA_UNDER
            if ratio > correction["peak_factor"]
            else PEAK_ALPHA_OVER
        )
        correction["peak_factor"] = (
            (1 - alpha) * correction["peak_factor"] + alpha * ratio
        )

    correction["observations"] += 1
    return correction


def maybe_explore(frame, chosen, epsilon, rng):
    """With probability epsilon, swap the chosen candidate for another one.

    Without this, a candidate whose predictions became pessimistic (say, after
    a busy spell) is never picked again, so it is never re-measured and its
    correction never recovers. Only feasible candidates are considered, so
    exploring never breaks a constraint the twin can see.
    """
    if epsilon <= 0 or rng.random() >= epsilon:
        return chosen

    others = frame[frame["feasible"] & (frame["candidate"] != chosen["candidate"])]
    if others.empty:
        return chosen

    return others.iloc[int(rng.integers(len(others)))]


def run_and_learn(choice, args, models, state):
    settings = PLACEMENTS[choice["placement"]]

    configuration = {
        "label": choice["label"],
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "runs": 1,
        "prompt_mode": choice["prompt_mode"],
        "device": settings["device"],
    }
    if settings["device"] == "CPU":
        configuration["cores"] = settings["cores"]
        configuration["threads"] = settings["threads"]
    else:
        configuration["cache_dir"] = GPU_CACHE_DIR

    print(f"\nRunning {choice['candidate']} ...", flush=True)
    rows = run_configuration(models[choice["label"]]["path"], configuration)
    actual = rows[-1]

    if actual["status"] != "success":
        print(f"Run failed: {actual['status']} {actual.get('error', '')}")
        return actual

    correction = update_correction(
        state,
        choice["candidate"],
        raw_ttft_ms=choice["raw_ttft_ms"],
        raw_tpot_ms=choice["raw_tpot_ms"],
        actual_ttft_ms=actual["ttft_ms"],
        actual_tpot_ms=actual["tpot_ms_per_token"],
        raw_peak_mib=choice["raw_peak_mib"],
        actual_peak_mib=actual["sampled_peak_rss_mib"],
    )

    state["loaded_candidate"] = choice["candidate"]
    if settings["device"] == "GPU" and choice["label"] not in state["gpu_cache_warm"]:
        state["gpu_cache_warm"].append(choice["label"])

    print("\n            predicted    actual    error")
    for name, predicted, measured in [
        ("load ms", choice["pred_load_ms"], actual["pipeline_construction_ms"]),
        ("TTFT ms", choice["pred_ttft_ms"], actual["ttft_ms"]),
        ("TPOT ms", choice["pred_tpot_ms"], actual["tpot_ms_per_token"]),
        ("peak MiB", choice["pred_peak_mib"], actual["sampled_peak_rss_mib"]),
    ]:
        error = 100 * (predicted - measured) / measured
        print(f"  {name:<9}{predicted:>10.1f}{measured:>10.1f}{error:>+8.1f}%")

    print(
        f"\nCorrection for {choice['candidate']}: "
        f"TTFT x{correction['ttft_factor']:.3f}, "
        f"TPOT x{correction['tpot_factor']:.3f}, "
        f"memory x{correction['peak_factor']:.3f} "
        f"({correction['observations']} observations)"
    )

    return actual


# ---------------------------------------------------------------------------


def print_table(frame, chosen_id, front):
    print(
        f"\n{'candidate':<22}{'load':>8}{'TTFT':>9}{'TPOT':>8}"
        f"{'total':>9}{'peak MiB':>10}  status"
    )
    print("-" * 80)

    for index, row in frame.sort_values("pred_total_ms").iterrows():
        marks = []
        if row["candidate"] == chosen_id:
            marks.append("CHOSEN")
        if index in front:
            marks.append("pareto")
        if not row["feasible"]:
            marks.append(f"violates {row['violations']}")

        print(
            f"{row['candidate']:<22}"
            f"{row['pred_load_ms'] / 1000:>7.1f}s"
            f"{row['pred_ttft_ms'] / 1000:>8.2f}s"
            f"{row['pred_tpot_ms']:>6.0f}ms"
            f"{row['pred_total_ms'] / 1000:>8.1f}s"
            f"{row['pred_peak_mib']:>10.0f}  "
            + " ".join(marks)
        )


def main():
    args = get_arguments()

    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    labels = args.models or list(models)
    state = load_state()

    quality_path = Path("results/quality.json")
    quality = (
        json.loads(quality_path.read_text(encoding="utf-8"))
        if quality_path.exists()
        else {}
    )

    dataset = normalize(load_rows())
    twin = AnalyticalTwin().fit(dataset)

    memory = psutil.virtual_memory()
    machine = {
        "sys_cpu_busy_percent": psutil.cpu_percent(interval=0.5),
        "sys_available_ram_mib": memory.available / 2**20,
    }

    candidates = observed_candidates(dataset, labels)
    frame = build_candidate_frame(args, candidates, state, machine)
    frame = predict(frame, twin, dataset, state)
    frame = check_constraints(frame, args, machine["sys_available_ram_mib"])

    choice = choose(frame, args.objective, quality)
    if choice is not None:
        choice = maybe_explore(
            frame, choice, args.explore, np.random.default_rng()
        )
    front = pareto_front(frame[frame["feasible"]])

    print("=" * 80)
    print(
        f"Request: {args.prompt_tokens} prompt tokens -> "
        f"{args.output_tokens} output tokens"
        + (" (repeated prefix)" if args.repeat_prefix else "")
    )
    print(
        f"Objective: {args.objective}   Constraints: "
        f"TTFT<={args.max_ttft_ms or '-'} ms, "
        f"total<={args.max_total_ms or '-'} ms, "
        f"memory<={args.max_memory_mib or '-'} MiB"
    )
    print(
        f"Machine: {machine['sys_available_ram_mib']:.0f} MiB free RAM, "
        f"{machine['sys_cpu_busy_percent']:.0f}% CPU busy   "
        f"Loaded: {state['loaded_candidate'] or 'nothing'}"
    )
    print("=" * 80)

    print_table(frame, None if choice is None else choice["candidate"], front)

    if choice is None:
        print("\nNo candidate satisfies the constraints.")
        return

    print(f"\nChosen: {choice['candidate']}")

    actual = None
    if args.run:
        actual = run_and_learn(choice, args, models, state)
        save_state(state)

    DECISIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_JSONL.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "request": {
                        "prompt_tokens": args.prompt_tokens,
                        "output_tokens": args.output_tokens,
                        "repeat_prefix": args.repeat_prefix,
                    },
                    "objective": args.objective,
                    "constraints": {
                        "max_ttft_ms": args.max_ttft_ms,
                        "max_total_ms": args.max_total_ms,
                        "max_memory_mib": args.max_memory_mib,
                    },
                    "machine": machine,
                    "chosen": choice["candidate"],
                    "predicted": {
                        "load_ms": choice["pred_load_ms"],
                        "ttft_ms": choice["pred_ttft_ms"],
                        "tpot_ms": choice["pred_tpot_ms"],
                        "peak_mib": choice["pred_peak_mib"],
                    },
                    "actual": None
                    if actual is None or actual["status"] != "success"
                    else {
                        "load_ms": actual["pipeline_construction_ms"],
                        "ttft_ms": actual["ttft_ms"],
                        "tpot_ms": actual["tpot_ms_per_token"],
                        "peak_mib": actual["sampled_peak_rss_mib"],
                    },
                },
                default=float,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()