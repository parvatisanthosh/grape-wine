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

Systems: baseline (OpenVINO defaults), static (one fixed configuration),
orbit (predictions only) and orbit_robust (ORBIT that also checks the
plausibly-bad case of every prediction, with a fallback order when no
candidate is safe).

Systems take turns request by request (an interleaved, paired design):
for each request every system runs it back-to-back in a shuffled order,
so all systems face the same machine conditions at the same time.
Running one system's whole trace after another let laptop drift (heat,
background activity) differ between systems and bias the comparison.

Two phases are run for every system. First a calm phase (the scenarios
below on an idle machine). Then a busy phase: the latency-sensitive
scenarios are repeated while background processes keep several CPU cores
busy, like a browser or a video call would. The busy phase is where a
fixed configuration cannot adapt and ORBIT's online correction should
show its value. Every system faces the same load.

Executors (--executor)
- fresh:  every request starts a new orbit_run.exe, so every request really
  loads the model; residency is simulated by charging the load only when a
  system switches configuration. This was the only mode before the worker.
- worker: every system owns one persistent orbit_worker.exe that keeps models
  loaded (ORBIT up to --max-resident of them, least recently used evicted
  first). Load and unload times are charged only when they really happen,
  a prefix-cache hit happens only if the same prompt really was prefilled
  on a pipeline that is still loaded, and peak memory is the whole worker
  process, including every model it keeps loaded. A request with a memory
  limit therefore evicts the system's other models first and pays for it.
"""

import argparse
import atexit
import json
import random
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd
import psutil

import orbit
from build_dataset import load_rows, normalize
from orbit_exec import run_configuration
from orbit_worker_client import OrbitWorker
from run_experiment import prevent_sleep
from twin_v0 import AnalyticalTwin


RESULTS_JSONL = Path("results/evaluation_runs.jsonl")
SUMMARY_CSV = Path("results/evaluation_summary.csv")
WORKER_RESULTS_JSONL = Path("results/evaluation_runs_worker.jsonl")
WORKER_SUMMARY_CSV = Path("results/evaluation_summary_worker.csv")

DEFAULT_SYSTEMS = {
    "fresh": ["baseline", "static", "orbit"],
    # Every system holds its own models in RAM at the same time, so the
    # default leaves out the baseline (INT8 on CPU, ~2 GB), whose result does
    # not depend on residency anyway.
    "worker": ["static", "orbit"],
}

# How many models ORBIT keeps loaded in worker mode. Measured switch costs
# (results/experiments/residency.jsonl): loading Q3B-INT4 takes 8-10 s, more
# than two whole 1024-token requests, so one-at-a-time would pay that on every
# change between the small and the large model.
MAX_RESIDENT = 2

# Prompt variants built into orbit_run.cpp. Each request gets
# "<variant>#<request number>", a prompt no earlier request used, so it can
# never hit the prefix cache by accident. A repeated-prefix scenario uses one
# fixed id: the same conversation every time.
PROMPT_VARIANTS = ["run_1", "run_2", "run_3", "run_4", "run_5"]
CONVERSATION_PROMPT_ID = "run_1#conversation"

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

REPEATS = 3

# Scenarios repeated in the busy phase: the ones with a time limit.
BUSY_SCENARIO_NAMES = {"short_chat", "long_document", "follow_up"}
BUSY_REPEATS = 3
LOAD_WORKERS = 6
REST_SECONDS = 0
SEED = 0


class BackgroundLoad:
    """Keep some CPU cores busy, the way a browser or a video call would.

    Each worker is a separate process running an endless empty loop, which
    uses one core fully. Only CPU is loaded. It slows inference down (TTFT,
    TPOT) but does not change how much memory the model itself needs.
    """

    def __init__(self, workers, settle_seconds=3):
        self.workers = workers
        self.settle_seconds = settle_seconds
        self.processes = []

    @property
    def active(self):
        return bool(self.processes)

    def start(self):
        if self.active or self.workers <= 0:
            return

        for _ in range(self.workers):
            self.processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", "while True: pass"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

        # Let CPU usage settle so the controller's own reading sees the load.
        time.sleep(self.settle_seconds)

    def stop(self):
        for process in self.processes:
            process.terminate()

        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

        self.processes = []


def get_arguments():
    parser = argparse.ArgumentParser(description="Run the Phase 8 evaluation.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        choices=["baseline", "static", "orbit", "orbit_robust"],
        help="Default: baseline static orbit (fresh), static orbit (worker).",
    )
    parser.add_argument(
        "--executor",
        choices=["fresh", "worker"],
        default="fresh",
        help="fresh: a new process per request (residency simulated); "
        "worker: persistent workers (residency measured).",
    )
    parser.add_argument(
        "--max-resident",
        type=int,
        default=MAX_RESIDENT,
        help="worker mode: how many models each ORBIT system keeps loaded.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=orbit.MARGIN_COVERAGE,
        help="orbit_robust: how many past runs the safety margin must cover "
        "(0.9 = the margin covers 9 held-out runs in 10).",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--busy-repeats",
        type=int,
        default=BUSY_REPEATS,
        help="Repeats of the time-limited scenarios under background load "
        "(0 skips the busy phase).",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        default=LOAD_WORKERS,
        help="Background processes, each keeping one CPU core busy.",
    )
    parser.add_argument(
        "--rest",
        type=int,
        default=REST_SECONDS,
        help="Seconds to rest after each request (all systems), to let heat "
        "settle. With interleaving every system gets the same rest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed for the per-request shuffle of system order (reproducible).",
    )
    args = parser.parse_args()
    if args.systems is None:
        args.systems = DEFAULT_SYSTEMS[args.executor]
    if args.max_resident < 1:
        parser.error("--max-resident must be at least 1")
    return args


def build_trace(repeats=REPEATS, busy_repeats=BUSY_REPEATS):
    """The request list: a calm phase, then a busy phase.

    Busy requests reuse the time-limited scenarios under the name
    "<scenario>_busy", so results for calm and busy runs stay separate.
    """
    trace = []

    def add(scenarios, count, busy):
        for repeat in range(count):
            for name, prompt, output, repeat_prefix, objective, limits in scenarios:
                number = len(trace) + 1
                if repeat_prefix:
                    prompt_id = CONVERSATION_PROMPT_ID
                else:
                    variant = PROMPT_VARIANTS[number % len(PROMPT_VARIANTS)]
                    prompt_id = f"{variant}#{number}"
                trace.append(
                    Namespace(
                        scenario=f"{name}_busy" if busy else name,
                        phase="busy" if busy else "calm",
                        busy=busy,
                        repeat=repeat + 1,
                        prompt_tokens=prompt,
                        output_tokens=output,
                        repeat_prefix=repeat_prefix,
                        prompt_id=prompt_id,
                        # What a pipeline remembers: the worker caches the
                        # tokenized prompt per (id, length).
                        prompt_key=f"{prompt_id}:{prompt}",
                        objective=objective,
                        max_ttft_ms=limits.get("max_ttft_ms"),
                        max_total_ms=limits.get("max_total_ms"),
                        max_memory_mib=limits.get("max_memory_mib"),
                    )
                )

    add(SCENARIOS, repeats, busy=False)
    add(
        [row for row in SCENARIOS if row[0] in BUSY_SCENARIO_NAMES],
        busy_repeats,
        busy=True,
    )

    return trace


def summarize_systems(frame):
    """Per-system results over everything each system was asked to do.

    A refused request counts as a violation, but it has no latency, so the
    means below cover only the requests that system served. Those means cannot
    be compared across systems that served different requests: use
    summarize_matched for that.
    """
    frame = frame.assign(
        served=frame["status"] == "success",
        refused=frame["status"] == "no_feasible_candidate",
    )
    summary = frame.groupby("system").agg(
        requests=("index", "count"),
        served=("served", "sum"),
        refused=("refused", "sum"),
        sla_violation_rate=("violated", "mean"),
        mean_total_s=("total_ms", lambda values: values.mean() / 1000),
        mean_ttft_s=("ttft_ms", lambda values: values.mean() / 1000),
        mean_peak_mib=("peak_mib", "mean"),
        mean_bits_per_byte=("bits_per_byte", "mean"),
        switches=("switched", "sum"),
    )
    summary["sla_violation_rate"] *= 100
    return summary


def summarize_matched(frame):
    """Compare systems only on the requests every one of them served.

    Returns None with fewer than two systems or no request in common.
    """
    systems = frame["system"].nunique()
    served = frame[frame["status"] == "success"]
    counts = served.groupby("index")["system"].nunique()
    common = counts[counts == systems].index
    if systems < 2 or len(common) == 0:
        return None

    matched = served[served["index"].isin(common)].groupby("system").agg(
        requests=("index", "count"),
        violations=("violated", "sum"),
        mean_total_s=("total_ms", lambda values: values.mean() / 1000),
        mean_ttft_s=("ttft_ms", lambda values: values.mean() / 1000),
        mean_peak_mib=("peak_mib", "mean"),
        mean_bits_per_byte=("bits_per_byte", "mean"),
    )
    matched["violations"] = matched["violations"].astype(int)
    return matched


def paired_comparison(frame, first, second):
    """Compare two systems on the same requests, run at the same time.

    Returns (only_first_kept, only_second_kept, both_kept, both_broke): how
    many requests one system kept its promise on while the other broke it.
    Requests both kept or both broke say nothing about which is better, so
    this cancels most of the machine's run-to-run noise.
    """
    pivot = frame.pivot_table(
        index="index", columns="system", values="violated", aggfunc="first"
    )
    if first not in pivot or second not in pivot:
        return None

    pair = pivot[[first, second]].dropna().astype(bool)
    return (
        int((~pair[first] & pair[second]).sum()),
        int((pair[first] & ~pair[second]).sum()),
        int((~pair[first] & ~pair[second]).sum()),
        int((pair[first] & pair[second]).sum()),
    )


def interleaved_schedule(trace, systems, seed=SEED):
    """The run order: every request is run by all systems back-to-back.

    Returns (index, request, system) tuples. The systems' order is shuffled
    per request with a seeded generator, so no system always runs right after
    the same other one (carry-over heat or GPU state spreads evenly) and the
    same seed always gives the same plan.
    """
    rng = random.Random(seed)
    plan = []

    for index, request in enumerate(trace, start=1):
        order = list(systems)  # a copy: shuffle() changes the list in place
        rng.shuffle(order)
        for system in order:
            plan.append((index, request, system))

    return plan


def load_options(label, placement):
    """The worker load flags for one candidate configuration."""
    settings = orbit.PLACEMENTS[placement]
    options = {"label": label, "device": settings["device"]}
    if settings["device"] == "CPU":
        options["cores"] = settings["cores"]
        options["threads"] = settings["threads"]
    else:
        options["cache_dir"] = orbit.GPU_CACHE_DIR
    return options


class ResidentService:
    """One system's long-lived inference service, backed by its own worker.

    Keeps up to max_resident configurations loaded, least recently used first
    in self.resident. Every load, unload and generation is real, and run()
    reports what it cost. self.cached_prompts records which prompts each
    loaded pipeline has prefilled: the prefix cache lives inside a pipeline,
    so unloading a model empties its list.

    Each system has its own worker process, so the memory one system's
    requests are judged on contains only that system's models.
    """

    def __init__(self, models, max_resident=1, worker_factory=OrbitWorker):
        self.models = models
        self.max_resident = max_resident
        self.worker = worker_factory()
        self.resident = []
        self.cached_prompts = {}

    def _unload(self, slot):
        reply = self.worker.unload(slot)
        self.resident.remove(slot)
        self.cached_prompts.pop(slot, None)
        if reply["status"] != "success":
            print(f"  !! unload {slot}: {reply.get('error', '')}", flush=True)
        return reply.get("unload_ms") or 0.0

    def run(self, label, placement, request):
        """Serve one request. Returns the worker's generate row plus:

        pipeline_construction_ms  load time if this request had to load, else None
        unload_ms_charged         time spent evicting other models first
        evicted                   which configurations were evicted
        cache_hit                 whether this exact prompt was already prefilled
        """
        slot = orbit.candidate_id(label, placement)
        unload_ms = 0.0
        evicted = []

        # The request is judged on the memory of the whole process, so with
        # a memory limit it runs alone: everything else is unloaded first.
        if request.max_memory_mib:
            for other in [name for name in self.resident if name != slot]:
                unload_ms += self._unload(other)
                evicted.append(other)

        load_reply = None
        if slot in self.resident:
            self.resident.remove(slot)  # re-added at the end: most recently used
        else:
            while len(self.resident) >= self.max_resident:
                oldest = self.resident[0]
                unload_ms += self._unload(oldest)
                evicted.append(oldest)
            load_reply = self.worker.load(
                slot, self.models[label]["path"], **load_options(label, placement)
            )
            if load_reply["status"] != "success":
                return {
                    **load_reply,
                    "label": label,
                    "pipeline_construction_ms": None,
                    "unload_ms_charged": unload_ms,
                    "evicted": evicted,
                    "cache_hit": False,
                }
        self.resident.append(slot)

        prompts = self.cached_prompts.setdefault(slot, [])
        cache_hit = request.prompt_key in prompts

        row = dict(self.worker.generate(
            slot, request.prompt_tokens, request.output_tokens, prompt_id=request.prompt_id
        ))
        if row["status"] == "success" and not cache_hit:
            prompts.append(request.prompt_key)

        row["label"] = label
        row["pipeline_construction_ms"] = (
            load_reply["pipeline_construction_ms"] if load_reply else None
        )
        if load_reply and row.get("sampled_peak_rss_mib") is not None:
            # Peak memory is sampled during generation only; loading can
            # briefly need more, so take the larger of the two.
            row["sampled_peak_rss_mib"] = max(
                row["sampled_peak_rss_mib"], load_reply.get("rss_after_mib") or 0.0
            )
        row["unload_ms_charged"] = unload_ms
        row["evicted"] = evicted
        row["cache_hit"] = cache_hit
        return row

    def share_state(self, state):
        """Tell ORBIT what is really loaded before it decides."""
        state["resident"] = list(self.resident)
        state["cached_prompts"] = {
            slot: list(prompts) for slot, prompts in self.cached_prompts.items()
        }

    def close(self):
        self.worker.close()


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


def orbit_decide(request, state, dataset, twin, models, quality, margins=None):
    """One ORBIT decision. With margins it is the robust variant: candidates
    must stay inside the limits even in the plausibly-bad case, and it falls
    back to point predictions, then to the smallest miss, rather than giving
    up. Without margins it is the original selector (it may find nothing)."""
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
        margins,
    )

    row, mode = orbit.select(
        frame, request.objective, quality, fallback=margins is not None
    )
    if row is not None:
        row["mode"] = mode
    return row, frame


def charged_load_ms(row, switched, executor):
    """The load-related time a request pays before generating.

    fresh:  the measured load, but only when the system switched (simulated
            residency: a fresh process always loads, a service would not).
    worker: whatever really happened: a load if one was needed, plus the
            time spent unloading other models to make room.
    """
    if executor == "fresh":
        return row["pipeline_construction_ms"] if switched else 0.0
    return (row.get("pipeline_construction_ms") or 0.0) + row.get("unload_ms_charged", 0.0)


def judge(request, row, load_ms, quality):
    if row["status"] != "success":
        return {
            "status": row["status"],
            "violated": True,
            "violations": "failed",
        }

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
    trace = build_trace(args.repeats, args.busy_repeats)

    margins = None
    if "orbit_robust" in args.systems:
        margins = orbit.calibrate_margins(dataset, args.coverage)
        print(f"\nSafety margins at coverage {args.coverage} (worst-case multiplier):")
        for metric, table in margins.items():
            shown = ", ".join(
                f"{'/'.join(map(str, key))} x{value:.2f}"
                for key, value in sorted(table.items(), key=str)
                if len(key) == 1
            )
            print(f"  {metric:<9} {shown}")

    load = BackgroundLoad(args.load_workers)
    atexit.register(load.stop)  # never leave burners running after a crash

    worker_mode = args.executor == "worker"
    results_path = WORKER_RESULTS_JSONL if worker_mode else RESULTS_JSONL
    summary_path = WORKER_SUMMARY_CSV if worker_mode else SUMMARY_CSV

    services = {}
    if worker_mode:
        print(f"\nStarting one worker per system: {', '.join(args.systems)}", flush=True)
        for system in args.systems:
            keep = args.max_resident if system.startswith("orbit") else 1
            services[system] = ResidentService(models, max_resident=keep)
            atexit.register(services[system].close)

    records = []

    # What each system has loaded, and ORBIT's learned state, kept per system
    # for the whole run. Each ORBIT variant learns only from its own runs.
    loaded = {system: None for system in args.systems}
    states = {}
    for system in args.systems:
        state = {"loaded_candidate": None, "gpu_cache_warm": [], "corrections": {}}
        # The compiled-model cache on disk is shared, so start from what is
        # really cached rather than from nothing.
        orbit.sync_gpu_cache_state(state, dataset)
        states[system] = state

    plan = interleaved_schedule(trace, args.systems, args.seed)
    previous_index = None

    for index, request, system in plan:
        if index != previous_index:
            if previous_index is not None and args.rest > 0:
                time.sleep(args.rest)
            # The same load applies to every system's run of this request.
            if request.busy and not load.active:
                print("\nStarting background load ...", flush=True)
                load.start()
            elif not request.busy and load.active:
                load.stop()
            previous_index = index

        state = states[system]
        if worker_mode:
            services[system].share_state(state)

        if system == "baseline":
            label, placement = BASELINE
            choice = None
        elif system == "static":
            label, placement = STATIC
            choice = None
        else:
            choice, decision_frame = orbit_decide(
                request,
                state,
                dataset,
                twin,
                models,
                quality,
                margins if system == "orbit_robust" else None,
            )

            if choice is None:
                print(
                    f"\n[{system} {index}/{len(trace)}] {request.scenario}: "
                    "NO CANDIDATE FITS the request's limits. Closest:",
                    flush=True,
                )
                print(orbit.nearest_misses(decision_frame).round(0).to_string(index=False))
                print(
                    "  free RAM when deciding: "
                    f"{decision_frame['sys_available_ram_mib'].iloc[0]:.0f} MiB",
                    flush=True,
                )
                refusal = {
                    "system": system,
                    "index": index,
                    "scenario": request.scenario,
                    "phase": request.phase,
                    "repeat": request.repeat,
                    "chosen": None,
                    "status": "no_feasible_candidate",
                    "violated": True,
                    "violations": "no-candidate",
                }
                records.append(refusal)
                # A refusal is a result: keep it in the raw log as well.
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with results_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(refusal) + "\n")
                continue  # next system's run (or the next request)

            label, placement = choice["label"], choice["placement"]

        identifier = orbit.candidate_id(label, placement)
        if worker_mode:
            switched = identifier not in services[system].resident
        else:
            switched = loaded[system] != identifier

        print(
            f"\n[{system} {index}/{len(trace)}] {request.scenario} "
            f"-> {identifier}{' (switch)' if switched else ''}",
            flush=True,
        )

        if worker_mode:
            service = services[system]
            if system.startswith("orbit"):
                row = orbit.run_and_learn(
                    choice,
                    request,
                    models,
                    state,
                    executor=lambda chosen, args_: service.run(
                        chosen["label"], chosen["placement"], args_
                    ),
                )
            else:
                row = service.run(label, placement, request)
        elif system.startswith("orbit"):
            row = orbit.run_and_learn(choice, request, models, state)
        else:
            row = execute(label, placement, request, models, loaded[system])

        loaded[system] = identifier if row["status"] == "success" else None

        record = {
            "system": system,
            "index": index,
            "scenario": request.scenario,
            "phase": request.phase,
            "repeat": request.repeat,
            "chosen": identifier,
            "switched": switched,
            "executor": args.executor,
            **judge(request, row, charged_load_ms(row, switched, args.executor), quality),
        }
        if worker_mode:
            record["prompt_id"] = request.prompt_id
            record["cache_hit"] = bool(row.get("cache_hit"))
            record["generation_index"] = row.get("generation_index")
            record["evicted"] = ",".join(row.get("evicted", []))
            record["resident_after"] = ",".join(services[system].resident)
        if choice is not None:
            record["predicted_total_ms"] = float(choice["pred_total_ms"])
            record["mode"] = choice.get("mode")

        records.append(record)

        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, default=float) + "\n")

    load.stop()
    for service in services.values():
        service.close()

    frame = pd.DataFrame(records)
    summary = summarize_systems(frame)
    summary.to_csv(summary_path)

    print("\n" + summary.round(2).to_string())

    matched = summarize_matched(frame)
    if matched is not None:
        print("\nLike for like (only requests every system served):")
        print(matched.round(2).to_string())

    by_scenario = frame.pivot_table(
        index="scenario",
        columns="system",
        values="violated",
        aggfunc="mean",
    )
    print("\nSLA violation rate by scenario:")
    print((100 * by_scenario).round(0).to_string())

    by_phase = frame.pivot_table(
        index="phase",
        columns="system",
        values="violated",
        aggfunc="mean",
    )
    print("\nSLA violation rate by phase (calm vs busy machine):")
    print((100 * by_phase).round(0).to_string())

    if "mode" in frame:
        modes = frame.dropna(subset=["mode"]).pivot_table(
            index="mode", columns="system", values="index", aggfunc="count"
        ).fillna(0).astype(int)
        print("\nHow ORBIT's choices were made (robust = safe even in the bad case,")
        print("point = predictions only, point-only = fell back, best-effort = nothing fit):")
        print(modes.to_string())

    pairs = [
        (a, b)
        for a in ["orbit", "orbit_robust"]
        for b in ["static", "baseline"]
        if a in args.systems and b in args.systems
    ]
    if pairs:
        print("\nPaired comparison (same request, same time):")
        print(f"  {'':<26}{'only A kept':>12}{'only B kept':>12}{'both kept':>11}{'both broke':>11}")
        for a, b in pairs:
            result = paired_comparison(frame, a, b)
            if result:
                print(f"  A={a:<12} B={b:<9}" + "".join(f"{n:>12}" for n in result[:2]) + "".join(f"{n:>11}" for n in result[2:]))

    if worker_mode:
        print("\nResidency (measured):")
        served = frame.dropna(subset=["chosen"])
        served = served[served["status"] == "success"]
        residency = served.groupby("system").agg(
            loads=("switched", "sum"),
            load_s_charged=("load_ms_charged", lambda values: values.sum() / 1000),
            cache_hits=("cache_hit", "sum"),
            evicting_requests=("evicted", lambda values: int((values != "").sum())),
        )
        print(residency.round(1).to_string())

    print(f"\nRuns: {results_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()