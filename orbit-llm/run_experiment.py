import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

from orbit_exec import run_configuration


ATTEMPTED_STATUSES = {
    "success",
    "failed",
    "load_failed",
    "warmup_failed",
    "crashed",
}


def get_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run an ORBIT-LLM experiment plan through the "
            "C++ orbit_run executor."
        )
    )

    parser.add_argument(
        "plan",
        type=Path,
        help="Experiment plan JSON file (see plans/).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the schedule without running anything.",
    )

    parser.add_argument(
        "--allow-battery",
        action="store_true",
        help="Run even when the laptop is not plugged in.",
    )

    return parser.parse_args()


def load_plan(plan_path):
    with plan_path.open("r", encoding="utf-8") as file:
        plan = json.load(file)

    for key in ["name", "models", "grid"]:
        if key not in plan:
            raise KeyError(f"Plan is missing '{key}'")

    if "label" not in plan["grid"]:
        raise KeyError(
            "Plan grid must contain 'label' "
            "(keys of 'models')"
        )

    unknown_labels = (
        set(plan["grid"]["label"]) - set(plan["models"])
    )

    if unknown_labels:
        raise KeyError(
            f"Labels without a model path: "
            f"{sorted(unknown_labels)}"
        )

    # Named variants are bundles of settings that do not form a clean grid
    # (e.g. "pcore", "ecore", "gpu"). They become one more grid dimension.
    if "variants" in plan:
        plan["grid"]["variant"] = list(plan["variants"])

    plan.setdefault("variants", {})
    plan.setdefault("base", {})
    plan.setdefault("rounds", 1)
    plan.setdefault("seed", 0)
    plan.setdefault("cooldown_seconds", 3)

    return plan


def expand_grid(grid):
    keys = sorted(grid)

    return [
        dict(zip(keys, values))
        for values in itertools.product(
            *(grid[key] for key in keys)
        )
    ]


def make_config_id(grid_values):
    return "|".join(
        f"{key}={grid_values[key]}"
        for key in sorted(grid_values)
    )


def build_schedule(plan):
    """Return (round, grid_values) pairs in run order.

    Each round runs every configuration once, in a different random
    order, so slow drift in machine state (thermals, background load)
    spreads across all configurations instead of biasing whichever
    ran last.
    """
    configurations = expand_grid(plan["grid"])
    generator = random.Random(plan["seed"])
    schedule = []

    for round_number in range(1, plan["rounds"] + 1):
        order = list(configurations)
        generator.shuffle(order)

        schedule.extend(
            (round_number, grid_values)
            for grid_values in order
        )

    return schedule


def load_completed(jsonl_path):
    completed = set()

    if not jsonl_path.exists():
        return completed

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("status") in ATTEMPTED_STATUSES:
                completed.add(
                    (row["config_id"], row["round"])
                )

    return completed


def prevent_sleep():
    """Keep Windows from idle-sleeping while this process runs.

    SetThreadExecutionState is a per-process request (what media players use);
    it is released automatically when the process exits and changes no
    power settings. A closed lid or a manual Sleep still sleeps.
    """
    if sys.platform != "win32":
        return

    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )


def require_ac_power(allow_battery=False):
    """Refuse to benchmark on battery: it runs ~10x slower (Finding 20)."""
    import psutil

    battery = psutil.sensors_battery()

    if battery is None or battery.power_plugged:
        return

    message = (
        f"The laptop is on battery ({battery.percent:.0f}%). "
        f"Measurements on battery are not comparable with AC runs."
    )

    if allow_battery:
        print(f"Warning: {message}")
        return

    raise SystemExit(f"{message} Plug in, or pass --allow-battery.")


def main():
    args = get_arguments()
    require_ac_power(args.allow_battery)
    plan = load_plan(args.plan)

    output_path = Path(
        plan.get(
            "output",
            f"results/experiments/{plan['name']}.jsonl",
        )
    )

    schedule = build_schedule(plan)
    completed = load_completed(output_path)

    pending = [
        (round_number, grid_values)
        for round_number, grid_values in schedule
        if (make_config_id(grid_values), round_number)
        not in completed
    ]

    runs_per_config = plan["base"].get("runs", 3)

    print("=" * 70)
    print(f"EXPERIMENT: {plan['name']}")
    print(f"Plan: {args.plan}")
    print(f"Output: {output_path}")
    print(
        f"Configurations: {len(expand_grid(plan['grid']))} "
        f"x {plan['rounds']} rounds = {len(schedule)} invocations"
    )
    print(f"Already completed: {len(schedule) - len(pending)}")
    print(f"Pending: {len(pending)}")
    print(
        f"Measured rows expected: "
        f"{len(schedule) * runs_per_config}"
    )
    print("=" * 70)

    if args.dry_run:
        for round_number, grid_values in pending:
            print(
                f"round {round_number}: "
                f"{make_config_id(grid_values)}"
            )
        return

    prevent_sleep()

    failed = 0
    experiment_start = time.perf_counter()

    for index, (round_number, grid_values) in enumerate(
        pending,
        start=1,
    ):
        config_id = make_config_id(grid_values)

        print("\n" + "#" * 70)
        print(
            f"[{index}/{len(pending)}] "
            f"round {round_number}: {config_id}"
        )
        print("#" * 70, flush=True)

        configuration = {
            **plan["base"],
            **{
                key: value
                for key, value in grid_values.items()
                if key != "variant"
            },
            **plan["variants"].get(
                grid_values.get("variant"),
                {},
            ),
        }

        rows = run_configuration(
            plan["models"][grid_values["label"]],
            configuration,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open("a", encoding="utf-8") as file:
            for row in rows:
                row["experiment"] = plan["name"]
                row["config_id"] = config_id
                row["round"] = round_number
                file.write(json.dumps(row) + "\n")

        if any(row["status"] != "success" for row in rows):
            failed += 1

        if index < len(pending):
            time.sleep(plan["cooldown_seconds"])

    elapsed_minutes = (
        time.perf_counter() - experiment_start
    ) / 60

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Invocations with failures: {failed}")
    print(f"Elapsed time: {elapsed_minutes:.2f} minutes")
    print(f"Results: {output_path.resolve()}")
    print(
        f"Analyze: python analyze_experiment.py "
        f"{output_path}"
    )
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
