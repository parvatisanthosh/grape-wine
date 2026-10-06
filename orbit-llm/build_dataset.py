import json
from pathlib import Path

import pandas as pd


RESULTS = Path("results")
OUTPUT_CSV = RESULTS / "dataset.csv"

# All runs made through the C++ executor. The earlier Python-harness CSVs
# are left out so every row comes from the same measurement code.
SOURCES = [
    *sorted((RESULTS / "experiments").glob("*.jsonl")),
    RESULTS / "cpp_matrix_cold_raw.jsonl",
    RESULTS / "ab_python_vs_cpp_cpp.jsonl",
]

# Features a controller knows *before* running a request: the configuration,
# the workload shape, and the machine state just before the run.
CONFIG_FEATURES = [
    "precision",
    "device",
    "backend",
    "cores",
    "threads",
    "kv_precision",
    "prefix_caching",
    "cache_size_gb",
    "prompt_mode",
]

WORKLOAD_FEATURES = [
    "prompt_tokens",
    "output_tokens",
    "cache_hit",
]

MACHINE_FEATURES = [
    "sys_cpu_busy_percent",
    "sys_available_ram_mib",
]

TARGETS = [
    "ttft_ms",
    "tpot_ms_per_token",
    "sampled_peak_rss_mib",
]


def load_rows():
    rows = []

    for source in SOURCES:
        if not source.exists():
            continue

        with source.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                row = json.loads(line)
                row.setdefault(
                    "experiment",
                    source.stem,
                )
                rows.append(row)

    return pd.DataFrame(rows)


def normalize(frame):
    frame = frame[frame["status"] == "success"].copy()

    frame["precision"] = frame["label"]
    frame["prompt_tokens"] = frame["requested_input_tokens"].astype(int)
    frame["output_tokens"] = frame["requested_output_tokens"].astype(int)

    # u8 is the verified CPU default (Core.get_property, FINDINGS Phase 4).
    frame["kv_precision"] = frame["kv_precision"].replace(
        {"default": "u8"}
    )

    # SDPA has no prefix cache or paged-cache settings.
    frame["prefix_caching"] = frame["prefix_caching"].fillna(False).astype(bool)
    frame["cache_size_gb"] = frame["cache_size_gb"].fillna(0).astype(int)
    frame["threads"] = frame["threads"].fillna(0).astype(int)

    # A repeated prompt with prefix caching on skips almost all prefill
    # (FINDINGS Finding 8). The workload profiler can know this in advance.
    frame["cache_hit"] = (
        (frame["prompt_mode"] == "repeat")
        & frame["prefix_caching"]
    ).astype(int)

    # One identifier per distinct configuration + workload, ignoring machine
    # state. Cross-validation holds out whole configurations, so the model is
    # always tested on combinations it has not seen.
    frame["config_key"] = (
        frame[CONFIG_FEATURES + ["prompt_tokens", "output_tokens"]]
        .astype(str)
        .agg("|".join, axis=1)
    )

    columns = (
        ["experiment", "config_key"]
        + CONFIG_FEATURES
        + WORKLOAD_FEATURES
        + MACHINE_FEATURES
        + TARGETS
    )

    return frame[columns].reset_index(drop=True)


def main():
    frame = normalize(load_rows())
    frame.to_csv(OUTPUT_CSV, index=False)

    print(f"Rows: {len(frame)}")
    print(f"Distinct configurations: {frame['config_key'].nunique()}")
    print("\nRows per experiment:")
    print(frame["experiment"].value_counts().to_string())
    print(f"\nDataset saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
