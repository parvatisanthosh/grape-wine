"""Replay logged runs through the online correction (Layer 3), no hardware.

Takes the real measured runs of one configuration and plays them back as if
they were requests arriving one after another. Part-way through, a
"disturbance" is injected by scaling the measured values (a busy machine makes
TTFT slower and memory larger). At every request we record how far off the
twin's raw prediction and the corrected prediction are, then plot both.

What this shows: how many requests the correction needs to recover after the
machine changes. What it does NOT show: the twin's accuracy. The twin is fit
on these same rows, so the error before the disturbance is optimistic.

    python replay_correction.py
    python replay_correction.py --ttft-scale 2.0 --requests 40
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import orbit  # noqa: E402
from twin_v0 import AnalyticalTwin  # noqa: E402

DATASET = Path("results/dataset.csv")
OUTPUT_CSV = Path("results/replay_correction.csv")
OUTPUT_PNG = Path("results/replay_correction.png")

# The most-measured configuration in the dataset.
DEFAULT_CONFIG = "INT4|CPU|pa|any|0|u8|True|0|cold|1024|64"


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config-key", default=DEFAULT_CONFIG)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--disturb-at", type=int, default=16, help="1-based request")
    parser.add_argument("--ttft-scale", type=float, default=1.5)
    parser.add_argument("--peak-scale", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def percent_error(predicted, actual):
    return 100 * abs(predicted - actual) / actual


def main():
    args = get_arguments()

    dataset = pd.read_csv(DATASET)
    rows = dataset[dataset["config_key"] == args.config_key].reset_index(drop=True)
    if rows.empty:
        raise SystemExit(f"No rows for config_key {args.config_key!r}")

    twin = AnalyticalTwin().fit(dataset)
    raw = twin.predict(rows)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(np.resize(np.arange(len(rows)), args.requests))

    state = {"corrections": {}}
    candidate = "replay"
    records = []

    for request, index in enumerate(order, start=1):
        disturbed = request >= args.disturb_at
        ttft_scale = args.ttft_scale if disturbed else 1.0
        peak_scale = args.peak_scale if disturbed else 1.0

        actual_ttft = rows.loc[index, "ttft_ms"] * ttft_scale
        actual_tpot = rows.loc[index, "tpot_ms_per_token"]
        actual_peak = rows.loc[index, "sampled_peak_rss_mib"] * peak_scale

        raw_ttft = float(raw["ttft_ms"][index])
        raw_tpot = float(raw["tpot_ms_per_token"][index])
        raw_peak = float(raw["sampled_peak_rss_mib"][index])

        # The prediction the controller would have made BEFORE seeing this run.
        factors = state["corrections"].get(candidate, {})
        corrected_ttft = raw_ttft * factors.get("ttft_factor", 1.0)
        corrected_peak = raw_peak * factors.get("peak_factor", 1.0)

        records.append(
            {
                "request": request,
                "disturbed": disturbed,
                "raw_ttft_err_pct": percent_error(raw_ttft, actual_ttft),
                "corrected_ttft_err_pct": percent_error(corrected_ttft, actual_ttft),
                "raw_peak_err_pct": percent_error(raw_peak, actual_peak),
                "corrected_peak_err_pct": percent_error(corrected_peak, actual_peak),
                "ttft_factor": factors.get("ttft_factor", 1.0),
                "peak_factor": factors.get("peak_factor", 1.0),
                # Memory under-prediction is the dangerous direction.
                "peak_underpredicted_raw": raw_peak < actual_peak,
                "peak_underpredicted_corrected": corrected_peak < actual_peak,
            }
        )

        orbit.update_correction(
            state,
            candidate,
            raw_ttft_ms=raw_ttft,
            raw_tpot_ms=raw_tpot,
            actual_ttft_ms=actual_ttft,
            actual_tpot_ms=actual_tpot,
            raw_peak_mib=raw_peak,
            actual_peak_mib=actual_peak,
        )

    result = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    summarize(result, args)
    plot(result, args)


def summarize(result, args):
    after = result[result["disturbed"]]
    windows = {
        "before disturbance": result[~result["disturbed"]],
        f"first 3 after (req {args.disturb_at}-{args.disturb_at + 2})": after.head(3),
        "last 5 requests": result.tail(5),
    }

    print(f"\nConfig: {args.config_key}")
    print(
        f"Disturbance at request {args.disturb_at}: "
        f"TTFT x{args.ttft_scale}, memory x{args.peak_scale}\n"
    )
    print(f"{'window':<34}{'TTFT err raw':>14}{'corrected':>11}{'mem err raw':>13}{'corrected':>11}")
    for name, window in windows.items():
        print(
            f"{name:<34}"
            f"{window['raw_ttft_err_pct'].mean():>13.1f}%"
            f"{window['corrected_ttft_err_pct'].mean():>10.1f}%"
            f"{window['raw_peak_err_pct'].mean():>12.1f}%"
            f"{window['corrected_peak_err_pct'].mean():>10.1f}%"
        )

    print(
        "\nMemory under-predictions after the disturbance: "
        f"raw {int(after['peak_underpredicted_raw'].sum())}/{len(after)}, "
        f"corrected {int(after['peak_underpredicted_corrected'].sum())}/{len(after)}"
    )


def plot(result, args):
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for axis, raw_col, corr_col, title in [
        (top, "raw_ttft_err_pct", "corrected_ttft_err_pct", "TTFT prediction error"),
        (bottom, "raw_peak_err_pct", "corrected_peak_err_pct", "Peak memory prediction error"),
    ]:
        axis.plot(result["request"], result[raw_col], marker="o", label="twin only", color="tab:gray")
        axis.plot(result["request"], result[corr_col], marker="o", label="twin + online correction", color="tab:blue")
        axis.axvline(args.disturb_at - 0.5, color="tab:red", linestyle="--", label="disturbance starts")
        axis.set_ylabel("error (%)")
        axis.set_title(title)
        axis.grid(alpha=0.3)

    bottom.set_xlabel("request number")
    top.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=130)
    print(f"\nWrote {OUTPUT_CSV} and {OUTPUT_PNG}")


if __name__ == "__main__":
    main()