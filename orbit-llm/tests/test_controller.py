"""Unit tests for the Performance Twin and the controller logic.

Pure logic only: no models are loaded and nothing is executed.
Run from orbit-llm/:  python -m pytest tests -q
"""

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orbit  # noqa: E402
from evaluate_controller import judge  # noqa: E402
from twin_v0 import AnalyticalTwin, GroupedLinearLaw, noise_floor  # noqa: E402


# ---------------------------------------------------------------------------
# Twin


def test_linear_law_fits_two_points_exactly():
    frame = pd.DataFrame({"g": ["a", "a"], "x": [100, 300], "y": [10.0, 30.0]})
    law = GroupedLinearLaw(lambda f: f["x"], [["g"]]).fit(frame, "y")

    prediction = law.predict(pd.DataFrame({"g": ["a"], "x": [200]}))

    assert prediction[0] == pytest.approx(20.0)


def test_single_point_group_is_constant_by_default():
    frame = pd.DataFrame({"g": ["a"], "x": [100], "y": [50.0]})
    law = GroupedLinearLaw(lambda f: f["x"], [["g"]]).fit(frame, "y")

    prediction = law.predict(pd.DataFrame({"g": ["a"], "x": [1000]}))

    assert prediction[0] == pytest.approx(50.0)


def test_single_point_group_scales_when_proportional():
    # Regression test: TTFT seen only at 1024 tokens must not be predicted
    # flat for a 128-token request (the 10x error found in evaluation).
    frame = pd.DataFrame({"g": ["a"], "x": [1024], "y": [8000.0]})
    law = GroupedLinearLaw(
        lambda f: f["x"],
        [["g"]],
        single_point="proportional",
    ).fit(frame, "y")

    prediction = law.predict(pd.DataFrame({"g": ["a"], "x": [128]}))

    assert prediction[0] == pytest.approx(1000.0)


def test_unseen_group_falls_back_to_coarser_level():
    frame = pd.DataFrame(
        {
            "fine": ["a", "a"],
            "coarse": ["c", "c"],
            "x": [0, 10],
            "y": [5.0, 15.0],
        }
    )
    law = GroupedLinearLaw(
        lambda f: f["x"],
        [["fine", "coarse"], ["coarse"]],
    ).fit(frame, "y")

    prediction = law.predict(
        pd.DataFrame({"fine": ["unseen"], "coarse": ["c"], "x": [5]})
    )

    assert prediction[0] == pytest.approx(10.0)


def make_runs(cache_hit, ttft):
    return {
        "precision": "INT4",
        "device": "CPU",
        "backend": "pa",
        "cores": "any",
        "threads": 0,
        "kv_precision": "u8",
        "prompt_tokens": 512,
        "output_tokens": 32,
        "cache_hit": cache_hit,
        "ttft_ms": ttft,
        "tpot_ms_per_token": 50.0,
        "sampled_peak_rss_mib": 1350.0,
    }


def test_twin_predicts_cache_hits_separately():
    frame = pd.DataFrame(
        [make_runs(0, 3500.0), make_runs(0, 3600.0),
         make_runs(1, 60.0), make_runs(1, 70.0)]
    )
    twin = AnalyticalTwin().fit(frame)

    predictions = twin.predict(
        pd.DataFrame([make_runs(0, 0.0), make_runs(1, 0.0)])
    )

    assert predictions["ttft_ms"][0] > 1000
    assert predictions["ttft_ms"][1] < 200


def test_noise_floor_uses_only_other_runs():
    frame = pd.DataFrame(
        {"config_key": ["a", "a", "a"], "ttft_ms": [10.0, 20.0, 30.0]}
    )

    floor = noise_floor(frame, "ttft_ms")

    # Median of the *other* two runs for each run.
    np.testing.assert_allclose(floor, [25.0, 20.0, 15.0])


# ---------------------------------------------------------------------------
# Optimizer


def request(**overrides):
    values = {
        "prompt_tokens": 512,
        "output_tokens": 64,
        "repeat_prefix": False,
        "objective": "latency",
        "max_ttft_ms": None,
        "max_total_ms": None,
        "max_memory_mib": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_prefix_cache_hit_only_on_loaded_candidate():
    state = {"loaded_candidate": "INT4@gpu", "gpu_cache_warm": [], "corrections": {}}
    frame = orbit.build_candidate_frame(
        request(repeat_prefix=True),
        [("INT4", "gpu"), ("INT8", "gpu")],
        state,
        {"sys_cpu_busy_percent": 10.0, "sys_available_ram_mib": 8000.0},
    )

    hits = dict(zip(frame["candidate"], frame["cache_hit"]))

    assert hits == {"INT4@gpu": 1, "INT8@gpu": 0}


def candidates_frame(rows):
    return pd.DataFrame(
        [
            {
                "candidate": name,
                "label": name.split("@")[0],
                "device": device,
                "pred_ttft_ms": ttft,
                "pred_total_ms": total,
                "pred_peak_mib": peak,
            }
            for name, device, ttft, total, peak in rows
        ]
    )


def test_constraints_and_oom_risk():
    frame = candidates_frame(
        [
            ("A@cpu", "CPU", 900, 4000, 1000),
            ("B@cpu", "CPU", 2000, 4000, 1000),
            ("C@cpu", "CPU", 900, 4000, 3000),
            ("D@gpu", "GPU", 900, 4000, 3000),
        ]
    )

    checked = orbit.check_constraints(
        frame,
        request(max_ttft_ms=1000),
        available_ram_mib=2000,
    )
    violations = dict(zip(checked["candidate"], checked["violations"]))

    assert violations["A@cpu"] == ""
    assert violations["B@cpu"] == "ttft"
    # 3000 MiB + headroom does not fit in 2000 MiB free RAM ...
    assert violations["C@cpu"] == "oom-risk"
    # ... but GPU RSS is a floor, not a full count, so it is not checked.
    assert violations["D@gpu"] == ""


def test_pareto_front_drops_dominated_candidates():
    frame = candidates_frame(
        [
            ("fast@x", "GPU", 0, 1000, 2000),
            ("small@x", "GPU", 0, 3000, 1000),
            ("worse@x", "GPU", 0, 3500, 2500),
        ]
    )

    front = orbit.pareto_front(frame)

    assert set(frame.loc[front, "candidate"]) == {"fast@x", "small@x"}


def test_choose_by_objective():
    frame = candidates_frame(
        [
            ("INT4@gpu", "GPU", 0, 1000, 2000),
            ("Q3B-INT4@gpu", "GPU", 0, 3000, 2600),
            ("INT8@cpu", "CPU", 0, 2000, 900),
        ]
    ).assign(feasible=True)
    quality = {
        "INT4": {"bits_per_byte": 0.9},
        "Q3B-INT4": {"bits_per_byte": 0.6},
        "INT8": {"bits_per_byte": 0.8},
    }

    assert orbit.choose(frame, "latency", quality)["candidate"] == "INT4@gpu"
    assert orbit.choose(frame, "memory", quality)["candidate"] == "INT8@cpu"
    assert orbit.choose(frame, "quality", quality)["candidate"] == "Q3B-INT4@gpu"


def test_choose_returns_none_when_nothing_feasible():
    frame = candidates_frame([("A@cpu", "CPU", 0, 1, 1)]).assign(feasible=False)

    assert orbit.choose(frame, "latency", {}) is None


def test_correction_learns_persistent_bias():
    state = {"corrections": {}}

    # The twin under-predicts TTFT by 2x on every run.
    for _ in range(15):
        correction = orbit.update_correction(
            state,
            "INT4@gpu",
            raw_ttft_ms=100.0,
            raw_tpot_ms=50.0,
            actual_ttft_ms=200.0,
            actual_tpot_ms=50.0,
        )

    assert correction["ttft_factor"] == pytest.approx(2.0, rel=0.01)
    assert correction["tpot_factor"] == pytest.approx(1.0)
    assert correction["observations"] == 15


def test_single_noisy_run_moves_correction_only_partly():
    state = {"corrections": {}}

    correction = orbit.update_correction(
        state, "INT4@gpu", 100.0, 50.0, actual_ttft_ms=300.0, actual_tpot_ms=50.0
    )

    assert correction["ttft_factor"] == pytest.approx(1 + orbit.CORRECTION_ALPHA * 2)


# ---------------------------------------------------------------------------
# Evaluation scoring


def measured_row(**overrides):
    row = {
        "status": "success",
        "label": "INT4",
        "pipeline_construction_ms": 2000.0,
        "generation_ms": 3000.0,
        "ttft_ms": 500.0,
        "tpot_ms_per_token": 40.0,
        "sampled_peak_rss_mib": 1300.0,
    }
    row.update(overrides)
    return row


def test_load_time_charged_only_on_switch():
    quality = {"INT4": {"bits_per_byte": 0.9}}

    switched = judge(request(), measured_row(), switched=True, quality=quality)
    stayed = judge(request(), measured_row(), switched=False, quality=quality)

    assert switched["total_ms"] == 5000.0
    assert stayed["total_ms"] == 3000.0


def test_judge_reports_each_violation():
    verdict = judge(
        request(max_ttft_ms=400, max_total_ms=4000, max_memory_mib=1000),
        measured_row(),
        switched=True,
        quality={},
    )

    assert verdict["violated"]
    assert verdict["violations"] == "ttft,total,memory"


def test_failed_run_is_a_violation():
    verdict = judge(request(), {"status": "warmup_failed"}, switched=True, quality={})

    assert verdict["violated"]
    assert verdict["violations"] == "failed"
