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
from evaluate_controller import BackgroundLoad, build_trace, judge  # noqa: E402
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


def test_outlier_run_is_clipped_before_averaging():
    state = {"corrections": {}}

    # A freak run 10x slower than predicted counts as only RATIO_MAX (3x):
    # 0.7 * 1.0 + 0.3 * 3.0 = 1.6, not 0.7 + 0.3 * 10 = 3.7.
    correction = orbit.update_correction(
        state, "INT4@gpu", 100.0, 50.0, actual_ttft_ms=1000.0, actual_tpot_ms=50.0
    )

    assert correction["ttft_factor"] == pytest.approx(1.6)


def test_memory_underprediction_is_learned_faster_than_overprediction():
    under = {"corrections": {}}
    over = {"corrections": {}}

    # Measured 1500 MiB when 1000 was predicted: ratio 1.5 > 1.0, alpha 0.6.
    # 0.4 * 1.0 + 0.6 * 1.5 = 1.3
    orbit.update_correction(
        under, "A", 100.0, 50.0, 100.0, 50.0, raw_peak_mib=1000.0, actual_peak_mib=1500.0
    )
    # Measured 700 MiB when 1000 was predicted: ratio 0.7 < 1.0, alpha 0.2.
    # 0.8 * 1.0 + 0.2 * 0.7 = 0.94
    orbit.update_correction(
        over, "A", 100.0, 50.0, 100.0, 50.0, raw_peak_mib=1000.0, actual_peak_mib=700.0
    )

    assert under["corrections"]["A"]["peak_factor"] == pytest.approx(1.3)
    assert over["corrections"]["A"]["peak_factor"] == pytest.approx(0.94)


def test_memory_factor_converges_to_persistent_bias():
    state = {"corrections": {}}

    for _ in range(12):
        orbit.update_correction(
            state, "A", 100.0, 50.0, 100.0, 50.0,
            raw_peak_mib=1000.0, actual_peak_mib=1400.0,
        )

    assert state["corrections"]["A"]["peak_factor"] == pytest.approx(1.4, rel=0.01)


def test_state_written_before_memory_factor_existed_still_works():
    state = {
        "corrections": {
            "A": {"ttft_factor": 1.2, "tpot_factor": 1.1, "observations": 4}
        }
    }

    correction = orbit.update_correction(
        state, "A", 100.0, 50.0, 100.0, 50.0, raw_peak_mib=1000.0, actual_peak_mib=1000.0
    )

    assert correction["peak_factor"] == pytest.approx(1.0)
    assert correction["observations"] == 5


def test_predict_applies_learned_memory_factor():
    class FakeTwin:
        def predict(self, frame):
            n = len(frame)
            return {
                "ttft_ms": np.full(n, 100.0),
                "tpot_ms_per_token": np.full(n, 10.0),
                "sampled_peak_rss_mib": np.full(n, 1000.0),
            }

    frame = pd.DataFrame(
        [{"candidate": "A", "label": "INT4", "device": "CPU", "output_tokens": 10}]
    )
    state = {
        "corrections": {"A": {"peak_factor": 1.5}},
        "loaded_candidate": "A",
        "gpu_cache_warm": [],
    }

    result = orbit.predict(frame, FakeTwin(), None, state)

    assert result["raw_peak_mib"].iloc[0] == pytest.approx(1000.0)
    assert result["pred_peak_mib"].iloc[0] == pytest.approx(1500.0)


def explore_frame():
    return pd.DataFrame(
        {
            "candidate": ["best", "second", "third", "infeasible"],
            "feasible": [True, True, True, False],
        }
    )


def test_no_exploration_when_epsilon_is_zero():
    frame = explore_frame()
    chosen = frame.iloc[0]

    result = orbit.maybe_explore(frame, chosen, 0.0, np.random.default_rng(0))

    assert result["candidate"] == "best"


def test_exploration_picks_only_other_feasible_candidates():
    frame = explore_frame()
    chosen = frame.iloc[0]
    rng = np.random.default_rng(0)

    picked = {orbit.maybe_explore(frame, chosen, 1.0, rng)["candidate"] for _ in range(50)}

    assert picked <= {"second", "third"}
    assert picked == {"second", "third"}


def test_exploration_keeps_choice_when_nothing_else_is_feasible():
    frame = pd.DataFrame({"candidate": ["only", "bad"], "feasible": [True, False]})

    result = orbit.maybe_explore(frame, frame.iloc[0], 1.0, np.random.default_rng(0))

    assert result["candidate"] == "only"


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


def test_trace_has_calm_phase_then_busy_phase():
    trace = build_trace(repeats=2, busy_repeats=3)

    calm = [r for r in trace if not r.busy]
    busy = [r for r in trace if r.busy]

    # 5 scenarios x 2 repeats, then 3 time-limited scenarios x 3 repeats.
    assert len(calm) == 10
    assert len(busy) == 9
    # All calm requests come first, so the load is started once and stopped once.
    assert trace[: len(calm)] == calm
    assert {r.phase for r in calm} == {"calm"}
    assert {r.phase for r in busy} == {"busy"}


def test_busy_requests_keep_their_limits_but_get_their_own_names():
    busy = [r for r in build_trace(repeats=1, busy_repeats=1) if r.busy]

    assert {r.scenario for r in busy} == {
        "short_chat_busy",
        "long_document_busy",
        "follow_up_busy",
    }
    follow_up = next(r for r in busy if r.scenario == "follow_up_busy")
    assert follow_up.max_ttft_ms == 1000
    assert follow_up.repeat_prefix


def test_skipping_the_busy_phase():
    trace = build_trace(repeats=1, busy_repeats=0)

    assert len(trace) == 5
    assert not any(r.busy for r in trace)


def test_background_load_starts_and_stops_its_processes():
    load = BackgroundLoad(workers=2, settle_seconds=0)

    load.start()
    try:
        assert load.active
        assert len(load.processes) == 2
        assert all(process.poll() is None for process in load.processes)  # running
        processes = list(load.processes)
    finally:
        load.stop()

    assert not load.active
    assert all(process.poll() is not None for process in processes)  # all gone


def test_background_load_with_zero_workers_does_nothing():
    load = BackgroundLoad(workers=0, settle_seconds=0)

    load.start()

    assert not load.active


def curved_frame():
    # Prefill-like data: time grows faster than linearly with prompt length.
    tokens = np.array([256, 1024, 2048, 4096, 8192], dtype=float)
    return pd.DataFrame({"g": "a", "x": tokens, "y": 1.4 * tokens**1.15})


def test_power_law_recovers_a_curve_and_never_goes_negative():
    frame = curved_frame()
    law = GroupedLinearLaw(
        lambda f: f["x"], [["g"]], single_point="proportional", form="power"
    ).fit(frame, "y")

    short = pd.DataFrame({"g": ["a"], "x": [64.0]})
    prediction = law.predict(short)[0]

    # 1.4 * 64 ** 1.15 = about 169 ms.
    assert prediction == pytest.approx(1.4 * 64**1.15, rel=0.01)
    assert prediction > 0


def test_straight_line_through_the_same_curve_goes_negative_at_short_prompts():
    # This is the bug the power law fixes: the line dips below zero, and the
    # old safety floor of 0.001 ms then hid it.
    frame = curved_frame()
    law = GroupedLinearLaw(lambda f: f["x"], [["g"]]).fit(frame, "y")
    slope, intercept = law.fits[0][("a",)][1:3]

    assert slope * 64 + intercept < 0


def test_power_law_exponent_is_kept_in_a_sane_range():
    # Two noisy points that imply an exponent of 5 must not be trusted.
    frame = pd.DataFrame({"g": "a", "x": [100.0, 200.0], "y": [10.0, 320.0]})
    law = GroupedLinearLaw(
        lambda f: f["x"], [["g"]], form="power"
    ).fit(frame, "y")

    exponent = law.fits[0][("a",)][2]

    assert exponent == pytest.approx(GroupedLinearLaw.POWER_MAX)


def test_power_law_with_one_prompt_length_stays_proportional():
    frame = pd.DataFrame({"g": "a", "x": [512.0, 512.0], "y": [1000.0, 1000.0]})
    law = GroupedLinearLaw(
        lambda f: f["x"], [["g"]], single_point="proportional", form="power"
    ).fit(frame, "y")

    double = pd.DataFrame({"g": ["a"], "x": [1024.0]})

    # One length gives no curve, so assume time scales with tokens: 2x -> 2000.
    assert law.predict(double)[0] == pytest.approx(2000.0)


def cache_dataset():
    return pd.DataFrame(
        {
            "precision": ["INT4", "INT4", "Q3B-INT4", "FP16", "INT8"],
            "device": ["GPU", "GPU", "GPU", "GPU", "CPU"],
            "model_cache": [True, False, True, False, True],
            "load_ms": [2800.0, 8000.0, 4600.0, 17000.0, 2000.0],
        }
    )


def test_empty_cache_folder_means_every_gpu_model_is_cold(tmp_path):
    state = {"gpu_cache_warm": ["INT4"]}  # stale belief from an old session

    orbit.sync_gpu_cache_state(state, cache_dataset(), cache_dir=tmp_path)

    assert state["gpu_cache_warm"] == []


def test_missing_cache_folder_means_every_gpu_model_is_cold(tmp_path):
    state = {"gpu_cache_warm": []}

    orbit.sync_gpu_cache_state(state, cache_dataset(), cache_dir=tmp_path / "nope")

    assert state["gpu_cache_warm"] == []


def test_models_loaded_with_the_cache_before_count_as_warm(tmp_path):
    (tmp_path / "model.blob").write_text("compiled")
    state = {"gpu_cache_warm": []}

    orbit.sync_gpu_cache_state(state, cache_dataset(), cache_dir=tmp_path)

    # INT4 and Q3B-INT4 were loaded cached on the GPU. FP16 never was, and
    # INT8's cached row was on the CPU, which does not use this cache.
    assert state["gpu_cache_warm"] == ["INT4", "Q3B-INT4"]


def test_warm_models_from_this_session_are_kept(tmp_path):
    (tmp_path / "model.blob").write_text("compiled")
    state = {"gpu_cache_warm": ["FP16"]}

    orbit.sync_gpu_cache_state(state, cache_dataset(), cache_dir=tmp_path)

    assert state["gpu_cache_warm"] == ["FP16", "INT4", "Q3B-INT4"]


def candidate_row(name, ttft, total, device="GPU", tpot=25.0, peak=1100.0):
    return {
        "candidate": name,
        "label": name.split("@")[0],
        "device": device,
        "cache_hit": 0,
        "output_tokens": 128,
        "pred_ttft_ms": ttft,
        "pred_tpot_ms": tpot,
        "pred_peak_mib": peak,
        "pred_load_ms": 0.0,
        "pred_total_ms": total,
    }


MARGINS = {
    "ttft_ms": {("GPU",): 1.4, ("GPU", 1): 1.1},
    "tpot_ms": {("GPU",): 1.0},
    "peak_mib": {("GPU",): 1.0},
}


def long_document_request(**limits):
    from argparse import Namespace

    defaults = {"max_ttft_ms": 4000, "max_total_ms": None, "max_memory_mib": None}
    return Namespace(**{**defaults, **limits})


def test_margin_uses_the_most_specific_group_that_exists():
    assert orbit.margin_for(MARGINS, "ttft_ms", "GPU", 1) == pytest.approx(1.1)
    assert orbit.margin_for(MARGINS, "ttft_ms", "GPU", 0) == pytest.approx(1.4)
    assert orbit.margin_for(MARGINS, "ttft_ms", "CPU", 0) == pytest.approx(1.0)
    assert orbit.margin_for(MARGINS, "unknown_metric", "GPU", 0) == pytest.approx(1.0)


def test_prediction_that_barely_fits_is_not_robustly_feasible():
    # Predicted TTFT 3066 ms against a 4000 ms limit looks safe, but with a
    # 1.4x worst case it is 4292 ms, which breaks the promise.
    frame = pd.DataFrame([candidate_row("INT4@gpu", 3066.0, 8000.0)])

    checked = orbit.check_constraints(
        frame, long_document_request(), 8000.0, margins=MARGINS
    )

    assert checked["point_feasible"].iloc[0]
    assert not checked["feasible"].iloc[0]
    assert checked["upper_ttft_ms"].iloc[0] == pytest.approx(3066.0 * 1.4)
    # The worst case misses by 4292 / 4000 - 1 = about 7%.
    assert checked["severity"].iloc[0] == pytest.approx(0.0731, abs=1e-3)
    assert checked["violations"].iloc[0] == "ttft"


def test_without_margins_the_old_behaviour_is_unchanged():
    frame = pd.DataFrame([candidate_row("INT4@gpu", 3066.0, 8000.0)])

    checked = orbit.check_constraints(frame, long_document_request(), 8000.0)

    assert checked["feasible"].iloc[0]
    assert not checked["margins_used"].iloc[0]


def test_select_prefers_a_robustly_safe_candidate_even_if_slower():
    frame = pd.DataFrame(
        [
            candidate_row("fast@gpu", 3066.0, 8000.0),  # worst case 4292: unsafe
            candidate_row("slow@gpu", 2000.0, 9000.0),  # worst case 2800: safe
        ]
    )
    checked = orbit.check_constraints(
        frame, long_document_request(), 8000.0, margins=MARGINS
    )

    row, mode = orbit.select(checked, "latency", {}, fallback=True)

    assert row["candidate"] == "slow@gpu"
    assert mode == "robust"


def test_select_falls_back_to_point_prediction_when_nothing_is_robust():
    frame = pd.DataFrame([candidate_row("only@gpu", 3066.0, 8000.0)])
    checked = orbit.check_constraints(
        frame, long_document_request(), 8000.0, margins=MARGINS
    )

    row, mode = orbit.select(checked, "latency", {}, fallback=True)

    assert row["candidate"] == "only@gpu"
    assert mode == "point-only"


def test_select_uses_the_smallest_miss_when_nothing_fits_at_all():
    frame = pd.DataFrame(
        [
            candidate_row("bad@gpu", 6000.0, 9000.0),  # predicted 50% over
            candidate_row("less_bad@gpu", 4500.0, 9000.0),  # predicted 12.5% over
        ]
    )
    checked = orbit.check_constraints(
        frame, long_document_request(), 8000.0, margins=MARGINS
    )

    row, mode = orbit.select(checked, "latency", {}, fallback=True)

    assert row["candidate"] == "less_bad@gpu"
    assert mode == "best-effort"


def test_select_without_fallback_returns_nothing_when_nothing_fits():
    frame = pd.DataFrame([candidate_row("bad@gpu", 6000.0, 9000.0)])
    checked = orbit.check_constraints(frame, long_document_request(), 8000.0)

    assert orbit.select(checked, "latency", {}) == (None, None)


def test_select_labels_a_plain_choice_as_point_when_no_margins_were_used():
    frame = pd.DataFrame([candidate_row("INT4@gpu", 3066.0, 8000.0)])
    checked = orbit.check_constraints(frame, long_document_request(), 8000.0)

    row, mode = orbit.select(checked, "latency", {})

    assert mode == "point"


def test_power_law_never_grows_slower_than_linearly_beyond_measured_prompts():
    # Short prompts look sublinear (exponent 0.8): a fixed cost dominates.
    tokens = np.array([32.0, 128.0, 1024.0])
    frame = pd.DataFrame({"g": "a", "x": tokens, "y": 8.0 * tokens**0.8})
    law = GroupedLinearLaw(
        lambda f: f["x"], [["g"]], single_point="proportional", form="power"
    ).fit(frame, "y")

    edge = 8.0 * 1024**0.8  # the value at the longest measured prompt
    beyond = pd.DataFrame({"g": ["a"], "x": [2048.0]})

    # Doubling the prompt past the measured range must at least double the time.
    assert law.predict(beyond)[0] == pytest.approx(2 * edge, rel=0.01)
    # Inside the measured range the fitted curve is still used.
    inside = pd.DataFrame({"g": ["a"], "x": [512.0]})
    assert law.predict(inside)[0] == pytest.approx(8.0 * 512**0.8, rel=0.01)


def test_nearest_misses_lists_the_closest_candidates_with_reasons():
    frame = pd.DataFrame(
        [
            candidate_row("far@gpu", 9000.0, 20000.0),
            candidate_row("near@gpu", 4300.0, 9000.0),
            candidate_row("nearest@gpu", 4100.0, 9000.0),
        ]
    )
    checked = orbit.check_constraints(frame, long_document_request(), 8000.0)

    misses = orbit.nearest_misses(checked, count=2)

    assert list(misses["candidate"]) == ["nearest@gpu", "near@gpu"]
    assert (misses["violations"] == "ttft").all()