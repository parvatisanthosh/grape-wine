"""Unit tests for evaluate_v2.py's request model and decisions. Nothing runs.

Run from orbit-llm/:  python -m pytest tests -q
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate_v2 as ev  # noqa: E402

CONTEXTS = {"INT4": 2048, "Q3B-INT4": 32768}


def request(stream="chat", prompt=100, output=50, turn=1):
    limits = ev.LIMITS[stream]
    return Namespace(
        stream=stream, prompt_tokens=prompt, output_tokens=output, turn=turn,
        objective="latency", max_ttft_ms=limits["max_ttft_ms"],
        max_tpot_ms=limits["max_tpot_ms"], max_total_ms=None, max_memory_mib=None,
        needs_code_quality=limits["needs_code_quality"],
    )


def frame(rows):
    return pd.DataFrame(
        [
            {"candidate": c, "label": c.split("@")[0], "pred_ttft_ms": ttft,
             "pred_tpot_ms": tpot, "pred_total_ms": ttft + 50 * tpot, "pred_peak_mib": 1000.0,
             "violations": "", "feasible": True, "point_feasible": True, "severity": 0.0,
             "margins_used": False}
            for c, ttft, tpot in rows
        ]
    )


def test_load_trace_sets_limits_and_prefix_flags(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [
        {"index": 1, "stream": "chat", "prompt_id": "run_1#u1", "prompt_tokens": 50,
         "output_tokens": 10, "turn": 1, "conversation": 1},
        {"index": 2, "stream": "chat", "prompt_id": "run_1#u1", "prompt_tokens": 90,
         "output_tokens": 10, "turn": 2, "conversation": 1},
        {"index": 3, "stream": "code", "prompt_id": "run_3#r3", "prompt_tokens": 900,
         "output_tokens": 5, "turn": 0, "conversation": None},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    first, second, code = ev.load_trace(path)
    assert not first.repeat_prefix and second.repeat_prefix
    assert second.prompt_key == "run_1#u1:90"
    assert code.max_ttft_ms == 1500.0 and code.max_tpot_ms is None and code.needs_code_quality


def test_context_and_quality_are_hard_limits():
    scores = {"INT4": None, "Q3B-INT4": 0.82}
    assert ev.can_serve("INT4", request(prompt=2000, output=100), CONTEXTS, scores) == "context"
    assert ev.can_serve("INT4", request("code"), CONTEXTS, scores) == "quality"
    assert ev.can_serve("Q3B-INT4", request("code"), CONTEXTS, scores) is None
    assert ev.can_serve("INT4", request("code"), CONTEXTS, {"INT4": 0.3}) == "quality"


def test_tpot_limit_marks_slow_candidates():
    result = ev.add_tpot_limit(frame([("INT4@gpu", 500, 40), ("INT4@cpu_auto", 500, 150)]), request())
    assert result["feasible"].tolist() == [True, False]
    assert result.loc[1, "violations"] == "tpot"
    assert result.loc[1, "severity"] == 0.5


def test_orbit_never_picks_a_candidate_that_may_not_serve():
    candidates = frame([("INT4@gpu", 100, 30), ("Q3B-INT4@gpu", 3000, 60)])
    # What check_constraints marks: the 3B model misses the 1.5 s TTFT limit.
    candidates.loc[1, ["violations", "feasible", "point_feasible", "severity"]] = ["ttft", False, False, 1.0]
    row, mode = ev.orbit_choice(candidates, request("code"), CONTEXTS, {"Q3B-INT4": 0.82}, {})
    assert row["candidate"] == "Q3B-INT4@gpu"
    assert mode == "best-effort"  # it misses the TTFT limit, but it is the only one allowed


def test_orbit_reports_unservable_requests():
    candidates = frame([("INT4@gpu", 100, 30)])
    row, mode = ev.orbit_choice(candidates, request("code"), CONTEXTS, {}, {})
    assert row is None and mode is None


def test_rule_sends_code_and_long_requests_to_the_large_model():
    assert ev.rule_choice(request("chat")) == "INT4@gpu"
    assert ev.rule_choice(request("code")) == "Q3B-INT4@gpu"
    assert ev.rule_choice(request("one_shot", prompt=2000, output=100)) == "Q3B-INT4@gpu"


def test_predicted_outcome_combines_refusal_and_limits():
    scores = {"INT4": None, "Q3B-INT4": 0.82}
    rows = frame([("INT4@gpu", 2500, 30)]).iloc[0]
    assert ev.predicted_outcome(rows, request(), CONTEXTS, scores) == (True, "ttft")
    assert ev.predicted_outcome(rows, request("code"), CONTEXTS, scores) == (True, "quality")
    assert ev.predicted_outcome(None, request(), CONTEXTS, scores) == (True, "unservable")
