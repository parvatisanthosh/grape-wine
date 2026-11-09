"""The measured v2 runs, end to end against tests/fake_worker.py.

No model is loaded, but the real Twin and dataset are used for ORBIT's
decisions, so this checks the whole path from trace to analysis.
Run from orbit-llm/:  python -m pytest tests -q
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab"))

import evaluate_v2 as ev  # noqa: E402
from orbit_worker_client import OrbitWorker  # noqa: E402
from quality_core import load_jsonl  # noqa: E402

FAKE = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]


def fake_worker():
    return OrbitWorker(command=FAKE)


def write_trace(path):
    rows = [
        {"index": 1, "stream": "chat", "prompt_id": "run_1#u1", "prompt_tokens": 300,
         "output_tokens": 16, "turn": 1, "conversation": 1},
        {"index": 2, "stream": "code", "prompt_id": "run_3#r2", "prompt_tokens": 600,
         "output_tokens": 8, "turn": 0, "conversation": None},
        {"index": 3, "stream": "one_shot", "prompt_id": "run_2#r3", "prompt_tokens": 3000,
         "output_tokens": 16, "turn": 0, "conversation": None},
        {"index": 4, "stream": "chat", "prompt_id": "run_1#u1", "prompt_tokens": 400,
         "output_tokens": 16, "turn": 2, "conversation": 1},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def context():
    import os

    os.chdir(ROOT)
    return ev.Context.load()


def test_pool_covers_active_chats_and_the_largest_request(tmp_path):
    trace = ev.load_trace(write_trace(tmp_path / "t.jsonl"))
    pool = ev.pool_for(trace, active_users=4)
    # Longest chat 416 tokens, largest request 3016, then 4 + 1 spare blocks.
    assert pool["GPU"] == 4 * 26 + 189 + 5
    assert pool["CPU"] == 4 * 13 + 95 + 5


def test_judge_checks_each_limit():
    request = Namespace(max_ttft_ms=2000.0, max_tpot_ms=100.0)
    row = {"status": "success", "ttft_ms": 2500.0, "tpot_ms_per_token": 120.0,
           "generation_ms": 4000.0, "sys_available_ram_mib": 400.0}
    result = ev.judge(request, row, load_ms=100.0)
    assert result["violations"] == "ttft,tpot,memory"
    assert result["total_ms"] == 4100.0
    assert ev.judge(request, {"status": "error"}, 0.0)["violations"] == "failed"


def test_fixed_system_is_refused_requests_it_may_not_serve(tmp_path, context):
    from evaluate_controller import ResidentService

    trace = ev.load_trace(write_trace(tmp_path / "t.jsonl"))
    service = ResidentService(context.models, 1, fake_worker, kv_blocks=ev.pool_for(trace))
    try:
        outcomes = [ev.serve("fixed", r, service, None, context, fixed="INT4@gpu") for r in trace]
    finally:
        service.close()
    assert [o["violations"] for o in outcomes][1:3] == ["quality", "context"]
    assert outcomes[0]["status"] == "success"


def test_sweep_compare_and_analyze_run_end_to_end(tmp_path, context):
    trace = write_trace(tmp_path / "t.jsonl")
    sweep_out = tmp_path / "sweep.jsonl"

    best = ev.sweep(trace, 2, sweep_out, worker_factory=fake_worker, context=context)
    assert (tmp_path / "best_fixed.json").exists()
    assert len(load_jsonl(sweep_out)) == 8  # 2 candidates x 4 requests

    for pass_name in ("A", "B"):
        out = tmp_path / f"pass_{pass_name}.jsonl"
        ev.compare(trace, pass_name, best, out, worker_factory=fake_worker, context=context)
        rows = load_jsonl(out)
        assert len(rows) == 4 * len(ev.PASSES[pass_name])
        # Running again resumes: nothing is repeated.
        ev.compare(trace, pass_name, best, out, worker_factory=fake_worker, context=context)
        assert len(load_jsonl(out)) == len(rows)

    rows = load_jsonl(tmp_path / "pass_A.jsonl")
    orbit_rows = [r for r in rows if r["system"] == "orbit"]
    # ORBIT never sends code or long prompts to TinyLlama.
    assert {r["candidate"].split("@")[0] for r in orbit_rows if r["index"] in (2, 3)} <= {"Q3B-INT4"}

    results = ev.analyze([tmp_path / "pass_A.jsonl", tmp_path / "pass_B.jsonl"])
    assert any(result["pair"] == "orbit vs best_fixed" for result in results)
