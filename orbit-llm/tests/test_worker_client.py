"""Tests for orbit_worker_client.OrbitWorker against tests/fake_worker.py.

No models are loaded: the fake worker speaks the same protocol as
orbit_worker.exe. Run from orbit-llm/:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbit_worker_client import OrbitWorker, WorkerError, quote  # noqa: E402

FAKE = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]


@pytest.fixture
def worker():
    with OrbitWorker(command=FAKE) as running:
        yield running


def test_worker_says_ready_on_start(worker):
    assert worker.ready["command"] == "ready"


def test_load_generate_unload_round_trip(worker):
    assert worker.load("A", "C:/m", label="INT4", device="GPU")["status"] == "success"

    reply = worker.generate("A", prompt_tokens=1024, output_tokens=64)

    assert reply["status"] == "success"
    assert reply["requested_input_tokens"] == 1024
    assert reply["generation_index"] == 1
    assert worker.unload("A")["generations_served"] == 1


def test_loading_an_occupied_slot_is_refused(worker):
    worker.load("A", "C:/m")

    reply = worker.load("A", "C:/m")

    assert reply["status"] == "error"
    assert "unload it first" in reply["error"]


def test_several_slots_can_be_loaded_at_once(worker):
    worker.load("A", "C:/m", device="GPU")
    worker.load("B", "C:/m", device="CPU", cores="pcore")

    assert worker.status()["loaded_slots"] == "A,B"


def test_errors_are_replies_and_the_worker_keeps_going(worker):
    assert worker.generate("missing", 8, 4)["status"] == "error"
    assert worker.unload("missing")["status"] == "error"

    # Still in step: the next command gets its own reply.
    assert worker.status()["status"] == "success"


def test_repeated_prompt_on_the_same_slot_hits_the_prefix_cache(worker):
    worker.load("A", "C:/m")

    first = worker.generate("A", 1024, 64, prompt_id="run_1")
    repeat = worker.generate("A", 1024, 64, prompt_id="run_1")
    other = worker.generate("A", 1024, 64, prompt_id="run_2")

    assert repeat["ttft_ms"] < first["ttft_ms"]
    assert other["ttft_ms"] == first["ttft_ms"]


def test_request_options_are_rejected_at_load(worker):
    with pytest.raises(ValueError):
        worker.load("A", "C:/m", prompt_tokens=1024)


def test_a_crashed_worker_raises_instead_of_hanging(worker):
    with pytest.raises(WorkerError):
        worker._send("crash")

    with pytest.raises(WorkerError):
        worker.status()


def test_words_with_spaces_are_quoted():
    assert quote("C:/My Models/int4") == '"C:/My Models/int4"'
    assert quote("plain") == "plain"
    with pytest.raises(ValueError):
        quote('bad"word')