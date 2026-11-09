"""Worker-mode evaluation: residency, eviction and prefix-cache bookkeeping.

Uses tests/fake_worker.py, so no OpenVINO or GPU is needed.
"""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orbit  # noqa: E402
from evaluate_controller import (  # noqa: E402
    CONVERSATION_PROMPT_ID,
    ResidentService,
    build_trace,
    charged_load_ms,
    load_options,
)
from orbit_worker_client import OrbitWorker  # noqa: E402

FAKE = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]
MODELS = {"INT4": {"path": "C:/models/int4"}, "Q3B-INT4": {"path": "C:/models/q3b"}}


def service(max_resident, kv_blocks=0):
    return ResidentService(
        MODELS, max_resident, worker_factory=lambda: OrbitWorker(command=FAKE), kv_blocks=kv_blocks
    )


def request(prompt_id, prompt_tokens=256, repeat_prefix=False, max_memory_mib=None):
    return Namespace(
        prompt_id=prompt_id,
        prompt_key=f"{prompt_id}:{prompt_tokens}",
        prompt_tokens=prompt_tokens,
        output_tokens=16,
        repeat_prefix=repeat_prefix,
        max_memory_mib=max_memory_mib,
    )


@pytest.fixture
def two_slots():
    running = service(max_resident=2)
    yield running
    running.close()


def test_first_request_loads_and_later_ones_do_not(two_slots):
    first = two_slots.run("INT4", "gpu", request("run_2#1"))
    second = two_slots.run("INT4", "gpu", request("run_3#2"))

    assert first["pipeline_construction_ms"] == 2500.0
    assert second["pipeline_construction_ms"] is None
    assert charged_load_ms(second, switched=False, executor="worker") == 0.0


def test_default_pool_keeps_only_the_previous_prompt(two_slots):
    conversation = request(CONVERSATION_PROMPT_ID, repeat_prefix=True)

    assert not two_slots.run("INT4", "gpu", conversation)["cache_hit"]
    assert two_slots.run("INT4", "gpu", conversation)["cache_hit"]
    # Any other request in between pushes the conversation out (Finding 26).
    two_slots.run("INT4", "gpu", request("run_2#5"))
    back = two_slots.run("INT4", "gpu", conversation)
    assert not back["cache_hit"]
    assert back["ttft_ms"] == 1500.0
    # Another model never saw the conversation.
    assert not two_slots.run("Q3B-INT4", "gpu", conversation)["cache_hit"]


def test_fixed_pool_keeps_prompts_that_fit():
    # Two 256-token prompts + 16 output tokens need 17 GPU blocks each.
    fits = service(max_resident=1, kv_blocks=34)
    try:
        first, second = request("run_1#a"), request("run_2#b")
        fits.run("INT4", "gpu", first)
        fits.run("INT4", "gpu", second)
        assert fits.run("INT4", "gpu", first)["cache_hit"]
        assert fits.run("INT4", "gpu", second)["cache_hit"]
    finally:
        fits.close()


def test_pool_size_can_be_set_per_device():
    # GPU blocks hold 16 tokens: two 256+16-token prompts need 34 of them.
    per_device = service(max_resident=1, kv_blocks={"GPU": 34, "CPU": 1})
    try:
        first, second = request("run_1#a"), request("run_2#b")
        per_device.run("INT4", "gpu", first)
        per_device.run("INT4", "gpu", second)
        assert per_device.run("INT4", "gpu", first)["cache_hit"]
        assert per_device.pool_blocks("CPU") == 1
    finally:
        per_device.close()


def test_pool_one_block_short_loses_every_prompt_when_taking_turns():
    short = service(max_resident=1, kv_blocks=33)
    try:
        first, second = request("run_1#a"), request("run_2#b")
        short.run("INT4", "gpu", first)
        short.run("INT4", "gpu", second)
        assert not short.run("INT4", "gpu", first)["cache_hit"]
        assert not short.run("INT4", "gpu", second)["cache_hit"]
    finally:
        short.close()


def test_measured_hit_comes_from_timing(two_slots):
    conversation = request(CONVERSATION_PROMPT_ID, repeat_prefix=True)
    assert two_slots.run("INT4", "gpu", conversation)["measured_hit"] is False
    assert two_slots.run("INT4", "gpu", conversation)["measured_hit"] is True
    assert two_slots.run("INT4", "gpu", request("run_2#9", prompt_tokens=64))["measured_hit"] is None


def test_least_recently_used_model_is_evicted_when_full():
    one = service(max_resident=1)
    try:
        one.run("INT4", "gpu", request("run_2#1"))
        switched = one.run("Q3B-INT4", "gpu", request("run_3#2"))

        assert switched["evicted"] == ["INT4@gpu"]
        assert one.resident == ["Q3B-INT4@gpu"]
        # A real switch pays the unload and the load.
        assert charged_load_ms(switched, switched=True, executor="worker") == 150.0 + 2500.0
    finally:
        one.close()


def test_using_a_model_makes_it_most_recently_used(two_slots):
    two_slots.run("INT4", "gpu", request("run_2#1"))
    two_slots.run("Q3B-INT4", "gpu", request("run_3#2"))
    two_slots.run("INT4", "gpu", request("run_4#3"))  # INT4 used again
    third = two_slots.run("INT4", "cpu_auto", request("run_5#4"))

    assert third["evicted"] == ["Q3B-INT4@gpu"]  # the one unused for longest


def test_eviction_throws_the_prefix_cache_away():
    one = service(max_resident=1)
    try:
        conversation = request(CONVERSATION_PROMPT_ID, repeat_prefix=True)
        one.run("INT4", "gpu", conversation)
        one.run("Q3B-INT4", "gpu", request("run_3#2"))

        back = one.run("INT4", "gpu", conversation)

        assert back["pipeline_construction_ms"] == 2500.0
        assert not back["cache_hit"]
    finally:
        one.close()


def test_memory_limited_request_runs_alone(two_slots):
    two_slots.run("Q3B-INT4", "gpu", request("run_2#1"))
    two_slots.run("INT4", "gpu", request("run_3#2"))

    capped = two_slots.run("INT4", "gpu", request("run_4#3", max_memory_mib=1500))

    assert capped["evicted"] == ["Q3B-INT4@gpu"]
    assert two_slots.resident == ["INT4@gpu"]
    assert capped["sampled_peak_rss_mib"] == 900.0  # runtime + one model only
    assert capped["unload_ms_charged"] == 150.0


def test_two_resident_models_cost_memory(two_slots):
    two_slots.run("INT4", "gpu", request("run_2#1"))
    both = two_slots.run("Q3B-INT4", "gpu", request("run_3#2"))

    assert both["sampled_peak_rss_mib"] == 1400.0


def test_shared_state_lets_orbit_see_residency(two_slots):
    conversation = request(CONVERSATION_PROMPT_ID, repeat_prefix=True)
    two_slots.run("INT4", "gpu", conversation)
    state = {"loaded_candidate": None}

    two_slots.share_state(state)

    assert orbit.is_resident(state, "INT4@gpu")
    assert not orbit.is_resident(state, "Q3B-INT4@gpu")
    assert orbit.prefix_cached(state, "INT4@gpu", conversation)
    assert not orbit.prefix_cached(state, "INT4@gpu", request("run_2#9", repeat_prefix=True))


def test_without_a_worker_residency_is_the_last_candidate_run():
    state = {"loaded_candidate": "INT4@gpu"}
    repeat = request(CONVERSATION_PROMPT_ID, repeat_prefix=True)

    assert orbit.is_resident(state, "INT4@gpu")
    assert orbit.prefix_cached(state, "INT4@gpu", repeat)
    assert not orbit.prefix_cached(state, "INT8@gpu", repeat)


def test_fresh_executor_charges_load_only_on_switch():
    row = {"pipeline_construction_ms": 2000.0}

    assert charged_load_ms(row, switched=True, executor="fresh") == 2000.0
    assert charged_load_ms(row, switched=False, executor="fresh") == 0.0


def test_every_request_gets_its_own_prompt_except_the_conversation():
    trace = build_trace(repeats=3, busy_repeats=3)

    others = [r.prompt_id for r in trace if not r.repeat_prefix]
    follow_ups = {r.prompt_id for r in trace if r.repeat_prefix}

    assert len(others) == len(set(others))
    assert follow_ups == {CONVERSATION_PROMPT_ID}
    assert all("#" in prompt_id for prompt_id in others)


def test_load_options_match_the_placement():
    assert load_options("INT4", "gpu") == {
        "label": "INT4", "device": "GPU", "cache_dir": orbit.GPU_CACHE_DIR,
    }
    assert load_options("INT8", "cpu_pcore") == {
        "label": "INT8", "device": "CPU", "cores": "pcore", "threads": 0,
    }


# ---------------------------------------------------------------------------
# What ORBIT learns from worker runs


def choice():
    return {
        "candidate": "INT4@gpu",
        "label": "INT4",
        "placement": "gpu",
        "prompt_mode": "cold",
        "cache_hit": 0,
        "raw_ttft_ms": 1000.0,
        "raw_tpot_ms": 30.0,
        "raw_peak_mib": 1000.0,
        "raw_load_ms": 2000.0,
        "est_load_ms": 2000.0,
        "pred_ttft_ms": 1000.0,
        "pred_tpot_ms": 30.0,
        "pred_peak_mib": 1000.0,
    }


def learn_from(**row):
    """Run run_and_learn on a made-up worker row; return the INT4 factors."""
    measured = {
        "status": "success",
        "ttft_ms": 2000.0,  # twice the prediction
        "tpot_ms_per_token": 60.0,  # twice the prediction
        "sampled_peak_rss_mib": 2000.0,  # twice the prediction
        "pipeline_construction_ms": None,
        **row,
    }
    state = {"corrections": {}, "gpu_cache_warm": [], "loaded_candidate": None}
    orbit.run_and_learn(
        choice(), request("run_2#1"), MODELS, state, executor=lambda c, a: measured
    )
    return state["corrections"]["INT4@gpu"]


def test_first_request_after_a_load_does_not_teach_ttft():
    first = learn_from(generation_index=1, loaded_slots="INT4@gpu")

    assert first["ttft_factor_miss"] == 1.0  # skipped
    assert first["tpot_factor"] > 1.0  # decode is still learned


def test_later_requests_teach_ttft():
    later = learn_from(generation_index=2, loaded_slots="INT4@gpu")

    assert later["ttft_factor_miss"] > 1.0


def test_unexpected_cache_hit_does_not_teach_ttft():
    # Predicted a miss (choice cache_hit = 0) but the run hit the cache.
    surprise = learn_from(generation_index=2, loaded_slots="INT4@gpu", measured_hit=True)

    assert surprise["ttft_factor_miss"] == 1.0
    assert surprise["ttft_factor_hit"] == 1.0
    assert surprise["tpot_factor"] > 1.0


def test_memory_is_not_learned_while_other_models_are_loaded():
    shared = learn_from(generation_index=2, loaded_slots="INT4@gpu,Q3B-INT4@gpu")
    alone = learn_from(generation_index=2, loaded_slots="INT4@gpu")

    assert shared["peak_factor"] == 1.0
    assert alone["peak_factor"] > 1.0


def test_fresh_process_rows_teach_everything():
    # orbit_run rows have neither field: nothing is skipped.
    fresh = learn_from()

    assert fresh["ttft_factor_miss"] > 1.0
    assert fresh["peak_factor"] > 1.0