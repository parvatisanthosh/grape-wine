"""Unit tests for the prefix pool study's planning. Nothing is executed.

Run from orbit-llm/:  python -m pytest tests -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefix_pool_study import blocks_per_conversation, plan_runs  # noqa: E402


def test_blocks_per_conversation_uses_device_block_size():
    assert blocks_per_conversation(1920, 16, "GPU") == 121  # 1936 / 16
    assert blocks_per_conversation(1920, 16, "CPU") == 61  # 1936 / 32, rounded up
    assert blocks_per_conversation(32, 0, "CPU") == 1


def test_boundary_plan_tests_each_k_around_its_need():
    runs = plan_runs([1, 2], need=10, offsets=(-1, 0))
    assert runs == [(0, 1), (9, 1), (10, 1), (0, 2), (19, 2), (20, 2)]


def test_boundary_plan_drops_empty_pools():
    assert plan_runs([1], need=1, offsets=(-1, 0)) == [(0, 1), (1, 1)]


def test_explicit_pools_give_a_full_grid():
    assert plan_runs([1, 2], need=10, pools=[0, 50]) == [(0, 1), (0, 2), (50, 1), (50, 2)]
