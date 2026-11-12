"""kv_quality.py: retrieval prompts, scoring and answer comparison (no model needed).

Run from orbit-llm/:  python -m pytest tests -q
"""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kv_quality as kq  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    if not Path(kq.QWEN_TOKENIZER).exists():
        pytest.skip("Qwen tokenizer not on this machine")
    return tokenizers.Tokenizer.from_file(kq.QWEN_TOKENIZER)


def test_needle_prompt_has_the_length_and_one_answer(tokenizer):
    replies = ["Rivers carry water to the sea. They shape valleys over time. " * 3,
               "Bread rises because yeast makes gas. Bakers knead dough to build gluten."]
    rng = random.Random(1)
    content, answer, decoys = kq.needle_prompt(tokenizer, replies, rng, 600, 0.5)
    tokens = len(tokenizer.encode(content).ids)
    assert abs(tokens - 600) <= 8  # sentence joins can merge a few tokens
    assert content.count(answer) == 1 and all(content.count(d) == 1 for d in decoys)
    assert len({answer, *decoys}) == 3
    assert content.rstrip().endswith("Answer with the number only.")
    # The real fact sits near the requested depth.
    assert 0.3 < content.index(answer) / len(content) < 0.7


def test_needle_outcome_reads_the_first_six_digit_number():
    key = {"answer": "482913", "decoys": ["111222", "333444"]}
    assert kq.needle_outcome("482913", key) == "correct"
    assert kq.needle_outcome("The code is 482913.", key) == "correct"
    assert kq.needle_outcome("111222", key) == "decoy"
    assert kq.needle_outcome("I don't know", key) == "wrong"
    assert kq.needle_outcome("333444 or 482913", key) == "decoy"


def test_first_difference():
    assert kq.first_difference("abc", "abc") is None
    assert kq.first_difference("abc", "abd") == 2
    assert kq.first_difference("ab", "abc") == 2


def test_compare_counts_identical_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(kq, "OUT_DIR", tmp_path)
    for kv, answers in {"u4": ["x", "y2"], "f16": ["x", "y"]}.items():
        rows = [{"id": f"p{i}", "answer": a, "tokens": 5, "tpot_ms": 50.0, "ttft_ms": 900.0}
                for i, a in enumerate(answers)]
        kq.answers_path("M", "GPU", kv).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    summary = {row["kv"]: row for row in kq.compare("M", "GPU", ["u4", "f16"])}
    assert summary["f16"]["identical_to_reference"] == 2
    assert summary["u4"]["identical_to_reference"] == 1
    assert summary["u4"]["median_first_difference_chars"] == 1
