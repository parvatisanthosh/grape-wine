"""Unit tests for build_real_trace.py on hand-made data. Nothing is downloaded.

Run from orbit-llm/:  python -m pytest tests -q
"""

import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_real_trace as brt  # noqa: E402


def messages():
    rows = [
        # message_id, parent_id, role, text, rank, lang, deleted
        ("r1", None, "prompter", "hi", None, "en", False),
        ("a1", "r1", "assistant", "best reply", 0, "en", False),
        ("a2", "r1", "assistant", "worse reply", 1, "en", False),
        ("u2", "a1", "prompter", "follow up", None, "en", False),
        ("a3", "u2", "assistant", "second reply", 0, "en", False),
        ("u3", "a3", "prompter", "unanswered", None, "en", False),
        ("r2", None, "prompter", "hola", None, "es", False),
        ("a4", "r2", "assistant", "respuesta", 0, "es", False),
        ("r3", None, "prompter", "no reply", None, "en", False),
    ]
    return pd.DataFrame(
        rows, columns=["message_id", "parent_id", "role", "text", "rank", "lang", "deleted"]
    )


def test_paths_follow_best_reply_and_end_on_an_answer():
    paths = brt.conversation_paths(messages())
    assert paths == [
        [("user", "hi"), ("assistant", "best reply"), ("user", "follow up"), ("assistant", "second reply")]
    ]


def test_chat_turn_prompt_is_the_whole_history():
    path = [("user", "a"), ("assistant", "bb"), ("user", "ccc"), ("assistant", "dddd")]
    seen = []

    def count_prompt(history):
        seen.append([message["content"] for message in history])
        return sum(len(message["content"]) for message in history)

    turns = brt.chat_turns(path, count_prompt, len)
    assert seen == [["a"], ["a", "bb", "ccc"]]
    assert turns == [(1, 2), (6, 4)]


def test_cap_limits_and_flags():
    assert brt.cap(100, 50) == (100, 50, False)
    assert brt.cap(9000, 50) == (brt.MAX_PROMPT_TOKENS, 50, True)
    assert brt.cap(1, 0) == (2, 1, False)


def test_interleave_keeps_turn_order_shares_and_active_users():
    conversations = [
        {"id": n, "turns": [{"turn": t, "prompt_tokens": 10 * t, "output_tokens": 5, "capped": False}
                            for t in range(1, 4)]}
        for n in range(1, 200)
    ]
    sizes = [{"prompt_tokens": 7, "output_tokens": 3, "capped": False}] * 1000
    trace = brt.interleave(conversations, sizes, sizes, 400, random.Random(0))

    assert len(trace) == 400
    shares = Counter(request["stream"] for request in trace)
    assert abs(shares["chat"] / 400 - 0.5) < 0.08

    last_turn, active, finished = {}, set(), set()
    for request in trace:
        if request["stream"] != "chat":
            continue
        conversation = request["conversation"]
        assert request["turn"] == last_turn.get(conversation, 0) + 1
        last_turn[conversation] = request["turn"]
        active.add(conversation)
        if request["turn"] == 3:
            finished.add(conversation)
        assert len(active - finished) <= brt.ACTIVE_CHAT_USERS
