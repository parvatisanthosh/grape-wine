"""Build the ORBIT v2 request trace from real data (notes/evaluation_v2_plan.md).

Three streams, sampled with a fixed seed:
- chat: OpenAssistant oasst1 English conversations. Each turn's prompt is the
  whole conversation so far, rendered with TinyLlama's chat template and counted
  with its tokenizer. The output is the real assistant reply's length.
- one-shot: prompt and output sizes from the Azure 2023 conversation trace.
- code: prompt and output sizes from the Azure 2023 code trace.

Only lengths come from the data; the worker builds prompts of exactly that
length. A chat turn reuses its conversation's prompt id, so a later turn
starts with the earlier turn's prompt, as in a real chat.

    python build_real_trace.py --seed 1 --requests 200 --out traces/tuning.jsonl
    python build_real_trace.py --seed 2 --requests 200 --out traces/test.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


DATA = Path("data/raw")
TOKENIZER_DIR = Path("C:/Users/parva/tinyllama-int4")

MAX_PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 256
ACTIVE_CHAT_USERS = 4
SHARES = {"chat": 0.50, "one_shot": 0.25, "code": 0.25}
PROMPT_VARIANT = {"chat": "run_1", "one_shot": "run_2", "code": "run_3"}

# TinyLlama's template, as stored in its OpenVINO tokenizer (openvino_tokenizer.xml).
CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n"
    "{% endif %}\n"
    "{% endfor %}"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    return parser.parse_args()


def conversation_paths(messages):
    """One path per English root: at every step follow the best-ranked reply.

    Returns lists of (role, text) that start with the user and alternate.
    messages: DataFrame with message_id, parent_id, role, text, rank, lang.
    """
    english = messages[(messages["lang"] == "en") & (~messages["deleted"].fillna(False))]
    children = defaultdict(list)
    for row in english.itertuples(index=False):
        children[row.parent_id].append(row)

    paths = []
    for root in english[english["parent_id"].isna()].itertuples(index=False):
        if root.role != "prompter":
            continue
        path, node = [], root
        while node is not None:
            path.append(("user" if node.role == "prompter" else "assistant", node.text))
            replies = children.get(node.message_id, [])
            # rank 0 is the best; unranked replies come last, then by id for determinism.
            replies = sorted(
                replies,
                key=lambda r: (r.rank if pd.notna(r.rank) else 1e9, r.message_id),
            )
            node = replies[0] if replies else None
        if path[-1][0] == "user":
            path.pop()  # a turn needs a real reply to give its output length
        if len(path) >= 2:
            paths.append(path)
    return paths


def chat_turns(path, count_prompt, count_text):
    """(prompt_tokens, output_tokens) for each user turn of one conversation."""
    turns = []
    for end in range(0, len(path) - 1, 2):
        history = [{"role": role, "content": text} for role, text in path[: end + 1]]
        reply = path[end + 1][1]
        turns.append((count_prompt(history), count_text(reply)))
    return turns


def cap(prompt_tokens, output_tokens):
    capped = prompt_tokens > MAX_PROMPT_TOKENS or output_tokens > MAX_OUTPUT_TOKENS
    return (
        max(2, min(prompt_tokens, MAX_PROMPT_TOKENS)),
        max(1, min(output_tokens, MAX_OUTPUT_TOKENS)),
        capped,
    )


def interleave(conversations, one_shots, codes, total, rng):
    """Mix the streams in the planned shares; ACTIVE_CHAT_USERS chat at once."""
    streams = list(SHARES)
    weights = [SHARES[name] for name in streams]
    waiting = list(conversations)
    active = []
    one_shots, codes = iter(one_shots), iter(codes)
    trace = []

    while len(trace) < total:
        stream = rng.choices(streams, weights)[0]
        if stream == "chat":
            while len(active) < ACTIVE_CHAT_USERS and waiting:
                active.append(waiting.pop(0))
            if not active:
                continue
            conversation = rng.choice(active)
            turn = conversation["turns"].pop(0)
            if not conversation["turns"]:
                active.remove(conversation)
            trace.append({"stream": "chat", "conversation": conversation["id"], **turn})
        else:
            sizes = next(one_shots if stream == "one_shot" else codes)
            trace.append({"stream": stream, "conversation": None, "turn": 0, **sizes})
    return trace


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    from jinja2 import Template
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    template = Template(CHAT_TEMPLATE)

    def count_prompt(history):
        text = template.render(messages=history, eos_token="</s>", add_generation_prompt=True)
        return len(tokenizer.encode(text).ids)  # includes the BOS token

    def count_text(text):
        return len(tokenizer.encode(text, add_special_tokens=False).ids) + 1  # + EOS

    paths = conversation_paths(pd.read_parquet(args.data / "oasst1" / "oasst1-train.parquet"))
    rng.shuffle(paths)

    conversations = []
    for number, path in enumerate(paths[: args.requests], 1):
        turns = []
        for turn, (prompt, output) in enumerate(chat_turns(path, count_prompt, count_text), 1):
            prompt, output, capped = cap(prompt, output)
            turns.append(
                {"turn": turn, "prompt_tokens": prompt, "output_tokens": output, "capped": capped}
            )
        conversations.append({"id": number, "turns": turns})

    def azure(name):
        frame = pd.read_csv(args.data / "azure" / f"AzureLLMInferenceTrace_{name}.csv")
        frame = frame.sample(frac=1.0, random_state=args.seed)
        for row in frame.itertuples(index=False):
            prompt, output, capped = cap(int(row.ContextTokens), int(row.GeneratedTokens))
            yield {"prompt_tokens": prompt, "output_tokens": output, "capped": capped}

    trace = interleave(conversations, azure("conv"), azure("code"), args.requests, rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for index, request in enumerate(trace, 1):
            if request["stream"] == "chat":
                prompt_id = f"{PROMPT_VARIANT['chat']}#u{request['conversation']}"
            else:
                prompt_id = f"{PROMPT_VARIANT[request['stream']]}#r{index}"
            file.write(
                json.dumps({"index": index, "seed": args.seed, "prompt_id": prompt_id, **request})
                + "\n"
            )

    frame = pd.DataFrame(trace)
    print(f"Wrote {len(trace)} requests to {args.out}")
    print(frame.groupby("stream").agg(
        requests=("prompt_tokens", "size"),
        median_prompt=("prompt_tokens", "median"),
        max_prompt=("prompt_tokens", "max"),
        median_output=("output_tokens", "median"),
        capped=("capped", "mean"),
    ).round(2).to_string())


if __name__ == "__main__":
    main()
