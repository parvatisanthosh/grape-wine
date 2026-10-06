"""Quality proxy: bits per byte (BPB) of each model on a fixed local corpus.

BPB = total negative log-likelihood of the text (in bits) / text size in
bytes. Lower is better. Normalizing by bytes instead of tokens makes models
with different tokenizers (TinyLlama, Qwen) comparable, and within one model
family it measures exactly how much quantization degrades the model.

Uses optimum-intel to read logits from the same OpenVINO IR the C++ executor
runs. Run with the venv that has optimum-intel installed:

    C:/Users/parva/venv/Scripts/python.exe quality_eval.py

Corpus: frozen copies of Apache-2.0 files from openvino.genai at bd8d6542
(quality/corpus/): prose_* are English documentation, code_* are Python.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer


CORPUS_DIR = Path("quality/corpus")
MODELS_JSON = Path("models.json")
RAW_JSON = Path("results/quality_raw.json")
QUALITY_JSON = Path("results/quality.json")

# Non-overlapping windows. The first token of each window has no context and
# is not scored; this affects every model the same way.
WINDOW_TOKENS = 512


def get_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    parser.add_argument(
        "--models",
        nargs="+",
        help="Model labels to evaluate (default: all in models.json).",
    )

    return parser.parse_args()


def score_document(model, tokenizer, text):
    """Return (sum of NLL in nats, number of scored tokens)."""
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    total_nll = 0.0
    scored = 0

    for start in range(0, len(token_ids), WINDOW_TOKENS):
        window = token_ids[start : start + WINDOW_TOKENS].unsqueeze(0)

        if window.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model(
                input_ids=window,
                attention_mask=torch.ones_like(window),
            ).logits

        log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = window[0, 1:]
        total_nll -= log_probs.gather(1, targets.unsqueeze(1)).sum().item()
        scored += len(targets)

    return total_nll, scored


def evaluate_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = OVModelForCausalLM.from_pretrained(path, device="CPU")

    results = {}

    for document in sorted(CORPUS_DIR.iterdir()):
        text = document.read_text(encoding="utf-8")
        nll, scored = score_document(model, tokenizer, text)

        results[document.name] = {
            "bytes": len(text.encode("utf-8")),
            "scored_tokens": scored,
            "nll_nats": nll,
        }

    del model
    gc.collect()

    return results


def bits_per_byte(documents, prefix=""):
    selected = {
        name: values
        for name, values in documents.items()
        if name.startswith(prefix)
    }
    nll = sum(values["nll_nats"] for values in selected.values())
    size = sum(values["bytes"] for values in selected.values())
    return nll / math.log(2) / size


def main():
    args = get_arguments()

    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    labels = args.models or list(models)

    raw = (
        json.loads(RAW_JSON.read_text(encoding="utf-8"))
        if RAW_JSON.exists()
        else {}
    )

    for label in labels:
        print(f"Evaluating {label} ...", flush=True)
        raw[label] = evaluate_model(models[label]["path"])

        RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
        RAW_JSON.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    summary = {
        label: {
            "family": models[label]["family"],
            "weights": models[label]["weights"],
            "bits_per_byte": round(bits_per_byte(documents), 4),
            "bits_per_byte_prose": round(bits_per_byte(documents, "prose_"), 4),
            "bits_per_byte_code": round(bits_per_byte(documents, "code_"), 4),
        }
        for label, documents in raw.items()
    }

    QUALITY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'model':<10}{'family':<28}{'BPB':>8}{'prose':>8}{'code':>8}")
    print("-" * 62)
    for label, values in summary.items():
        print(
            f"{label:<10}{values['family']:<28}"
            f"{values['bits_per_byte']:>8.3f}"
            f"{values['bits_per_byte_prose']:>8.3f}"
            f"{values['bits_per_byte_code']:>8.3f}"
        )
    print(f"\nSaved: {QUALITY_JSON.resolve()}")


if __name__ == "__main__":
    main()
