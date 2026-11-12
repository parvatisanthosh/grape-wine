# Datasets

The raw files live in `data/raw/`, which git ignores. They were downloaded on 2026-11-05.
Nothing in them is executed on the laptop. HumanEval's tests and model-written code run
only on Colab, with Google Drive unmounted.

| Folder | Dataset | Used for | Size | SHA-256 (first 16) | Source |
|---|---|---|---|---|---|
| `azure/` | Azure LLM inference trace 2023: chat, 19,366 requests | Request sizes and arrival pattern | 0.72 MB | `2F1E5B666D4E3055` | [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md), CC-BY |
| `azure/` | Azure LLM inference trace 2023: code, 8,819 requests | Request sizes and arrival pattern | 0.32 MB | `54E9A6D2A4BD06BA` | Same |
| `humaneval/` | HumanEval, 164 problems | Task-level quality of the 3B model | 0.04 MB | `B796127E635A67F9` | [openai/human-eval](https://github.com/openai/human-eval) |
| `arc_easy/` | ARC-Easy test split, 2,376 questions | Task-level quality of TinyLlama | 0.33 MB | `4160597D618AE851` | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| `oasst1/` | OpenAssistant oasst1 train split, 84,437 messages | Real conversations | 37.7 MB | `BBFADF5ED1278BA2` | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1), Apache-2.0 |

## Limits to state when reporting results

- **Azure** has token counts and timestamps but no prompt text, and it covers about one
  hour. It carries production arrival rates that this laptop cannot serve, so a replay
  must slow them down and cap request sizes. 82.9% of chat requests and 61.3% of code
  requests fit within 2,000 prompt tokens and 512 output tokens.
- **Azure's file dates** read 2023-11-16, while its documentation says November 11.
- **oasst1** has real human dialogue, but its English conversations reach at most 7
  messages, or 3 user turns. It suits interleaved conversations, not long chats.
- **ARC-Easy** has some questions labelled 1–4 and one with a fifth option. Normalise or
  drop those before scoring.

## Derived files (committed, rebuilt by `kv_quality.py`)

| File | Built from | Command |
|---|---|---|
| `humaneval_prompts.jsonl` | HumanEval problems wrapped in the Colab instruction (`colab/quality_core.py`) | `python kv_quality.py prompts` |
| `needle_prompts.jsonl` | 72 long-context retrieval prompts. The filler is English oasst1 assistant replies (Apache-2.0); the 6-digit codes are synthetic, with seed 7. | `python kv_quality.py needle-prompts` |
