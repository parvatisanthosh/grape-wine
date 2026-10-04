import openvino_genai as ov_genai


MODEL_PATH = r"C:\Users\parva\tinyllama-ov"
TARGET_TOKENS = 512


def create_prompt(tokenizer, opening, body):
    text = opening + " " + (body + " ") * 200

    tokenized = tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=TARGET_TOKENS,
        truncation=True,
    )

    token_count = len(
        tokenized.input_ids.data[0]
    )

    if token_count != TARGET_TOKENS:
        raise RuntimeError(
            f"Expected {TARGET_TOKENS} tokens, "
            f"but produced {token_count}."
        )

    return tokenized


pipeline = ov_genai.LLMPipeline(
    MODEL_PATH,
    "CPU",
)

tokenizer = pipeline.get_tokenizer()

warmup_prompt = create_prompt(
    tokenizer,
    "Initialization workload.",
    "Computers execute instructions and process information.",
)

prompt_a = create_prompt(
    tokenizer,
    "Roman history workload.",
    "The Roman Empire developed government trade law and engineering.",
)

prompt_b = create_prompt(
    tokenizer,
    "Space science workload.",
    "Astronomers study planets stars galaxies gravity and radiation.",
)

config = ov_genai.GenerationConfig()
config.max_new_tokens = 8
config.ignore_eos = True
config.do_sample = False
config.apply_chat_template = False


def run_test(label, prompt):
    result = pipeline.generate(prompt, config)
    metrics = result.perf_metrics

    print(
        f"{label:<16} "
        f"input={metrics.get_num_input_tokens():<4} "
        f"TTFT={metrics.get_ttft().mean:.2f} ms"
    )


print("Unmeasured initialization:")
pipeline.generate(warmup_prompt, config)

print("\nMeasured requests:")
run_test("Prompt A first", prompt_a)
run_test("Prompt A repeat", prompt_a)
run_test("Prompt B new", prompt_b)