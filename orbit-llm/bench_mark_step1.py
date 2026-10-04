import os
import time

import openvino_genai as ov_genai
import psutil


MODEL_PATH = r"C:\Users\parva\tinyllama-ov"
DEVICE = "CPU"
MAX_NEW_TOKENS = 50

PROMPT = (
    "Write a detailed explanation of the history of the Roman Empire, "
    "including its founding, expansion, government, culture, and fall."
)


def get_rss_mb(process):
    """Return this Python process's resident memory in MiB."""
    return process.memory_info().rss / (1024 * 1024)


process = psutil.Process(os.getpid())

print(f"Model: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print(f"Process memory before loading: {get_rss_mb(process):.1f} MiB")

load_start = time.perf_counter()
pipeline = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
measured_load_ms = (time.perf_counter() - load_start) * 1000

print(f"Process memory after loading: {get_rss_mb(process):.1f} MiB")
print(f"Pipeline construction time: {measured_load_ms:.2f} ms")

config = ov_genai.GenerationConfig()
config.max_new_tokens = MAX_NEW_TOKENS
config.ignore_eos = True
config.do_sample = False
config.apply_chat_template = False

prompt_batch = [PROMPT]

print("\nRunning warm-up...")
pipeline.generate(prompt_batch, config)
print(f"Process memory after warm-up: {get_rss_mb(process):.1f} MiB")

print("Running measured generation...")
wall_start = time.perf_counter()
result = pipeline.generate(prompt_batch, config)
wall_time_ms = (time.perf_counter() - wall_start) * 1000

metrics = result.perf_metrics

input_tokens = metrics.get_num_input_tokens()
output_tokens = metrics.get_num_generated_tokens()
ttft = metrics.get_ttft()
tpot = metrics.get_tpot()
throughput = metrics.get_throughput()
generate_duration = metrics.get_generate_duration()

print("\n--- Benchmark result ---")
print(f"Requested output tokens: {MAX_NEW_TOKENS}")
print(f"Actual input tokens: {input_tokens}")
print(f"Actual generated tokens: {output_tokens}")
print(f"Wall-clock generation time: {wall_time_ms:.2f} ms")
print(f"OpenVINO generation time: {generate_duration.mean:.2f} ms")
print(f"TTFT: {ttft.mean:.2f} ms")
print(f"TPOT: {tpot.mean:.2f} ms/token")
print(f"Throughput: {throughput.mean:.2f} tokens/s")
print(f"Process memory after generation: {get_rss_mb(process):.1f} MiB")