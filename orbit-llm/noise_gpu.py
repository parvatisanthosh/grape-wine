"""Keep the iGPU busy by running another LLM on it in a loop (GPU contention test).

Stop with Ctrl+C.
"""

from orbit_exec import run_configuration

NOISE = {
    "label": "NOISE",
    "device": "GPU",
    "prompt_tokens": 512,
    "output_tokens": 1024,
    "warmup": 0,
    "runs": 5,
}

round_number = 0
while True:
    round_number += 1
    print(f"--- noise round {round_number} ---", flush=True)
    run_configuration("C:/Users/parva/tinyllama-int8", NOISE)