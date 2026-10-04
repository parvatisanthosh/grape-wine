import openvino_genai as ov_genai
import psutil 
import os
import time

def mem_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024*1024)

print(f"Memory before model load : {mem_mb():.1f} MB")


pipe = ov_genai.LLMPipeline(r"C:\Users\parva\tinyllama-int4", "CPU")
print(f"Memory after mode load : {mem_mb():.1f} MB")
prompt = "Write a very long, detailed essay about the history of the Roman Empire, covering its founding, expansion, government, culture, and eventual fall."

for token_count in [50,200,500,1000]:
    start = time.time()
    result = pipe.generate(prompt,max_new_tokens = token_count)
    elapsed = time.time() - start
    print(f"after generating {token_count} tokens : {mem_mb():.1f} MB (took {elapsed:.1f}s)")

