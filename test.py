import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline(r"C:\Users\parva\tinyllama-ov", "CPU")
print(pipe.generate("Explain Photo Synthesis",max_new_tokens = 100))
