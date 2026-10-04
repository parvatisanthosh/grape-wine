import openvino_genai as ov_genai


MODEL_PATH = r"C:\Users\parva\tinyllama-ov"

PROMPT = (
    "The Roman Empire developed complex systems of government, "
    "trade, military organization, engineering, and culture."
)

tokenizer = ov_genai.Tokenizer(MODEL_PATH)

tokenized = tokenizer.encode(
    PROMPT,
    add_special_tokens=True,
)

token_ids = tokenized.input_ids.data[0]

print("Original text:")
print(PROMPT)

print("\nToken IDs:")
print(token_ids)

print(f"\nNumber of tokens: {len(token_ids)}")

decoded_text = tokenizer.decode(token_ids)

print("\nDecoded text:")
print(decoded_text)

print("\nSpecial token IDs:")
print(f"BOS token ID: {tokenizer.get_bos_token_id()}")
print(f"EOS token ID: {tokenizer.get_eos_token_id()}")
print(f"PAD token ID: {tokenizer.get_pad_token_id()}")