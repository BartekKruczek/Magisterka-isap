import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

max_memory_mapping = {0: "30GB", 1: "30GB", 2: "30GB", 3: "30GB"}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cuda',
    attn_implementation = "flash_attention_2",
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
    quantization_config=quantization_config,
    max_memory=max_memory_mapping,
)
model.to(model.device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)