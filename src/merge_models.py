import os
import torch

from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM

base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
checkpoint_folder = "Checkpoints/20250223-081257"
cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"

base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_dir,
    attn_implementation="flash_attention_2",
)

peft_model = PeftModel.from_pretrained(
    base_model,
    checkpoint_folder,
    torch_dtype=torch.float16,
    device_map="auto",
)

merged_model = peft_model.merge_and_unload()
processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

save_path = "Saved_models"
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
merged_model.save_pretrained(save_path)
processor.save_pretrained(save_path)