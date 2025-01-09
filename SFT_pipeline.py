import torch

from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model
from DataCollator import DataSets

datacollator = DataSets()

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    llm_int8_enable_fp32_cpu_offload = True
)

qwen2model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct", 
    torch_dtype = torch.float16,
    device_map = "auto",
    attn_implementation = "flash_attention_2",
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
    quantization_config = quantization_config,
)

peft_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    target_modules = ["q_proj", "v_proj"],
    r = 8,
    lora_alpha = 32,
    lora_dropout = 0.1,
)

peft_model = get_peft_model(qwen2model, peft_config)
peft_model.print_trainable_parameters()

# print datasets
train_set, valid_set = datacollator.split_datasets()
collator = datacollator.collate_fn

print(f"Train set: {train_set}")
print(f"Valid set: {valid_set}")
print(f"Collator: {collator}")

print(f"Train set len: {len(train_set)}")
print(f"Valid set len: {len(valid_set)}")

args = SFTConfig(
    output_dir = "qwen2-72b",
    num_train_epochs = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8,
    gradient_checkpointing = True,
    optim = "adamw_torch_fused",
    logging_steps = 1,
    save_strategy = "epoch",
    learning_rate = 2e-4,
    max_grad_norm = 0.3,
    warmup_ratio = 0.03,
    fp16 = True,
    lr_scheduler_type = "constant",
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    dataset_text_field = "",
    dataset_kwargs = {"skip_prepare_dataset": True},
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

trainer = SFTTrainer(
    model = peft_model,
    args = args,
    train_dataset = train_set,
    eval_dataset = valid_set,
    data_collator = collator,
    peft_config = peft_config,
    tokenizer = tokenizer,
)

trainer.train()