from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(
    task_type = TaskType.CasualLM,
    target_modules = ["q_proj", "v_proj"],
    r = 8,
    lora_alpha = 32,
    lora_dropout = 0.1,
)

training_args = TrainingArguments(
    output_dir = "./qwen2_sft",
    evaluation_strategy = "epoch",
    logging_strategy = "steps",
    logging_steps = 10,
    save_strategy = "epoch",
    num_train_epochs = 3,
    per_device_train_batch_size = 1,
    learning_rate = 1e-5,
    log_level = "info",
)