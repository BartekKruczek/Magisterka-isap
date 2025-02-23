import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import os
import time

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq, get_scheduler
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig
from typing import List
from tqdm import tqdm

from early_stop import EarlyStopping
from custom_datasets import CustomDataSets
from plot_results import PlotResults

custom_sets = CustomDataSets()
plot = PlotResults()

my_time: float = time.time()
time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(my_time))

model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map = "auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
)
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = 'right'

train_set, valid_set, test_set = custom_sets.split_datasets()

train_loader = DataLoader(
    dataset=train_set,
    batch_size=1,
    shuffle=True,           
    collate_fn=custom_sets.collate_fn
)

valid_loader = DataLoader(
    dataset=valid_set,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_sets.collate_fn
)

epochs: int = 10
accumulation_steps: int = 16
train_losses: List[float] = []
valid_losses: List[float] = []

early_stopping = EarlyStopping(patience=3, delta=0.01)

num_training_steps = len(train_loader) * epochs
num_warmup_steps = int(0.1 * num_training_steps)

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

best_ckpt_dir: str = f"Checkpoints/{time_str}"
os.makedirs(best_ckpt_dir, exist_ok=True)

best_val_loss = float("inf")

for epoch in tqdm(range(1, epochs + 1), desc="Training Progress", leave=True):
    model.train()
    running_loss: float = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        outputs = model(**batch)

        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item()

        del batch, outputs
        torch.cuda.empty_cache()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    print(f" \n \n[Epoch {epoch}] Train Loss: {epoch_train_loss:.4f}", flush=True)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            outputs = model(**batch)

            val_loss += outputs.loss.item()

            del batch, outputs
            torch.cuda.empty_cache()

    epoch_val_loss = val_loss / len(valid_loader)
    valid_losses.append(epoch_val_loss)
    print(f"[Epoch {epoch}] Valid Loss: {epoch_val_loss:.4f}", flush=True)

    if early_stopping.best_score is None or -epoch_val_loss > early_stopping.best_score:
        best_val_loss = epoch_val_loss
        model.save_pretrained(best_ckpt_dir)
        processor.save_pretrained(best_ckpt_dir)

    early_stopping(val_loss=epoch_val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

plot.plot_loss_function(train_loss=train_losses, validation_loss=valid_losses)