import os
import datetime
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt

from DataCollator import DataSets
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from metrics import CustomMetrics

class TrainEvalLossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        """
        Wywoływana na koniec każdej epoki, zbiera stratę treningową.
        """
        if logs is not None:
            print(f"[on_epoch_end] logs keys: {list(logs.keys())}", flush = True)
            # Sprawdzamy dostępność klucza 'loss' lub 'train_loss'
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                print(f"[Train @ epoch={state.epoch:.2f}] Loss: {logs['loss']:.4f}", flush = True)
            elif "train_loss" in logs:
                self.train_losses.append(logs["train_loss"])
                print(f"[Train @ epoch={state.epoch:.2f}] Loss: {logs['train_loss']:.4f}", flush = True)
            else:
                print("[on_epoch_end] WARN: No 'loss' or 'train_loss' found in logs.", flush = True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Wywoływana po każdej ewaluacji, zbiera stratę walidacyjną.
        """
        if metrics is not None:
            print(f"[on_evaluate] metrics keys: {list(metrics.keys())}", flush = True)
            if "eval_loss" in metrics:
                self.eval_losses.append(metrics["eval_loss"])
                print(f"[Eval  @ epoch={state.epoch:.2f}] Loss: {metrics['eval_loss']:.4f}", flush = True)
            else:
                print("[on_evaluate] WARN: No 'eval_loss' found in metrics.", flush = True)

def process_years_to_excel(years_to_iterate=[2014, 2015], excel_filename="matched_dates_cleaned_version2.xlsx", debug: bool = False):
        base_json_path = "lemkin-json-from-html"
        json_results = []
        for year_val in years_to_iterate:
            json_folder_base = os.path.join(base_json_path, str(year_val))

            if debug:
                print(f"Przetwarzanie JSON w: {json_folder_base}", flush = True)

            for root, dirs, files in os.walk(json_folder_base):
                for file in files:
                    if file.endswith(".json"):
                        filename_no_ext = os.path.splitext(file)[0]
                        try:
                            year_str, doc_id_str = filename_no_ext.split("_", 1)
                            year = int(year_str)
                            doc_id = doc_id_str.strip()
                        except ValueError:
                            continue

                        match_json = re.search(r'(\d+)$', doc_id)
                        short_doc_id_json = str(int(match_json.group(1))) if match_json else doc_id

                        if len(short_doc_id_json) >= 4:
                            last4_doc_id_json = str(int(short_doc_id_json[-4:]))
                        else:
                            last4_doc_id_json = str(int(short_doc_id_json)) if short_doc_id_json.isdigit() else short_doc_id_json

                        full_path = os.path.join(root, file)
                        json_results.append({
                            "year": year,
                            "json_id": short_doc_id_json,
                            "last4_doc_id": last4_doc_id_json,
                            "json_path": full_path,
                        })
        
        if debug:
            print(f"Znaleziono {len(json_results)} plików JSON.", flush = True)

        base_pdf_path = "lemkin-pdf"
        png_results = []
        for year_val in years_to_iterate:
            pdf_folder_base = os.path.join(base_pdf_path, str(year_val))
            if not os.path.isdir(pdf_folder_base):

                if debug:
                    print(f"Folder nie istnieje: {pdf_folder_base}", flush = True)

                continue
            for root, dirs, files in os.walk(pdf_folder_base):
                for d in dirs:
                    if d.endswith("_png"):
                        folder_path = os.path.join(root, d)
                        
                        base_name = d[:-4] if d.endswith("_png") else d

                        match_png = re.search(r'(\d+)$', base_name)
                        short_doc_id_png = str(int(match_png.group(1))) if match_png else None
                        if short_doc_id_png and len(short_doc_id_png) >= 4:
                            last4_doc_id_png = str(int(short_doc_id_png[-4:]))
                        elif short_doc_id_png:
                            last4_doc_id_png = str(int(short_doc_id_png))
                        else:
                            last4_doc_id_png = None

                        png_results.append({
                            "year": year_val,
                            "png_id": short_doc_id_png,
                            "last4_doc_id": last4_doc_id_png,
                            "image_folder_path": folder_path,
                        })

        df_json = pd.DataFrame(json_results)
        df_png = pd.DataFrame(png_results)

        df_merged = pd.merge(df_json, df_png, on=["year", "last4_doc_id"], how="inner")

        df_result = df_merged[["year", "json_path", "image_folder_path"]]
        df_result = df_result.rename(columns={
            "json_path": "JSON file path",
            "image_folder_path": "Image folder path"
        })

        if debug:
            print("Przykładowe dopasowania:", flush = True)
            print(df_result.head(), flush = True)
            print(len(df_result), "dopasowań znaleziono.", flush = True)

        df_result.to_excel(excel_filename, index=False)

        if debug:
            print(f"Dane zapisane do {excel_filename}", flush = True)
        
        return df_result

df_result = process_years_to_excel([2014, 2015], "matched_dates_cleaned_version2.xlsx")
 
datacollator = DataSets()

train_set, valid_set, test_set = datacollator.split_datasets()
collator = datacollator.collate_fn

# debug datasets
print(f"Train set: {train_set}", flush = True)
print(f"Len train set: {len(train_set)}", flush = True)
print(f"Valid set: {valid_set}", flush = True)
print(f"Len valid set: {len(valid_set)}", flush = True)
print(f"Test set: {test_set}", flush = True)
print(f"Len test set: {len(test_set)}", flush = True)
 
# Hugging Face model id
model_id = "Qwen/Qwen2-VL-7B-Instruct" 
 
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
 
local_rank = int(os.environ["LOCAL_RANK"])
# load model and tokenizer
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    # device_map="auto",
    device_map={"": f"cuda:{local_rank}"},
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = 'right'
 
# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", 
)
 
args = SFTConfig(
    output_dir="qwen2-output", # directory to save and repository id
    num_train_epochs=15,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=False,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=False,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="linear",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
    dataset_kwargs = {"skip_prepare_dataset": True},
    load_best_model_at_end=True,
)
args.remove_unused_columns=False
args.ddp_find_unused_parameters = False

loss_callback = TrainEvalLossCallback()
 
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    data_collator=collator,
    peft_config=peft_config,
    processing_class=processor.tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 2)]
)

# Uruchom trening
trainer.train()

# Zbierz straty z log_history
log_history = trainer.state.log_history

train_losses = []
eval_losses = []

for log in log_history:
    if "loss" in log and "eval_loss" not in log:
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        eval_losses.append(log["eval_loss"])

print(f"Zebrano {len(train_losses)} wartości train_loss i {len(eval_losses)} wartości eval_loss.")

subfolder = "Results_images"
train_folder = os.path.join(subfolder, "train_loss")
eval_folder = os.path.join(subfolder, "eval_loss")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(eval_folder, exist_ok=True)

# Przygotowanie danych do wykresów
epochs_train = range(1, len(train_losses) + 1)
epochs_eval = range(1, len(eval_losses) + 1)

# Tworzenie i zapis wykresu dla train loss
plt.figure(figsize=(8, 6))
plt.plot(epochs_train, train_losses, label='Train Loss', color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.legend()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_filename = f"train_loss_{timestamp}.png"
train_filepath = os.path.join(train_folder, train_filename)
plt.savefig(train_filepath, dpi=300, bbox_inches='tight')
plt.close()  # Zamknięcie figury, aby nie nakładać kolejnych wykresów
print(f"Wykres strat treningowych zapisano w: {train_filepath}")

# Tworzenie i zapis wykresu dla eval loss
plt.figure(figsize=(8, 6))
plt.plot(epochs_eval, eval_losses, label='Eval Loss', color='red', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss per Epoch')
plt.grid(True)
plt.legend()

eval_filename = f"eval_loss_{timestamp}.png"
eval_filepath = os.path.join(eval_folder, eval_filename)
plt.savefig(eval_filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"Wykres strat walidacyjnych zapisano w: {eval_filepath}")

custom_metrics = CustomMetrics()
print("\nObliczam TED-based Accuracy na zbiorze testowym...")
final_ted_acc = custom_metrics.evaluate_ted_accuracy_on_testset(
    test_set=test_set,
    model=trainer.model,
    processor=processor,
    custom_metrics=custom_metrics,
    debug=True,
)
print(f"[TEST] Średni TED-based Accuracy: {final_ted_acc:.4f}")