import os
import datetime
import re
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback, Qwen2VLForConditionalGeneration
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from utils import Utils
from sklearn.model_selection import train_test_split

# local_rank = int(os.environ["LOCAL_RANK"])
# os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

mytime = time.time()

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
            print(f"[on_epoch_end] logs keys: {list(logs.keys())} \n", flush = True)
            # Sprawdzamy dostępność klucza 'loss' lub 'train_loss'
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                print(f"[Train @ epoch={state.epoch:.2f}] Loss: {logs['loss']:.4f} \n", flush = True)
            elif "train_loss" in logs:
                self.train_losses.append(logs["train_loss"])
                print(f"[Train @ epoch={state.epoch:.2f}] Loss: {logs['train_loss']:.4f} \n", flush = True)
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
                print(f"[Eval  @ epoch={state.epoch:.2f}] Loss: {metrics['eval_loss']:.4f} \n", flush = True)
            else:
                print("[on_evaluate] WARN: No 'eval_loss' found in metrics.", flush = True)

def process_years_to_excel(years_to_iterate, excel_filename="matched_dates_cleaned_version2.xlsx", debug: bool = True):
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

def read_excel() -> pd.DataFrame:
        return pd.read_excel("matched_dates_cleaned_version2.xlsx", engine = 'openpyxl')
    
def sorting_by_page_number(png_path: str = None) -> int:
    split1: str = png_path.split("/")[-1]
    split2 = split1.split(".")[0]
    split3 = split2.split("_")[-1]

    return int(split3)

def get_pngs_path_from_folder(given_folder_path: str = None) -> list[str]:
    folder_path: str = str(given_folder_path)
    pngs_paths: list[str] = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                pngs_paths.append(os.path.join(root, file))

    return sorted(pngs_paths, key = sorting_by_page_number)

def get_dataset(debug: bool = False, dataframe: pd.DataFrame = None) -> list[list[dict]]:
    dataset: list[list[dict]] = []
    max_batch_threshold: int = 4

    # convert to dataframe
    df = dataframe
    dataframe = pd.DataFrame(df)

    # iterate over .xlsx for JSON and image folder paths
    for _, row in tqdm(dataframe.iterrows(), total = df.shape[0], desc = "Generowanie datasetu"):
        images_folder_path: str = row["Image folder path"]
        json_ground_path: str = row["JSON file path"]

        # check if document is less than 10 pages, else skipp
        if Utils.check_length_of_simple_file(image_folder_path = images_folder_path):
            pngs_paths: list[str] = get_pngs_path_from_folder(given_folder_path = images_folder_path)
            # getting subfolder name
            subfolder_name: str = os.path.basename(images_folder_path)

            for i in range(0, len(pngs_paths), max_batch_threshold):
                current_batch = pngs_paths[i:i+max_batch_threshold]
                current_message_content: list[dict] = []

                for current_image_path in current_batch:
                    root_image_path: str = os.path.relpath(current_image_path)
                    current_message_content.append({
                        "type": "image",
                        "image": root_image_path
                    })

                current_message_content.append({
                    "type": "text",
                    "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Leave only generated structure, which will be dumped in the future"
                })

                message = [
                    {
                        "role": "user",
                        "content": current_message_content
                    },
                ]

                # dataset.append((message, subfolder_name, json_ground_path))
                dataset.append({
                    "messages": message,
                    "subfolder_name": subfolder_name,
                    "json_ground_path": json_ground_path
                })

                if debug:
                    print(f"Dataset: {dataset}")
                    print(type(dataset))
                    print(len(dataset))
        else:
            continue

    return dataset

def split_datasets():
    if not os.path.exists("test_csv/"):
        os.makedirs("test_csv/")

    df = read_excel()

    # split the dataset into training and validation
    train_df, tmp_df = train_test_split(df, test_size = 0.4, random_state = 42)

    # take 50% of the tmp_df for validation and test
    val_df, test_df = train_test_split(tmp_df, test_size = 0.5, random_state = 42)
    test_df.to_csv("test_csv/test_set.csv", index=False)

    # create dataset from the split
    train_dataset = get_dataset(dataframe = train_df)
    val_dataset = get_dataset(dataframe = val_df)
    test_df = get_dataset(dataframe = test_df)

    return train_dataset, val_dataset, test_df

df_result = process_years_to_excel([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], "matched_dates_cleaned_version2.xlsx")

model_id = "Qwen/Qwen2-VL-7B-Instruct"
# model_id = "Qwen/Qwen2-VL-7B-Instruct-AWQ" 
 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map = "auto",
    # quantization_config=bnb_config,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
)

def collate_fn(examples, debug: bool = False):
    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir="/net/scratch/hscra/plgrid/plgkruczek/.cache",
        min_pixels=128*28*28,
        max_pixels=256*28*28,
    )

    merged_texts = []
    image_inputs_list = []
    prompt_lengths = []

    for example in examples:
        message = example["messages"]
        json_ground_path = example["json_ground_path"]

        try:
            with open(json_ground_path, "r", encoding="utf-8") as f:
                json_str = f.read()
        except Exception as e:
            json_str = ""
            print(f"Error loading json file: {json_ground_path}: {e}", flush=True)

        prompt_str = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_tokens = processor.tokenizer(prompt_str, return_tensors="pt")
        prompt_length = prompt_tokens.input_ids.size(1)
        prompt_lengths.append(prompt_length)

        merged_text = prompt_str + "\n" + json_str
        merged_texts.append(merged_text)

        image_inputs, _ = process_vision_info(message)
        image_inputs_list.append(image_inputs)

    batch = processor(
        text=merged_texts,
        images=image_inputs_list,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    # model_device = next(model.parameters()).device
    # batch = {k: v.to(model_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # for key, value in batch.items():
    #     if isinstance(value, torch.Tensor):
    #         batch[key] = value.to(model.device)

    if debug:
        print(batch)
        print(type(batch))
        for key, value in batch.items():
            print(f"Key: {key}, Type: {type(value)}, Tensor dtype: {value.dtype}, Shape: {value.shape}")

    for key in batch:
        if key == 'pixel_values': 
            batch[key] = batch[key].to(torch.float16)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    potential_image_token_ids = [151652, 151653, 151655]
    for image_token_id in potential_image_token_ids:
        labels[labels == image_token_id] = -100

    for i, prompt_len in enumerate(prompt_lengths):
        seq_len = labels.size(1)
        if prompt_len < seq_len:
            labels[i, :prompt_len] = -100
        else:
            labels[i, :] = -100

    batch["labels"] = labels

    if debug:
        for key, value in batch.items():
            print(f"Key: {key}, Type: {type(value)}, Tensor dtype: {value.dtype}, Shape: {value.shape}")

    return batch

processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = 'right'
# all_special_tokens_from_ground_truth_dataset: list[str] = JsonHandler.get_special_tokens_json_ground_truth()
# num_added_tokens = processor.tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens_from_ground_truth_dataset})
# print(f"Added {num_added_tokens} special tokens.")
model.config.use_cache = False
 
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", 
)
 
args = SFTConfig(
    output_dir = f"Checkpoints/{model_id}_{mytime}",
    num_train_epochs = 15,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    gradient_checkpointing = True,
    optim = "adamw_torch_fused",
    logging_strategy = "epoch",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 2e-4,
    bf16 = False,
    tf32 = False,
    fp16 = True,
    use_cpu = False,
    max_grad_norm = 0.3,
    warmup_ratio = 0.03,
    lr_scheduler_type = "linear",
    push_to_hub = False,   
    dataset_kwargs = {"skip_prepare_dataset": True},
    load_best_model_at_end = True,
    dataloader_pin_memory = False,
)
args.remove_unused_columns=False
args.ddp_find_unused_parameters = False

datacollator = DataSets(model = model)

train_set, valid_set, test_set = split_datasets()
collator = collate_fn

# debug datasets
# print(f"Train set: {train_set}", flush = True)
print(f"Len train set: {len(train_set)}", flush = True)
# print(f"Valid set: {valid_set}", flush = True)
print(f"Len valid set: {len(valid_set)}", flush = True)
# print(f"Test set: {test_set}", flush = True)
print(f"Len test set: {len(test_set)}", flush = True)
 
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

print("=== Memory usage after loading model ===")
for i in range(torch.cuda.device_count()):
    device_name = torch.cuda.get_device_name(i)
    allocated = torch.cuda.memory_allocated(i) / (1024**3)
    reserved  = torch.cuda.memory_reserved(i)  / (1024**3)
    print(f"GPU {i} - {device_name}: "
          f"{allocated:.2f} GB allocated | "
          f"{reserved:.2f} GB reserved")

trainer.train()

log_history = trainer.state.log_history

safe_model_id = model_id.replace('/', '_')

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

epochs_train = range(1, len(train_losses) + 1)
epochs_eval = range(1, len(eval_losses) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_train, train_losses, label='Train Loss', color='blue', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.legend()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
train_filename = f"train_loss_{timestamp}_{safe_model_id}_{mytime}.png"
train_filepath = os.path.join(train_folder, train_filename)
plt.savefig(train_filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"Wykres strat treningowych zapisano w: {train_filepath}")

plt.figure(figsize=(8, 6))
plt.plot(epochs_eval, eval_losses, label='Eval Loss', color='red', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss per Epoch')
plt.grid(True)
plt.legend()

eval_filename = f"eval_loss_{timestamp}_{safe_model_id}_{mytime}.png"
eval_filepath = os.path.join(eval_folder, eval_filename)
plt.savefig(eval_filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"Wykres strat walidacyjnych zapisano w: {eval_filepath}")
