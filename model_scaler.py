import os
import torch
import pandas as pd

from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor
from metrics import CustomMetrics
from DataCollator import DataSets

base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
checkpoint_folder = "qwen2-output/checkpoint-90"
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
merged_model.eval()

processor = AutoProcessor.from_pretrained(base_model_id)

test_df = pd.read_csv("test_csv/test_set.csv")

datacollator = DataSets(excel_file_path="matched_dates_cleaned_version2.xlsx")
test_set = datacollator.get_dataset(debug=False, dataframe=test_df)

custom_metrics = CustomMetrics()
final_ted_acc = custom_metrics.evaluate_ted_accuracy_on_testset(
    test_set=test_set,
    model=merged_model,
    processor=processor,
    custom_metrics=custom_metrics,
    debug=True,
)

print(f"[TEST] Średni TED-based Accuracy: {final_ted_acc:.4f}")