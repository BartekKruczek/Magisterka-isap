import torch
import pandas as pd

from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM

from metrics import CustomMetrics
from custom_datasets import CustomDataSets
from plot_results import PlotResults

base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
checkpoint_folder = "Checkpoints/20250223-081257"
cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
model_fix_name: str = "Qwen/Qwen2.5-72B-Instruct"

base_model = AutoModelForVision2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=cache_dir,
    attn_implementation="flash_attention_2",
)

model_fix = AutoModelForCausalLM.from_pretrained(
    model_fix_name,
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
processor_fix = AutoProcessor.from_pretrained(model_fix_name)

test_df = pd.read_csv("Checkpoints/20250223-081257/test_set.csv")

custom_set = CustomDataSets()
test_set = custom_set.get_dataset(debug=False, dataframe=test_df)

custom_metrics = CustomMetrics()
plot = PlotResults()
artefact_pct, valid_pct, avg_lev_dist, pages_lev_map = custom_metrics.evaluate_on_testset(
    base_model_id,
    test_set,
    merged_model, 
    processor,
    model_fix,
    processor_fix,
    do_auto_fix=False,
    use_xgrammar=False,
    do_normalize_jsons=True,
    debug=False,
)

print(f"Percentage of all artefacts detected: {artefact_pct}")
print(f"Valid json files after cleaning: {valid_pct}")
print(f"Average lev dist: {avg_lev_dist}")
print(f"Pages - Lev map: {pages_lev_map}")

plot.plot_average_Lev(lev_dict=pages_lev_map)