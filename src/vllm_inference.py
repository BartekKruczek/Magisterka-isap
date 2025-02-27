import torch
import pandas as pd
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM
from vllm import LLM
from multiprocessing import freeze_support

from metrics import CustomMetrics
from custom_datasets import CustomDataSets
from plot_results import PlotResults

def main():
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
    model_fix_name: str = "Qwen/Qwen2.5-72B-Instruct"

    model_fix = AutoModelForCausalLM.from_pretrained(
        model_fix_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
        attn_implementation="flash_attention_2",
    )
    processor_fix = AutoProcessor.from_pretrained(model_fix_name, trust_remote_code=True)

    path = "Saved_models"
    vLLM_model = LLM(
        model=path,
        tensor_parallel_size=4,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 10},
    )

    test_df = pd.read_csv("Checkpoints/20250223-081257/test_set.csv")

    custom_set = CustomDataSets()
    test_set = custom_set.get_dataset(debug=False, dataframe=test_df)

    custom_metrics = CustomMetrics()
    plot = PlotResults()
    artefact_pct, valid_pct, avg_lev_dist, pages_lev_map = custom_metrics.evaluate_on_testset(
        test_set,
        vLLM_model, 
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

if __name__ == '__main__':
    freeze_support()
    main()