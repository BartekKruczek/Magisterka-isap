import pandas as pd

from transformers import AutoProcessor
from vllm import LLM
from multiprocessing import freeze_support

from metrics import CustomMetrics
from custom_datasets import CustomDataSets
from plot_results import PlotResults

def get_processor():
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        min_pixels=128*28*28,
        max_pixels=256*28*28,
        use_fast=True,
        )
    return processor

def main(processor):
    path = "/net/scratch/hscra/plgrid/plgkruczek/Saved_models/2_5_72B"
    vLLM_model = LLM(
        model=path,
        max_num_seqs=8192,
        max_num_batched_tokens=65536,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        dtype="float16",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 10},
        enforce_eager=False,
        enable_prefix_caching=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.95,
    )

    test_df = pd.read_csv("/net/scratch/hscra/plgrid/plgkruczek/Saved_models/2_5_72B/test_set.csv")

    custom_set = CustomDataSets()
    test_set = custom_set.get_dataset(debug=False, dataframe=test_df)

    custom_metrics = CustomMetrics()
    plot = PlotResults()
    artefact_pct, valid_pct, avg_lev_dist, pages_lev_map = custom_metrics.evaluate_on_testset(
        test_set,
        vLLM_model,
        processor, 
        model_fix=vLLM_model,
        processor_fix=processor,
        do_auto_fix=True,
        use_xgrammar=False,
        do_normalize_jsons=True,
        debug=True,
    )

    print(f"Percentage of all artefacts detected: {artefact_pct}")
    print(f"Valid json files after cleaning: {valid_pct}")
    print(f"Average lev dist: {avg_lev_dist}")
    print(f"Pages - Lev map: {pages_lev_map}")

    plot.plot_average_Lev(lev_dict=pages_lev_map)

if __name__ == '__main__':
    freeze_support()
    processor = get_processor()
    main(processor)