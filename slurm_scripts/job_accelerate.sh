#!/bin/bash -l
#SBATCH --job-name=x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mail-user=bkruczekk@student.agh.edu.pl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:4

module load ML-bundle/24.06a
 
cd $SCRATCH

source .venv/bin/activate
cd /net/storage/pr3/plgrid/plgglemkin/isap/Magisterka-isap

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export OMP_NUM_THREADS=288

# accelerate launch \
#     --config_file=deepspeed_zero3.yaml \
#     test_from_Internet.py \
#     --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
#     --model_name_or_path llava-hf/llava-1.5-7b-hf \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 8 \
#     --output_dir sft-llava-1.5-7b-hf \
#     --bf16 \
#     --torch_dtype bfloat16 \
#     --gradient_checkpointing

accelerate launch \
    --config_file=deepspeed_zero3.yaml \
    test_from_Internet.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --num_processes=4 \

    # --bf16 \
    # --torch_dtype bfloat16 \