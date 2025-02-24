#!/bin/bash -l
#SBATCH --job-name=x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=00:30:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mail-user=bkruczekk@student.agh.edu.pl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1

module load ML-bundle/24.06a
 
cd $SCRATCH

source .venv/bin/activate
cd /net/storage/pr3/plgrid/plgglemkin/isap/Magisterka-isap

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export VLLM_USE_V1=1

python src/merge_models.py