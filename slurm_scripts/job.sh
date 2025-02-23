#!/bin/bash -l
#SBATCH --job-name=x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=14:00:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4

module load ML-bundle/24.06a
 
cd $SCRATCH

source .venv/bin/activate
cd /net/storage/pr3/plgrid/plgglemkin/isap/Magisterka-isap

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

python src/custom_loop.py