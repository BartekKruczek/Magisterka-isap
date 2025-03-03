#!/bin/bash -l
#SBATCH --job-name=x
#SBATCH --time=05:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1

module load ML-bundle/24.06a
 
cd $SCRATCH
export MAX_JOBS=2

source .venv_abcd/bin/activate
cd vllm
python use_existing_torch.py
pip install -r requirements-build.txt -v
pip install -e . --no-build-isolation -v
pip list