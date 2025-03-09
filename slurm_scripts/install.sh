#!/bin/bash -l
#SBATCH --job-name=x
#SBATCH --time=03:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1

module load ML-bundle/24.06a
 
cd $SCRATCH/envs
export MAX_JOBS=24

# python -m venv .venv
source .venv/bin/activate
# pip install -r requirements3.txt --no-build-isolation

# pip install setuptools_scm
# git clone https://github.com/vllm-project/vllm.git
# cd vllm
# python use_existing_torch.py
# pip install -r requirements-build.txt -v
# pip install -e . --no-build-isolation -v

pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

pip list