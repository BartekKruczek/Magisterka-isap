#!/bin/bash -l
#SBATCH --job-name=pelnamoc
#SBATCH --time=00:15:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1

module load ML-bundle/24.06a
 
cd $SCRATCH

python -m venv .venv_testowe
source .venv_testowe/bin/activate
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip list