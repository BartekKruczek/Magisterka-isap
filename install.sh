#!/bin/bash -l
#SBATCH --job-name=pelnamoc
#SBATCH --time=03:00:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1

module load ML-bundle/24.06a
 
cd $SCRATCH

source .venv/bin/activate
pip install cupy
pip list