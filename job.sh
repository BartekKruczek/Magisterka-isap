#!/bin/bash -l
#SBATCH --job-name=pelnamoc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:2

module load ML-bundle/24.06a
 
cd $SCRATCH

source .venv/bin/activate
cd /net/storage/pr3/plgrid/plgglemkin/isap/Magisterka-isap

python main.py