#!/bin/bash -l
#SBATCH --job-name=x
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1

module load ML-bundle/24.06a
 
cd $SCRATCH
export MAX_JOBS=16

source .venv/bin/activate
cd triton
pip install ninja cmake wheel pybind11 -v
export PYTHONHTTPSVERIFY=0
pip install -e python -v --trusted-host developer.download.nvidia.com
pip list