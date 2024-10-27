#!/bin/bash -l
#SBATCH --job-name=72B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200

module load ML-bundle/24.06a
 
cd $SCRATCH

source .venv2/bin/activate
cd /net/storage/pr3/plgrid/plgglemkin/isap/Magisterka-isap

python test_awq.py