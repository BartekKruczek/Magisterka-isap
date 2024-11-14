#!/bin/bash -l
#SBATCH --job-name=pelnamoc
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --account=plgexaile2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mail-user=bkruczekk@student.agh.edu.pl
#SBATCH --mail-type=BEGIN,END,FAIL

module load ML-bundle/24.06a
 
cd $SCRATCH

source .venv2/bin/activate
cd /net/storage/pr3/plgrid/plgglemkin/isap/Magisterka-isap

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py