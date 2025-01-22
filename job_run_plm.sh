#!/bin/bash
 
#SBATCH --job-name=RunPLM
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

export TOKENIZERS_PARALLELISM=false
python src/run_plm.py