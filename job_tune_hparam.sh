#!/bin/bash
 
#SBATCH --job-name=OptunaTune
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu
#SBATCH --ntasks-per-node=4

export TOKENIZERS_PARALLELISM=false

python src/tune_hparam.py \
--n_trials 20 \
--expt_name "tune" \
--resume_dir "log/20250202_230203_single-prob_(2024,2022)-tune"
