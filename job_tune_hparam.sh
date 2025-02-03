#!/bin/bash
 
#SBATCH --job-name=OptunaTune
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu
#SBATCH --ntasks-per-node=4

export TOKENIZERS_PARALLELISM=false

python src/tune_hparam.py \
--n_trials 10 \
--expt_name "no-sanitise-no-noise" \
--approach "single-prob" \
