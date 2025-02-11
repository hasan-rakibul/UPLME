#!/bin/bash
 
#SBATCH --job-name=PairedText
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

export TOKENIZERS_PARALLELISM=false
python src/main.py \
--approach="single-prob" \
--expt_name_postfix="cross-encoder" \
--tune_hparams
