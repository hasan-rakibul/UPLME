#!/bin/bash
 
#SBATCH --job-name=RunPLM
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

export TOKENIZERS_PARALLELISM=false
python src/main_bi_encoder.py \
--approach="single-prob" \
--expt_name_postfix="tune-first" \
--tune_hparams
