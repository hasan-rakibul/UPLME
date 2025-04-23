#!/bin/bash
 
#SBATCH --job-name=PairedText
#SBATCH --output=outputs/log_slurm/%j_%x.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

singularity exec $SINGULARITY_CONTAINER bash -c "\
source $MYSOFTWARE/.venv/bin/activate && \
export TOKENIZERS_PARALLELISM=false && \
python src/main.py \
approach='cross-prob' \
is_ssl=True"
