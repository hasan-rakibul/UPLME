#!/bin/bash
 
#SBATCH --job-name=UPLME
#SBATCH --output=outputs/log_slurm/%j_%x.log
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.7.1-rocm6.3.3

# EXPT_FILE_NAME="train-plm-mlp"
# EXPT_FILE_NAME="test-plm-mlp"

EXPT_FILE_NAME="train-uplme"
# EXPT_FILE_NAME="test-uplme"
# EXPT_FILE_NAME="tune_uplme"

# EXPT_FILE_NAME="uplme-best-newsemp"
# EXPT_FILE_NAME="uplme-best-empstories"

# EXPT_FILE_NAME="test-newsemp22"

# EXPT_FILE_NAME="ucvme"

singularity exec $SINGULARITY_CONTAINER bash -c "\
source $MYSOFTWARE/.venv/bin/activate && \
export TOKENIZERS_PARALLELISM=false && \
python src/main.py \
expt=$EXPT_FILE_NAME"
