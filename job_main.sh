#!/bin/bash
 
#SBATCH --job-name=PairedText
#SBATCH --output=outputs/log_slurm/%j_%x.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

# EXPT_FILE_NAME="cross-basic-betn-text"
# EXPT_FILE_NAME="test_cross-basic-betn-text"

# EXPT_FILE_NAME="single-cross-prob"
# EXPT_FILE_NAME="tune_single-cross-prob"

# EXPT_FILE_NAME="two-cross-prob"
EXPT_FILE_NAME="tune_two-cross-prob"

# EXPT_FILE_NAME="best-newsemp-pcc_single-cross-prob"
# EXPT_FILE_NAME="best-newsemp_single-cross-prob"
# EXPT_FILE_NAME="best-newsemp_two-cross-prob"
# EXPT_FILE_NAME="test_noise-uncertainty"

singularity exec $SINGULARITY_CONTAINER bash -c "\
source $MYSOFTWARE/.venv/bin/activate && \
export TOKENIZERS_PARALLELISM=false && \
python src/main.py \
expt=$EXPT_FILE_NAME"
