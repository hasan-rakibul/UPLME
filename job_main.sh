#!/bin/bash
 
#SBATCH --job-name=PairedText
#SBATCH --output=outputs/log_slurm/%j_%x.log
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

# EXPT="ssl"
# EXPT="test_ssl"
# EXPT="cross-basic-betn-text"
# EXPT="test_cross-basic-betn-text"
# EXPT="two-cross-prob"
# EXPT="tune_two-cross-prob"
# EXPT="single-cross-prob"
# EXPT="best-newsemp-pcc_single-cross-prob"
# EXPT="best-newsemp_single-cross-prob"
EXPT="best-newsemp_two-cross-prob"
# EXPT="tune_single-cross-prob"
# EXPT="test_noise-uncertainty"

singularity exec $SINGULARITY_CONTAINER bash -c "\
source $MYSOFTWARE/.venv/bin/activate && \
export TOKENIZERS_PARALLELISM=false && \
python src/main.py \
expt=$EXPT"
