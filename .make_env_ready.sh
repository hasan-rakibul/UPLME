#!/bin/bash

# ENV_PATH=$MYSOFTWARE/.venv/noisempathy/bin/activate
# source $ENV_PATH
# echo "Activated $ENV_PATH"

read -p "Enter the GPU ID(s) to use: " GPU_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES is set."

export TOKENIZERS_PARALLELISM=false
echo "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM is set."

# export TMPDIR=/tmp/$SLURM_JOB_ID # as recommended in https://pawsey.atlassian.net/wiki/spaces/US/pages/51931230/PyTorch
# echo "TMPDIR=$TMPDIR is set."
