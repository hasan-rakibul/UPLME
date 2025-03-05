#!/bin/bash

# ENV_PATH=$MYSOFTWARE/.venv/bin/activate
# source $ENV_PATH
# echo "Activated $ENV_PATH"

read -p "What is the GPU IDs for CUDA_VISIBLE_DEVICES or press enter to skip: " GPU_ID
if [ ! -z "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES is set."
fi

export TOKENIZERS_PARALLELISM=false
echo "TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM is set."

# export TMPDIR=/tmp/$SLURM_JOB_ID # as recommended in https://pawsey.atlassian.net/wiki/spaces/US/pages/51931230/PyTorch
# echo "TMPDIR=$TMPDIR is set."
