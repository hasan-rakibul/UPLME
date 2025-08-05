#!/bin/bash
 
# EXPT_FILE_NAME="train-plm-mlp"
# EXPT_FILE_NAME="test-plm-mlp"

# EXPT_FILE_NAME="train-uplme"
# EXPT_FILE_NAME="test-uplme"
# EXPT_FILE_NAME="tune_uplme"

EXPT_FILE_NAME="uplme-best-newsemp"
# EXPT_FILE_NAME="uplme-best-empstories"

# EXPT_FILE_NAME="test-newsemp22"

# EXPT_FILE_NAME="ucvme"

export TOKENIZERS_PARALLELISM=false
python src/main.py expt=$EXPT_FILE_NAME
