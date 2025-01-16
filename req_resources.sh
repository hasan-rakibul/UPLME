#!/bin/bash

read -p "Which partition do you want to use? (gpu[default]/gpu-highmem/gpu-dev/cpu) " partition
partition=${partition:-gpu}

if [ "$partition" == "gpu" ] || [ "$partition" == "gpu-highmem" ]; then
    account="pawsey1001-gpu"
    max_walltime="24:00:00"
elif [ "$partition" == "gpu-dev" ]; then
    account="pawsey1001-gpu"
    max_walltime="4:00:00"
elif [ "$partition" == "cpu" ]; then
    read -p "How many CPUs per task do you need? " num_cpus
    account="pawsey1001"
    $max_walltime="24:00:00"
else
    echo "Invalid partition. Exiting."
    exit 1
fi

if [ "$partition" == "gpu" ] || [ "$partition" == "gpu-dev" ] || [ "$partition" == "gpu-highmem" ]; then
    read -p "How many GPUs do you need? (default: 8) " num_gpus
    num_gpus=${num_gpus:-8}
    salloc -N 1 --gres=gpu:$num_gpus -A $account --partition=$partition --time=$max_walltime
else
    salloc -N 1 --ntasks=1 --cpus-per-task=$num_cpus -A $account --time=$max_walltime
fi
