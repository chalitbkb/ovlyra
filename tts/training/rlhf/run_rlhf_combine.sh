#!/bin/bash
#
# Script launcher for A100 nodes.
#
# Example:
#     sbatch --nodes=2 --gpus-per-node=8 --cpus-per-task=114 --partition=compute ./tts/training/rlhf/run_rlhf_combine.sh
#
# Nodes selection can be done via running `sinfo -lN`
# and then specifying the node names directly by adding
# `--nodelist=<node1>,<node2>` to the sbatch command above.
#
# Check your task via `squeue -u $USER` and cancel it via `scancel <job_id>` if needed.
#
# Set script flags down below manually as you need.
#
#SBATCH --job-name=rlhf_training
#SBATCH --nodes=2  # adjust as needed
#SBATCH --ntasks-per-node=1
#SBATCH --output=./rlhf_experiments/slurm_%j/log.log
#SBATCH --exclusive


ROOT_DIR=/home/$USER/tts
cd $ROOT_DIR
source .venv/bin/activate

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign the first 1 node for training and the 2nd node for vLLM
TRAIN_NODE="${NODELIST[0]}"  # Node 0 for training
VLLM_NODE="${NODELIST[1]}"  # Node 1 for vLLM

export WANDB_PROJECT="RLHF_Training"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Increase memory fraction allowed for CUDA operations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# The base model for vLLM, use the same checkpoint in the training config: "rlhf_training".base_model_dir.
VLLM_MODEL_PATH="/path/to/your/serving/model"

srun --nodes=1 --ntasks-per-node=1 --gpus-per-task=$SLURM_GPUS_PER_NODE --nodelist=$TRAIN_NODE \
    accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=$SLURM_GPUS_PER_NODE \
    --machine_rank=${SLURM_PROCID} \
    --main_process_ip=$TRAIN_NODE \
    --rdzv_backend=c10d \
    $ROOT_DIR/tts/training/rlhf/rlhf_main.py \
    --use_wandb \
    --config_path=$ROOT_DIR/example/configs/rlhf.json \
    --run_name=rlhf_combine_$SLURM_JOB_ID \
    --vllm_server_host=$VLLM_NODE &

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE trl vllm-serve --model $VLLM_MODEL_PATH --tensor_parallel_size 8 --max_model_len 3072 &
wait
