#!/bin/bash
#SBATCH --gres=gpu:2       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-23:59
#SBATCH --output=%N-%j.out

# Variables
export OMP_NUM_THREADS=4
export MAKEFLAGS="-j$(nproc)"

# Modules
module load cuda/11.0
module load python/3.6

# GET ENVIRONMENT
source ~/venv/bin/activate

# Start task
CONFIG_FILE=config_mask_rcnn_swinT.py
WORK_DIR=~/projects/def-philg/spaceboy/work_dirs/mask_RCNN_swinT_no_kpt
GPUS=2

# On Narval
./tools/dist_train.sh ${CONFIG_FILE} ${GPUS} --work-dir ${WORK_DIR} --cfg-options cudnn_benchmark=True evaluation.gpu_collect=True

