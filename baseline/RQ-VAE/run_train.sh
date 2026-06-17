#!/bin/bash
# ============================================================
# TIGER 多卡训练启动脚本
# 用法:
#   bash run_train.sh          # 使用默认配置 (2卡, GPU 0,1)
#   bash run_train.sh 4        # 使用 4 张卡 (GPU 0,1,2,3)
#   bash run_train.sh 4 "0,2,4,6"  # 用 4 张卡，指定 GPU ID
#   bash run_train.sh 1        # 单卡训练
# ============================================================

set -e

# ---- 可调参数 ----
NUM_GPUS=${1:-2}                    # 默认 2 卡，可通过第一个参数覆盖
VISIBLE_DEVICES=${2:-"0,1"}         # 默认用 GPU 0,1，可通过第二个参数覆盖
TRAIN_SCRIPT="train_tiger_mul.py"   # 训练脚本名

# ---- 打印当前配置 ----
echo "=========================================="
echo "  TIGER 多卡训练启动器"
echo "=========================================="
echo "  训练脚本:     ${TRAIN_SCRIPT}"
echo "  使用 GPU 数量: ${NUM_GPUS}"
echo "  使用 GPU ID:   ${VISIBLE_DEVICES}"
echo "=========================================="
echo ""

# ---- 设置可见 GPU ----
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}

# ---- 启动分布式训练 ----
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --num_machines=1 \
    --mixed_precision=no \
    --dynamo_backend=no \
    ${TRAIN_SCRIPT}
