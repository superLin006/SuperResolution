#!/bin/bash
# 使用预训练模型（如果有）进行测试的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/../pyenv/venv"
MODELS_DIR="$PROJECT_DIR/data/models"

echo "=========================================="
echo "使用预训练模型测试算法"
echo "=========================================="
echo ""

# 激活虚拟环境
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "虚拟环境已激活"
else
    echo "错误: 虚拟环境不存在"
    exit 1
fi

# 检查模型目录
if [ ! -d "$MODELS_DIR" ]; then
    echo "模型目录不存在，运行下载脚本..."
    python "$SCRIPT_DIR/download_models.py" --models_dir "$MODELS_DIR"
fi

echo ""
echo "=========================================="
echo "测试EDSR算法"
echo "=========================================="

# 查找EDSR模型
EDSR_MODEL=""
if [ -d "$MODELS_DIR/edsr" ]; then
    # 查找可能的模型文件（排除文本文件）
    EDSR_MODEL=$(find "$MODELS_DIR/edsr" \( -name "*.h5" -o -name "*.ckpt*" -o -name "*.pb" -o -name "saved_model" -type d \) -type f 2>/dev/null | head -1)
    # 如果没有找到文件，尝试查找目录
    if [ -z "$EDSR_MODEL" ]; then
        EDSR_MODEL=$(find "$MODELS_DIR/edsr" -name "saved_model" -type d | head -1)
    fi
fi

if [ -n "$EDSR_MODEL" ] && [ -f "$EDSR_MODEL" ]; then
    echo "找到EDSR模型: $EDSR_MODEL"
    MODEL_ARG="--model_path $EDSR_MODEL"
else
    echo "未找到EDSR预训练模型，使用随机权重测试"
    MODEL_ARG=""
fi

cd "$PROJECT_DIR/algorithms/edsr"
python test_local.py \
    $MODEL_ARG \
    --test_image_dir ../../data/test_images \
    --output_dir ../../results/edsr \
    --scale 2 \
    --filters 64 \
    --num_blocks 16

echo ""
echo "=========================================="
echo "测试RCAN算法"
echo "=========================================="

# 查找RCAN模型
RCAN_MODEL=""
if [ -d "$MODELS_DIR/rcan" ]; then
    # 查找可能的模型文件（排除文本文件）
    RCAN_MODEL=$(find "$MODELS_DIR/rcan" \( -name "*.h5" -o -name "*.ckpt*" -o -name "*.pb" -o -name "saved_model" -type d \) -type f 2>/dev/null | head -1)
    # 如果没有找到文件，尝试查找目录
    if [ -z "$RCAN_MODEL" ]; then
        RCAN_MODEL=$(find "$MODELS_DIR/rcan" -name "saved_model" -type d | head -1)
    fi
fi

if [ -n "$RCAN_MODEL" ] && [ -f "$RCAN_MODEL" ]; then
    echo "找到RCAN模型: $RCAN_MODEL"
    MODEL_ARG="--model_path $RCAN_MODEL"
else
    echo "未找到RCAN预训练模型，使用随机权重测试"
    MODEL_ARG=""
fi

cd "$PROJECT_DIR/algorithms/rcan"
python test_local.py \
    $MODEL_ARG \
    --test_image_dir ../../data/test_images \
    --output_dir ../../results/rcan \
    --scale 2 \
    --channels 64 \
    --num_groups 5 \
    --num_blocks 10

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
