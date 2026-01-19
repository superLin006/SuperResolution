#!/bin/bash
# 运行所有算法测试的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/../pyenv/venv"

echo "=========================================="
echo "运行EDSR和RCAN算法测试"
echo "=========================================="
echo ""

# 激活虚拟环境
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "虚拟环境已激活"
else
    echo "错误: 虚拟环境不存在，请先运行: bash ../pyenv/install.sh"
    exit 1
fi

# 创建测试图像（如果不存在）
if [ ! -d "$PROJECT_DIR/data/test_images" ] || [ -z "$(ls -A $PROJECT_DIR/data/test_images 2>/dev/null)" ]; then
    echo "创建测试图像..."
    python "$SCRIPT_DIR/create_test_images.py" "$PROJECT_DIR/data/test_images"
fi

# 创建结果目录
mkdir -p "$PROJECT_DIR/results/edsr"
mkdir -p "$PROJECT_DIR/results/rcan"

echo ""
echo "=========================================="
echo "测试EDSR算法"
echo "=========================================="
cd "$PROJECT_DIR/algorithms/edsr"
python test_local.py \
    --test_image_dir ../../data/test_images \
    --output_dir ../../results/edsr \
    --scale 2 \
    --filters 64 \
    --num_blocks 16

echo ""
echo "=========================================="
echo "测试RCAN算法"
echo "=========================================="
cd "$PROJECT_DIR/algorithms/rcan"
python test_local.py \
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
echo "结果保存在:"
echo "  - EDSR: $PROJECT_DIR/results/edsr/"
echo "  - RCAN: $PROJECT_DIR/results/rcan/"
echo ""
