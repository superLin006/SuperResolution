#!/bin/bash
# 将 TensorFlow 模型转换为 MTK DLA 格式的脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/../pyenv/venv"
MODELS_DIR="$PROJECT_DIR/data/models"
ALGORITHMS_DIR="$PROJECT_DIR/algorithms"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查虚拟环境
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    print_info "虚拟环境已激活"
else
    print_error "虚拟环境不存在: $VENV_DIR"
    exit 1
fi

# 检查 mlkits 是否可用
print_info "检查 mlkits 是否已安装..."
python3 -c "from mlkits import api" 2>/dev/null || {
    print_error "mlkits 未安装或不可用"
    print_info "请安装 mlkits: pip install mlkits-8.6.4+apu7.apu8.2521.2-*.whl"
    exit 1
}
print_info "mlkits 检查通过"

# 解析命令行参数
ALGORITHM=""
OUTPUT_DIR="$PROJECT_DIR/data/models"
SCALE=""
PLATFORM="MT8371"
NEURON_SDK_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --neuron_sdk_path)
            NEURON_SDK_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --algorithm <edsr|rcan|all>  要转换的算法 (默认: all)"
            echo "  --output_dir <路径>          输出目录 (默认: data/models)"
            echo "  --scale <2|4>                 超分辨率倍数 (可选，从配置读取)"
            echo "  --platform <平台>             目标平台 (MT6989, MT6991, MT6899, MT8371, 默认: MT6989)"
            echo "  --neuron_sdk_path <路径>      Neuron SDK 路径 (可选，自动检测)"
            echo "  --help, -h                    显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --algorithm edsr"
            echo "  $0 --algorithm rcan --scale 2 --platform MT6991"
            echo "  $0 --algorithm all --output_dir ./converted_models --platform MT6989"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 默认值
if [ -z "$ALGORITHM" ]; then
    ALGORITHM="all"
fi

print_info "=========================================="
print_info "TensorFlow 模型转 DLA 格式"
print_info "=========================================="
print_info "算法: $ALGORITHM"
print_info "模型目录: $MODELS_DIR"
print_info "输出目录: $OUTPUT_DIR"
print_info "目标平台: $PLATFORM"
if [ -n "$NEURON_SDK_PATH" ]; then
    print_info "Neuron SDK: $NEURON_SDK_PATH"
fi
print_info "=========================================="
echo ""

# 转换 EDSR 模型
convert_edsr() {
    local model_dir="$MODELS_DIR/edsr"
    local weights_file="$model_dir/model.weights.h5"
    local config_file="$model_dir/config.json"
    local mlkits_config="$ALGORITHMS_DIR/edsr/config.yaml"
    local output_dir="$OUTPUT_DIR/edsr/dla"
    
    print_info "=========================================="
    print_info "转换 EDSR 模型"
    print_info "=========================================="
    
    # 检查文件
    if [ ! -f "$weights_file" ]; then
        print_error "权重文件不存在: $weights_file"
        return 1
    fi
    
    if [ ! -f "$config_file" ]; then
        print_error "配置文件不存在: $config_file"
        return 1
    fi
    
    if [ ! -f "$mlkits_config" ]; then
        print_error "mlkits 配置文件不存在: $mlkits_config"
        return 1
    fi
    
    # 读取模型参数
    print_info "读取模型配置..."
    local filters=$(python3 -c "import json; config=json.load(open('$config_file')); print(config.get('n_feats', 64))")
    local num_blocks=$(python3 -c "import json; config=json.load(open('$config_file')); print(config.get('n_resblocks', 16))")
    local model_scale=$(python3 -c "import json; config=json.load(open('$config_file')); print(config.get('upscale', 4))")
    
    # 如果命令行指定了 scale，使用命令行参数
    if [ -n "$SCALE" ]; then
        model_scale="$SCALE"
    fi
    
    print_info "模型参数:"
    print_info "  filters: $filters"
    print_info "  num_blocks: $num_blocks"
    print_info "  scale: $model_scale"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 执行转换
    print_info "开始转换..."
    cd "$ALGORITHMS_DIR/edsr"
    
    # 构建命令参数
    CONVERT_CMD="python3 convert.py \
        --model_path \"$weights_file\" \
        --config \"$mlkits_config\" \
        --output_dir \"$output_dir\" \
        --filters \"$filters\" \
        --num_blocks \"$num_blocks\" \
        --scale \"$model_scale\" \
        --platform \"$PLATFORM\""
    
    if [ -n "$NEURON_SDK_PATH" ]; then
        CONVERT_CMD="$CONVERT_CMD --neuron_sdk_path \"$NEURON_SDK_PATH\""
    fi
    
    eval $CONVERT_CMD || {
        print_error "EDSR 模型转换失败"
        return 1
    }
    
    print_info "✅ EDSR 模型转换完成"
    print_info "输出目录: $output_dir"
    
    # 检查输出文件
    if [ -d "$output_dir" ]; then
        print_info "生成的文件:"
        ls -lh "$output_dir" | tail -n +2 || true
    fi
    
    return 0
}

# 转换 RCAN 模型
convert_rcan() {
    local model_dir="$MODELS_DIR/rcan"
    local weights_file="$model_dir/model.weights.h5"
    local config_file="$model_dir/config.json"
    local mlkits_config="$ALGORITHMS_DIR/rcan/config.yaml"
    local output_dir="$OUTPUT_DIR/rcan/dla"
    
    print_info "=========================================="
    print_info "转换 RCAN 模型"
    print_info "=========================================="
    
    # 检查文件
    if [ ! -f "$weights_file" ]; then
        print_error "权重文件不存在: $weights_file"
        return 1
    fi
    
    if [ ! -f "$config_file" ]; then
        print_error "配置文件不存在: $config_file"
        return 1
    fi
    
    if [ ! -f "$mlkits_config" ]; then
        print_error "mlkits 配置文件不存在: $mlkits_config"
        return 1
    fi
    
    # 读取模型参数
    print_info "读取模型配置..."
    local channels=$(python3 -c "import json; config=json.load(open('$config_file')); print(config.get('n_feats', 64))")
    local num_groups=$(python3 -c "import json; config=json.load(open('$config_file')); print(config.get('n_resgroups', 10))")
    local num_blocks=$(python3 -c "import json; config=json.load(open('$config_file')); print(config.get('n_resblocks', 20))")
    local reduction=$(python3 -c "import json; config=json.load(open('$config_file')); print(config.get('reduction', 16))")
    
    # RCAN 配置中通常没有 scale，默认使用 2
    local model_scale=2
    if [ -n "$SCALE" ]; then
        model_scale="$SCALE"
    fi
    
    print_info "模型参数:"
    print_info "  channels: $channels"
    print_info "  num_groups: $num_groups"
    print_info "  num_blocks: $num_blocks"
    print_info "  reduction: $reduction"
    print_info "  scale: $model_scale"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 执行转换
    print_info "开始转换..."
    cd "$ALGORITHMS_DIR/rcan"
    
    # 构建命令参数
    CONVERT_CMD="python3 convert.py \
        --model_path \"$weights_file\" \
        --config \"$mlkits_config\" \
        --output_dir \"$output_dir\" \
        --channels \"$channels\" \
        --num_groups \"$num_groups\" \
        --num_blocks \"$num_blocks\" \
        --scale \"$model_scale\" \
        --platform \"$PLATFORM\""
    
    if [ -n "$NEURON_SDK_PATH" ]; then
        CONVERT_CMD="$CONVERT_CMD --neuron_sdk_path \"$NEURON_SDK_PATH\""
    fi
    
    eval $CONVERT_CMD || {
        print_error "RCAN 模型转换失败"
        return 1
    }
    
    print_info "✅ RCAN 模型转换完成"
    print_info "输出目录: $output_dir"
    
    # 检查输出文件
    if [ -d "$output_dir" ]; then
        print_info "生成的文件:"
        ls -lh "$output_dir" | tail -n +2 || true
    fi
    
    return 0
}

# 执行转换
SUCCESS_COUNT=0
FAIL_COUNT=0

if [ "$ALGORITHM" = "edsr" ] || [ "$ALGORITHM" = "all" ]; then
    if convert_edsr; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    echo ""
fi

if [ "$ALGORITHM" = "rcan" ] || [ "$ALGORITHM" = "all" ]; then
    if convert_rcan; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    echo ""
fi

# 总结
print_info "=========================================="
print_info "转换总结"
print_info "=========================================="
print_info "成功: $SUCCESS_COUNT"
if [ $FAIL_COUNT -gt 0 ]; then
    print_error "失败: $FAIL_COUNT"
else
    print_info "失败: $FAIL_COUNT"
fi
print_info "=========================================="

if [ $FAIL_COUNT -eq 0 ]; then
    print_info "✅ 所有模型转换完成！"
    exit 0
else
    print_error "⚠️  部分模型转换失败"
    exit 1
fi
