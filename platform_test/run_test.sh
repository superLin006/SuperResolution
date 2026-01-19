#!/bin/bash
# 运行测试脚本 - 在Android设备上运行超分辨率测试

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/data/models"
TEST_IMAGE_DIR="${PROJECT_DIR}/data/test_images"
NEURON_SDK_PATH="${NEURON_SDK_PATH:-$PROJECT_DIR/../neuropilot-sdk}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 解析参数
ALGORITHM="edsr"
PLATFORM="mt8371"
INPUT_IMAGE=""
OUTPUT_DIR="/data/local/tmp"

while [[ $# -gt 0 ]]; do
    case $1 in
        --algorithm)
            ALGORITHM="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --input)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --algorithm <edsr|rcan>  算法名称 (默认: edsr)"
            echo "  --platform <平台>         目标平台 (默认: mt6989)"
            echo "  --input <路径>           输入图片路径 (可选，默认使用测试图片)"
            echo "  --output_dir <路径>      设备上的输出目录 (默认: /data/local/tmp)"
            echo ""
            echo "环境变量:"
            echo "  NEURON_SDK_PATH          Neuron SDK路径 (默认: ../neuropilot-sdk)"
            echo "  --help, -h               显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --algorithm edsr --platform mt6989"
            echo "  $0 --algorithm rcan --input /data/local/tmp/test.png"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

print_info "=========================================="
print_info "超分辨率平台测试"
print_info "=========================================="
print_info "算法: ${ALGORITHM}"
print_info "平台: ${PLATFORM}"
print_info "Neuron SDK: ${NEURON_SDK_PATH}"
print_info "=========================================="
echo ""

# 检查adb
if ! command -v adb &> /dev/null; then
    print_error "adb 未找到，请安装 Android SDK Platform Tools"
    exit 1
fi

# 检查设备连接
if ! adb devices | grep -q "device$"; then
    print_error "未找到已连接的Android设备"
    print_info "请确保设备已通过USB连接并启用USB调试"
    exit 1
fi

print_info "检测到Android设备"

# 查找DLA模型文件
MODEL_FILE="${MODEL_DIR}/${ALGORITHM}/dla/model_${PLATFORM}.dla"
if [ ! -f "${MODEL_FILE}" ]; then
    print_error "DLA模型文件不存在: ${MODEL_FILE}"
    print_info "请先运行 convert.sh 转换模型"
    exit 1
fi

print_info "找到模型文件: ${MODEL_FILE}"

# 确定输入图片
if [ -z "${INPUT_IMAGE}" ]; then
    # 使用默认测试图片
    TEST_IMAGES=($(find "${TEST_IMAGE_DIR}" -name "*.png" -o -name "*.jpg" 2>/dev/null | head -1))
    if [ ${#TEST_IMAGES[@]} -eq 0 ]; then
        print_error "未找到测试图片，请使用 --input 指定输入图片"
        exit 1
    fi
    INPUT_IMAGE="${TEST_IMAGES[0]}"
fi

if [ ! -f "${INPUT_IMAGE}" ]; then
    print_error "输入图片不存在: ${INPUT_IMAGE}"
    exit 1
fi

print_info "使用输入图片: ${INPUT_IMAGE}"

# 准备设备上的路径
DEVICE_MODEL_PATH="${OUTPUT_DIR}/model_${PLATFORM}.dla"
DEVICE_INPUT_PATH="${OUTPUT_DIR}/input_$(basename "${INPUT_IMAGE}")"
DEVICE_OUTPUT_PATH="${OUTPUT_DIR}/output_$(basename "${INPUT_IMAGE}")"
DEVICE_EXEC_PATH="${OUTPUT_DIR}/sr_test"
DEVICE_LIB_DIR="${OUTPUT_DIR}/lib"

# 检查可执行文件是否存在
if [ ! -f "${SCRIPT_DIR}/build/sr_test" ]; then
    print_warn "可执行文件不存在: ${SCRIPT_DIR}/build/sr_test"
    print_info "请先运行 build.sh 编译程序"
    exit 1
fi

# 检查Neuron SDK库目录
NEURON_LIB_DIR="${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/lib"
if [ ! -d "${NEURON_LIB_DIR}" ]; then
    print_error "Neuron SDK库目录不存在: ${NEURON_LIB_DIR}"
    print_info "请检查NEURON_SDK_PATH环境变量或确保Neuron SDK已正确安装"
    exit 1
fi

# 推送文件到设备
print_info "推送文件到设备..."

# 推送模型文件
adb push "${MODEL_FILE}" "${DEVICE_MODEL_PATH}" > /dev/null
print_info "  ✓ 模型文件"

# 推送输入图片
adb push "${INPUT_IMAGE}" "${DEVICE_INPUT_PATH}" > /dev/null
print_info "  ✓ 输入图片"

# 推送可执行文件
adb push "${SCRIPT_DIR}/build/sr_test" "${DEVICE_EXEC_PATH}" > /dev/null
adb shell chmod +x "${DEVICE_EXEC_PATH}" > /dev/null
print_info "  ✓ 可执行文件"

# 推送依赖库
print_info "推送Neuron SDK依赖库..."
adb shell mkdir -p "${DEVICE_LIB_DIR}" > /dev/null 2>&1

# 推送所有.so库文件
LIB_COUNT=0
for lib_file in "${NEURON_LIB_DIR}"/*.so*; do
    if [ -f "$lib_file" ]; then
        lib_name=$(basename "$lib_file")
        adb push "$lib_file" "${DEVICE_LIB_DIR}/${lib_name}" > /dev/null
        ((LIB_COUNT++))
    fi
done

if [ $LIB_COUNT -gt 0 ]; then
    print_info "  ✓ 已推送 ${LIB_COUNT} 个库文件到 ${DEVICE_LIB_DIR}"
else
    print_warn "  未找到库文件，可能无法运行"
fi

print_info "文件推送完成"

# 运行测试
print_info ""
print_info "开始在设备上运行测试..."
print_info ""

# 设置LD_LIBRARY_PATH以包含依赖库目录
adb shell "LD_LIBRARY_PATH=${DEVICE_LIB_DIR}:${LD_LIBRARY_PATH:-/system/lib64} ${DEVICE_EXEC_PATH}" \
    --model "${DEVICE_MODEL_PATH}" \
    --input "${DEVICE_INPUT_PATH}" \
    --output "${DEVICE_OUTPUT_PATH}"

RET=$?

if [ $RET -eq 0 ]; then
    print_info ""
    print_info "✅ 测试成功！"
    
    # 拉取结果
    LOCAL_OUTPUT="${SCRIPT_DIR}/output_$(basename "${INPUT_IMAGE}")"
    print_info "拉取结果到: ${LOCAL_OUTPUT}"
    adb pull "${DEVICE_OUTPUT_PATH}" "${LOCAL_OUTPUT}" > /dev/null
    
    print_info ""
    print_info "输出文件: ${LOCAL_OUTPUT}"
else
    print_error "测试失败 (退出码: $RET)"
    exit 1
fi
