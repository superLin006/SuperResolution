#!/bin/bash
# 编译脚本 - 用于在Android设备上运行的超分辨率测试程序

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NEURON_SDK_PATH="${NEURON_SDK_PATH:-$PROJECT_DIR/../neuropilot-sdk}"
PLATFORM="${PLATFORM:-mt8371}"
# 默认NDK路径
NDK_ROOT="${NDK_ROOT:-${HOME}/android-ndk-r27d}"
BUILD_DIR="${SCRIPT_DIR}/build"

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

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查stb库
    if [ ! -f "${SCRIPT_DIR}/stb_image.h" ]; then
        print_warn "stb_image.h 未找到，正在下载..."
        curl -s -o "${SCRIPT_DIR}/stb_image.h" \
            https://raw.githubusercontent.com/nothings/stb/master/stb_image.h || {
            print_error "下载 stb_image.h 失败"
            return 1
        }
    fi
    
    if [ ! -f "${SCRIPT_DIR}/stb_image_write.h" ]; then
        print_warn "stb_image_write.h 未找到，正在下载..."
        curl -s -o "${SCRIPT_DIR}/stb_image_write.h" \
            https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h || {
            print_error "下载 stb_image_write.h 失败"
            return 1
        }
    fi
    
    # 检查Neuron SDK
    if [ ! -d "${NEURON_SDK_PATH}" ]; then
        print_error "Neuron SDK 未找到: ${NEURON_SDK_PATH}"
        print_info "请设置 NEURON_SDK_PATH 环境变量"
        return 1
    fi
    
    NEURON_INCLUDE="${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/include"
    if [ ! -d "${NEURON_INCLUDE}" ]; then
        print_error "Neuron SDK 头文件目录不存在: ${NEURON_INCLUDE}"
        return 1
    fi
    
    print_info "依赖检查通过"
}

# 检查库文件架构
check_library_arch() {
    local lib_file="${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/lib/libneuron_runtime.so.8"
    if [ -f "$lib_file" ]; then
        if command -v readelf &> /dev/null; then
            local arch=$(readelf -h "$lib_file" 2>/dev/null | grep -i "Machine" | awk '{print $2}')
            if [[ "$arch" =~ "ARM|AArch64" ]]; then
                print_info "检测到ARM架构的库文件 - 需要使用Android NDK交叉编译"
                return 0
            fi
        fi
    fi
    return 1
}

# 使用NDK编译
build_with_ndk() {
    if [ -z "$NDK_ROOT" ]; then
        print_error "NDK_ROOT 未设置"
        print_info "请设置NDK_ROOT环境变量，例如:"
        print_info "  export NDK_ROOT=/path/to/android-ndk"
        return 1
    fi
    
    print_info "使用Android NDK编译..."
    
    # 尝试不同的NDK路径结构（按常见顺序）
    CLANG=""
    # 首先尝试常见的路径模式
    for api_level in 21 30 33; do
        for os in linux-x86_64 darwin-x86_64 darwin-arm64; do
            clang_path="${NDK_ROOT}/toolchains/llvm/prebuilt/${os}/bin/aarch64-linux-android${api_level}-clang++"
            if [ -f "$clang_path" ]; then
                CLANG="$clang_path"
                break 2
            fi
        done
    done
    
    # 如果还没找到，尝试通配符搜索
    if [ -z "$CLANG" ]; then
        CLANG=$(find "${NDK_ROOT}/toolchains/llvm/prebuilt" -name "aarch64-linux-android*-clang++" 2>/dev/null | head -1)
    fi
    
    if [ -z "$CLANG" ] || [ ! -f "$CLANG" ]; then
        print_error "找不到 NDK 编译器"
        print_info "请检查NDK路径: ${NDK_ROOT}"
        print_info "期望的编译器路径: ${NDK_ROOT}/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android*-clang++"
        return 1
    fi
    
    print_info "使用编译器: ${CLANG}"
    
    mkdir -p "${BUILD_DIR}"
    
    NEURON_INCLUDE="${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/include"
    NEURON_LIB="${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/lib"
    
    print_info "编译中..."
    
    # 获取NDK的sysroot路径（用于查找系统库）
    PREBUILT_DIR=$(dirname $(dirname "$CLANG"))
    SYSROOT="${PREBUILT_DIR}/sysroot"
    if [ ! -d "$SYSROOT" ]; then
        # 尝试其他可能的路径
        SYSROOT=$(find "${NDK_ROOT}/toolchains/llvm/prebuilt" -type d -name "sysroot" 2>/dev/null | head -1)
    fi
    
    # 从编译器路径提取API级别
    API_LEVEL=$(echo "$CLANG" | grep -oE "android[0-9]+" | grep -oE "[0-9]+" || echo "21")
    
    # 构建编译命令
    # 注意：使用动态链接，允许共享库未定义符号（因为运行时会在设备上解析）
    LINK_CMD=(
        "$CLANG"
        -std=c++11
        --sysroot="${SYSROOT}"
        -I"${NEURON_INCLUDE}"
        -I"${SCRIPT_DIR}"
        -L"${NEURON_LIB}"
        "${SCRIPT_DIR}/sr_test.cpp"
        -o "${BUILD_DIR}/sr_test"
        -lneuron_runtime
        -llog
        -landroid
        -Wl,-rpath-link="${NEURON_LIB}"
        -Wl,--allow-shlib-undefined
    )
    
    # 添加系统库路径
    if [ -n "$SYSROOT" ] && [ -d "$SYSROOT" ]; then
        SYSROOT_LIB="${SYSROOT}/usr/lib/aarch64-linux-android/${API_LEVEL}"
        if [ -d "$SYSROOT_LIB" ]; then
            # 添加库搜索路径
            LINK_CMD+=(-L"${SYSROOT_LIB}")
            LINK_CMD+=(-Wl,-rpath-link="${SYSROOT_LIB}")
        fi
        # 也添加通用库路径
        SYSROOT_LIB_GENERAL="${SYSROOT}/usr/lib/aarch64-linux-android"
        if [ -d "$SYSROOT_LIB_GENERAL" ]; then
            LINK_CMD+=(-L"${SYSROOT_LIB_GENERAL}")
            LINK_CMD+=(-Wl,-rpath-link="${SYSROOT_LIB_GENERAL}")
        fi
    fi
    
    # C++ 标准库和系统库必须放在最后链接（链接顺序很重要）
    # 注意：Android NDK 的 libc++.so 是链接脚本，指向 libc++_shared
    # 但链接时应该使用 -lc++，链接器会自动处理；如果不行，尝试 -lc++_shared
    LINK_CMD+=(-lc++_shared -lm -ldl)
    
    print_info "使用API级别: ${API_LEVEL}"
    if [ -n "$SYSROOT" ]; then
        print_info "使用sysroot: ${SYSROOT}"
    fi
    
    # 执行编译
    "${LINK_CMD[@]}" || {
        print_error "编译失败"
        print_info "提示: 确保NDK版本正确，并且Neuron SDK库文件存在"
        print_info "尝试的链接命令:"
        print_info "  ${LINK_CMD[*]}"
        return 1
    }
    
    print_info "✅ 编译成功: ${BUILD_DIR}/sr_test"
    
    # 验证输出文件
    if [ -f "${BUILD_DIR}/sr_test" ]; then
        if command -v file &> /dev/null; then
            print_info "输出文件信息:"
            file "${BUILD_DIR}/sr_test"
        fi
    fi
}

# 使用CMake编译
build_with_cmake() {
    print_info "使用CMake编译..."
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake "${SCRIPT_DIR}" \
        -DPLATFORM="${PLATFORM}" \
        -DNEURON_SDK_PATH="${NEURON_SDK_PATH}" \
        -DCMAKE_BUILD_TYPE=Release || {
        print_error "CMake配置失败"
        return 1
    }
    
    cmake --build . || {
        print_error "编译失败"
        return 1
    }
    
    print_info "✅ 编译成功: ${BUILD_DIR}/sr_test"
}

# 主函数
main() {
    print_info "=========================================="
    print_info "超分辨率测试程序编译脚本"
    print_info "=========================================="
    print_info "平台: ${PLATFORM}"
    print_info "Neuron SDK: ${NEURON_SDK_PATH}"
    print_info "NDK: ${NDK_ROOT}"
    if [ ! -d "$NDK_ROOT" ]; then
        print_warn "NDK目录不存在: ${NDK_ROOT}"
        print_info "请设置正确的NDK_ROOT环境变量"
    fi
    print_info "=========================================="
    echo ""
    
    check_dependencies || exit 1
    
    # 检查库文件架构
    if check_library_arch; then
        if [ ! -d "$NDK_ROOT" ]; then
            print_warn "Neuron SDK库是为ARM架构编译的，需要使用Android NDK交叉编译"
            print_error "NDK目录不存在: ${NDK_ROOT}"
            print_info ""
            print_info "请设置正确的NDK_ROOT环境变量:"
            print_info "  export NDK_ROOT=/path/to/android-ndk"
            print_info "  $0"
            print_info ""
            print_info "或者下载Android NDK:"
            print_info "  https://developer.android.com/ndk/downloads"
            exit 1
        fi
    fi
    
    # 如果NDK目录存在，优先使用NDK编译
    if [ -d "$NDK_ROOT" ]; then
        build_with_ndk
    else
        if command -v cmake &> /dev/null; then
            print_warn "NDK目录不存在，尝试使用CMake编译"
            print_warn "注意: 如果库文件是ARM架构，链接可能会失败"
            print_warn "建议使用Android NDK: export NDK_ROOT=/path/to/android-ndk"
            build_with_cmake
        else
            print_error "未找到 CMake，且NDK目录不存在: ${NDK_ROOT}"
            print_error "请安装 CMake 或设置正确的 NDK_ROOT 环境变量"
            exit 1
        fi
    fi
    
    print_info ""
    print_info "编译完成！"
    print_info "可执行文件位置: ${BUILD_DIR}/sr_test"
}

# 显示帮助
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "用法: $0 [选项]"
    echo ""
    echo "环境变量:"
    echo "  NEURON_SDK_PATH   Neuron SDK 路径 (默认: ../neuropilot-sdk)"
    echo "  PLATFORM          目标平台 (默认: mt6989)"
    echo "  NDK_ROOT          Android NDK 路径 (默认: ~/android-ndk-r29)"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认NDK路径"
    echo "  PLATFORM=mt6991 $0                   # 指定平台"
    echo "  NDK_ROOT=/path/to/ndk $0             # 指定NDK路径"
    exit 0
fi

main
