#!/bin/bash
# 在 WSL 中运行：打包所有需要的文件到 /mnt/d/work，并生成 Windows 端 sr_test.cmd

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 配置
ALGORITHM="${ALGORITHM:-edsr}"        # 或 rcan
PLATFORM="${PLATFORM:-mt8371}"
NEURON_SDK_PATH="${NEURON_SDK_PATH:-$PROJECT_DIR/../neuropilot-sdk}"

# Windows 映射目录，在 WSL 中对应 /mnt/d/work
WIN_ROOT="/mnt/d/work"
PKG_DIR="${WIN_ROOT}/sr_test_pkg"
PAYLOAD="${WIN_ROOT}/sr_test_payload.tar.gz"
CMD_FILE="${WIN_ROOT}/sr_test.cmd"

MODEL_FILE="${PROJECT_DIR}/data/models/${ALGORITHM}/dla/model_${PLATFORM}.dla"
TEST_IMAGE_DIR="${PROJECT_DIR}/data/test_images"
NEURON_LIB_DIR="${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/lib"
BUILD_BIN="${SCRIPT_DIR}/build/sr_test"

# 简单输出函数
green() { echo -e "\033[0;32m[INFO]\033[0m $*"; }
red()   { echo -e "\033[0;31m[ERR ]\033[0m $*"; }
yellow() { echo -e "\033[1;33m[WARN]\033[0m $*"; }

# 检查路径
[ -d "$WIN_ROOT" ] || { red "目录不存在: $WIN_ROOT (确认 D:\\work 已创建)"; exit 1; }
[ -f "$BUILD_BIN" ] || { red "可执行文件不存在: $BUILD_BIN，请先 ./build.sh"; exit 1; }
[ -f "$MODEL_FILE" ] || { red "模型文件不存在: $MODEL_FILE，请先 ./scripts/convert.sh"; exit 1; }
[ -d "$NEURON_LIB_DIR" ] || { red "Neuron SDK 库目录不存在: $NEURON_LIB_DIR"; exit 1; }

# 选一张测试图片
TEST_IMG="$(find "$TEST_IMAGE_DIR" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | head -1)"
[ -n "$TEST_IMG" ] || { red "未在 $TEST_IMAGE_DIR 找到测试图片"; exit 1; }

green "使用测试图片: $TEST_IMG"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/libs"

# 拷贝文件到打包目录
cp "$BUILD_BIN"            "$PKG_DIR/sr_test"
cp "$MODEL_FILE"           "$PKG_DIR/model_${PLATFORM}.dla"
cp "$TEST_IMG"             "$PKG_DIR/input.png"
cp "$NEURON_LIB_DIR"/*.so* "$PKG_DIR/libs/" 2>/dev/null || true

# 尝试从 NDK 复制 libc++_shared.so（如果需要）
NDK_ROOT="${NDK_ROOT:-${HOME}/android-ndk-r27d}"
if [ -d "$NDK_ROOT" ]; then
    # 查找 libc++_shared.so（可能在多个位置）
    LIBCXX_SHARED=""
    for possible_path in \
        "${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android" \
        "${NDK_ROOT}/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android" \
        "${NDK_ROOT}/toolchains/llvm/prebuilt/darwin-arm64/sysroot/usr/lib/aarch64-linux-android"; do
        if [ -f "${possible_path}/libc++_shared.so" ]; then
            LIBCXX_SHARED="${possible_path}/libc++_shared.so"
            break
        fi
    done
    
    # 如果找不到，尝试搜索
    if [ -z "$LIBCXX_SHARED" ]; then
        LIBCXX_SHARED=$(find "$NDK_ROOT" -name "libc++_shared.so" -path "*/aarch64-linux-android/*" 2>/dev/null | head -1)
    fi
    
    if [ -n "$LIBCXX_SHARED" ] && [ -f "$LIBCXX_SHARED" ]; then
        cp "$LIBCXX_SHARED" "$PKG_DIR/libs/" && green "已复制 libc++_shared.so"
    else
        yellow "警告: 未找到 libc++_shared.so，运行时可能需要设备上已有该库"
    fi
else
    yellow "警告: NDK_ROOT 未设置或目录不存在，无法复制 libc++_shared.so"
fi

green "已拷贝可执行文件、模型、图片和依赖库到 $PKG_DIR"

# 生成 tar.gz 载荷
rm -f "$PAYLOAD"
(
  cd "$PKG_DIR"
  tar -czf "$PAYLOAD" .
)
green "已生成打包文件: $PAYLOAD"

# 生成 Windows 端的 sr_test.cmd
# 先写入临时文件，然后转换为 Windows 格式（CRLF + GBK编码）
TEMP_CMD=$(mktemp)
cat > "$TEMP_CMD" <<EOF
@echo off
REM 在 Windows 主机上运行此脚本，需要 adb 在 PATH 中

setlocal ENABLEDELAYEDEXPANSION

REM 当前脚本所在目录
set SCRIPT_DIR=%~dp0

REM 文件路径
set PAYLOAD=%SCRIPT_DIR%sr_test_payload.tar.gz
set DEVICE_BASE=/data/local/tmp/sr_test
set DEVICE_LIB=%DEVICE_BASE%/libs
set DEVICE_MODEL=%DEVICE_BASE%/model_${PLATFORM}.dla
set DEVICE_INPUT=%DEVICE_BASE%/input.png
set DEVICE_OUTPUT=%DEVICE_BASE%/output.png
set DEVICE_BIN=%DEVICE_BASE%/sr_test
EOF

# 继续写入脚本的其余部分
cat >> "$TEMP_CMD" <<'EOF'

echo ==========================================
echo Super-Resolution Platform Test (via adb)
echo ==========================================

REM 检查 adb
where adb >NUL 2>&1
if errorlevel 1 (
  echo [ERROR] adb 未找到，请确保已安装 Android Platform Tools 并加入 PATH
  goto :eof
)

REM 检查设备连接
for /f "skip=1 tokens=1,2" %%i in ('adb devices') do (
  if "%%j"=="device" (
    set DEVICE_FOUND=1
  )
)
if not defined DEVICE_FOUND (
  echo [ERROR] 未找到已连接的 Android 设备
  goto :eof
)

REM 创建设备目录
adb shell "rm -rf %DEVICE_BASE% && mkdir -p %DEVICE_BASE% && mkdir -p %DEVICE_LIB%" 

REM 推送打包文件
echo [INFO] 推送打包文件到设备...
adb push "%PAYLOAD%" /data/local/tmp/sr_test_payload.tar.gz

REM 在设备上解压
echo [INFO] 在设备上解压文件...
adb shell "cd %DEVICE_BASE% && tar -xzf /data/local/tmp/sr_test_payload.tar.gz"

REM 赋予执行权限
adb shell "chmod +x %DEVICE_BIN%"

REM 验证文件是否存在
echo [INFO] 验证设备上的文件...
adb shell "test -f %DEVICE_BIN% && test -f %DEVICE_MODEL% && test -f %DEVICE_INPUT% && echo 'Files OK' || echo 'Files missing'"
adb shell "ls -l %DEVICE_BASE%"

REM 运行测试
echo [INFO] 开始在设备上运行测试...
echo [INFO] 模型路径: %DEVICE_MODEL%
adb shell "cd %DEVICE_BASE% && LD_LIBRARY_PATH=%DEVICE_LIB%:/system/lib64 %DEVICE_BIN% --model %DEVICE_MODEL% --input %DEVICE_INPUT% --output %DEVICE_OUTPUT%"

if errorlevel 1 (
  echo [ERROR] 设备上测试失败
  echo.
  echo [DEBUG] 检查模型文件...
  adb shell "test -f %DEVICE_MODEL% && echo 'Model file exists' || echo 'Model file NOT found'"
  adb shell "ls -l %DEVICE_MODEL%"
  adb shell "file %DEVICE_MODEL% 2>/dev/null || echo 'file command not available'"
  echo.
  echo [DEBUG] 检查库文件...
  adb shell "ls -l %DEVICE_LIB%/"
  echo.
  echo [DEBUG] 检查可执行文件...
  adb shell "ls -l %DEVICE_BIN%"
  adb shell "file %DEVICE_BIN% 2>/dev/null || echo 'file command not available'"
  echo.
  echo [DEBUG] 检查输入图片...
  adb shell "ls -l %DEVICE_INPUT%"
  echo.
  echo [HINT] 错误代码 4 通常表示:
  echo   - 模型文件格式不正确或损坏
  echo   - 模型文件与平台不匹配
  echo   - 模型文件无法读取
  echo   请检查模型文件是否已正确转换为 DLA 格式
  goto :eof
)

REM 拉取结果到当前目录
echo [INFO] 拉取结果图片到当前目录...
adb pull %DEVICE_OUTPUT% "%SCRIPT_DIR%output.png"

echo.
echo [OK] 测试完成，结果已保存为: %SCRIPT_DIR%output.png

endlocal
EOF

# 转换为 Windows 格式：LF -> CRLF，并转换为 GBK 编码
if command -v unix2dos >/dev/null 2>&1; then
  # 如果有 unix2dos，先转换换行符，再转换编码
  unix2dos "$TEMP_CMD" 2>/dev/null || sed -i 's/$/\r/' "$TEMP_CMD"
else
  # 如果没有 unix2dos，使用 sed 添加 \r
  sed -i 's/$/\r/' "$TEMP_CMD"
fi

# 转换编码为 GBK（如果 iconv 可用），否则保持 UTF-8
if command -v iconv >/dev/null 2>&1; then
  iconv -f UTF-8 -t GBK "$TEMP_CMD" > "$CMD_FILE" 2>/dev/null || cp "$TEMP_CMD" "$CMD_FILE"
else
  cp "$TEMP_CMD" "$CMD_FILE"
fi

rm -f "$TEMP_CMD"

green "已生成 Windows 脚本: $CMD_FILE"

echo
green "下一步：在 Windows 中双击或运行:"
echo "  D:\\work\\sr_test.cmd"
echo "确保已连接设备且 adb 可用。"