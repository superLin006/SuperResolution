# 超分辨率平台测试程序

这是一个用于在Android设备上通过adb shell运行DLA模型进行图片超分辨率测试的控制台程序。

**项目仓库**: [https://github.com/iwwjlivecn/MTKSuperResolution](https://github.com/iwwjlivecn/MTKSuperResolution)

## 功能特性

- 加载DLA格式的模型文件
- 支持PNG/JPEG格式的输入图片
- 使用Neuron SDK Runtime API进行推理
- 输出超分辨率后的图片
- 显示推理时间和性能指标

## 项目结构要求

本测试程序需要与上层目录结构配合使用：

```
neuro/
├── MTKSuperResolution/         # 主项目
│   └── platform_test/          # 本目录
│       ├── sr_test.cpp        # 测试程序源码
│       ├── build.sh           # 编译脚本
│       ├── run_test.sh        # 运行测试脚本
│       └── README.md          # 本文档
└── neuropilot-sdk/             # NeuroPilot SDK（必需）
    └── neuron_sdk/             # Neuron SDK运行时库
        ├── mt6989/            # MT6989平台支持
        ├── mt6991/            # MT6991平台支持
        └── ...                # 其他平台
```

## 依赖项

### 必需依赖

1. **Neuron SDK Runtime API**
   - 需要在编译时链接Neuron SDK的运行时库
   - 头文件位置: `../neuropilot-sdk/neuron_sdk/<platform>/include/neuron/api/`
   - 库文件位置: `../neuropilot-sdk/neuron_sdk/<platform>/lib/`
   - 可通过环境变量 `NEURON_SDK_PATH` 指定路径（默认: `../neuropilot-sdk`）

2. **stb_image 库** (单文件头文件库)
   - `stb_image.h` - 用于读取图片
   - `stb_image_write.h` - 用于保存图片
   - 下载地址: https://github.com/nothings/stb
   - **注意**: `build.sh` 脚本会自动下载这些文件

3. **Android NDK** (用于交叉编译)
   - 用于编译ARM架构的可执行文件
   - 可通过环境变量 `NDK_ROOT` 指定路径
   - 下载地址: https://developer.android.com/ndk/downloads

### 获取stb库

stb库会在运行 `build.sh` 时自动下载。如果需要手动下载：

```bash
# 下载stb_image.h
curl -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

# 下载stb_image_write.h
curl -o stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

## 编译说明

### 推荐方法: 使用提供的编译脚本

项目提供了自动化编译脚本 `build.sh`，会自动处理所有依赖和编译步骤：

```bash
cd platform_test

# 设置环境变量（可选，脚本会使用默认值）
export NEURON_SDK_PATH=../../neuropilot-sdk  # 默认: ../neuropilot-sdk
export PLATFORM=mt6989                      # 默认: mt8371
export NDK_ROOT=~/android-ndk-r27d          # 默认: ~/android-ndk-r27d

# 运行编译脚本
bash build.sh
```

**编译脚本功能**：
- 自动下载stb库（如果不存在）
- 检查Neuron SDK路径和库文件
- 自动检测NDK编译器路径
- 支持多种NDK版本和平台
- 自动处理库依赖和链接

**查看帮助**：
```bash
bash build.sh --help
```

### 手动编译方法

如果需要手动编译，可以参考以下步骤：

#### 方法1: 使用Android NDK编译

```bash
# 设置环境变量
export NDK_ROOT=/path/to/android-ndk
export NEURON_SDK_PATH=../../neuropilot-sdk
export PLATFORM=mt6989  # 根据目标平台选择: mt6989, mt6991, mt8371等

# 查找NDK编译器
CLANG=$(find ${NDK_ROOT}/toolchains/llvm/prebuilt -name "aarch64-linux-android*-clang++" | head -1)

# 编译
${CLANG} \
    -std=c++11 \
    -I${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/include \
    -I. \
    -L${NEURON_SDK_PATH}/neuron_sdk/${PLATFORM}/lib \
    -lneuron_runtime \
    sr_test.cpp \
    -o build/sr_test \
    -llog -landroid -lc++_shared
```

#### 方法2: 使用CMake

项目已包含 `CMakeLists.txt`，可以使用CMake编译：

```bash
mkdir -p build
cd build
cmake .. \
    -DPLATFORM=mt6989 \
    -DNEURON_SDK_PATH=../../neuropilot-sdk
cmake --build .
```

**注意**: CMake方法需要Neuron SDK库文件与编译主机架构兼容。如果库文件是ARM架构，建议使用Android NDK交叉编译。

## 使用方法

### 推荐方法: 使用提供的运行脚本

项目提供了自动化运行脚本 `run_test.sh`，会自动处理文件推送、运行和结果拉取：

```bash
cd platform_test

# 基本用法（使用默认参数）
bash run_test.sh

# 指定算法和平台
bash run_test.sh --algorithm edsr --platform mt6989

# 指定输入图片
bash run_test.sh --algorithm rcan --platform mt6991 --input ../data/test_images/test.png

# 查看帮助
bash run_test.sh --help
```

**运行脚本功能**：
- 自动检查adb连接和设备状态
- 自动查找DLA模型文件
- 自动推送模型、图片、可执行文件和依赖库到设备
- 自动设置库路径并运行测试
- 自动拉取结果文件

### 手动运行方法

如果需要手动运行，可以按以下步骤操作：

#### 1. 准备文件

将以下文件推送到Android设备:

```bash
# 推送DLA模型文件（需要先转换模型）
adb push ../data/models/edsr/dla/model_mt6989.dla /data/local/tmp/

# 推送输入图片
adb push ../data/test_images/test_pattern.png /data/local/tmp/

# 推送编译好的可执行文件
adb push build/sr_test /data/local/tmp/
adb shell chmod +x /data/local/tmp/sr_test

# 推送Neuron SDK运行时库（必需）
adb shell mkdir -p /data/local/tmp/lib
adb push ../../neuropilot-sdk/neuron_sdk/mt6989/lib/*.so* /data/local/tmp/lib/
```

#### 2. 运行测试

```bash
# 通过adb shell运行（需要设置库路径）
adb shell "LD_LIBRARY_PATH=/data/local/tmp/lib:/system/lib64 /data/local/tmp/sr_test \
    --model /data/local/tmp/model_mt6989.dla \
    --input /data/local/tmp/test_pattern.png \
    --output /data/local/tmp/output.png"
```

#### 3. 获取结果

```bash
# 从设备拉取输出图片
adb pull /data/local/tmp/output.png ./
```

## 命令行参数

```
用法: sr_test [选项]

选项:
  --model <路径>        DLA模型文件路径 (必需)
  --input <路径>        输入图片路径 (必需)
  --output <路径>       输出图片路径 (必需)
  --help, -h            显示帮助信息
```

## 输出说明

程序会输出以下信息:

- 输入图片尺寸
- 模型输入输出信息（数量、大小等）
- 推理时间（毫秒）
- FPS（每秒帧数）
- 输出图片尺寸
- 输出文件路径

示例输出:

```
==========================================
超分辨率测试程序
==========================================
DLA模型路径: /data/local/tmp/model_mt6989.dla
输入图片: /data/local/tmp/test_pattern.png
输出图片: /data/local/tmp/output.png
==========================================
输入图片尺寸: 256x256 (通道数: 3)
Runtime创建成功
DLA模型加载成功
输入数量: 1
输出数量: 1
输入大小: 786432 字节
输出大小: 3145728 字节

开始推理...
推理完成，耗时: 45 ms
FPS: 22.22
输出尺寸: 512x512 (通道数: 3)

结果已保存到: /data/local/tmp/output.png

✅ 测试完成！
```

## 完整工作流程示例

以下是一个完整的工作流程，从模型转换到平台测试：

```bash
# 1. 激活Python虚拟环境
source ../../pyenv/venv/bin/activate

# 2. 转换模型为DLA格式（在项目根目录）
cd ../..
cd algorithms/edsr
python convert.py \
    --model_path ../../data/models/edsr/model.weights.h5 \
    --config config.yaml \
    --output_dir ../../results/edsr/converted \
    --filters 256 \
    --num_blocks 32 \
    --scale 2

# 3. 编译测试程序
cd ../../platform_test
export NEURON_SDK_PATH=../../neuropilot-sdk
export PLATFORM=mt6989
bash build.sh

# 4. 运行测试（假设DLA模型已生成在 data/models/edsr/dla/）
bash run_test.sh --algorithm edsr --platform mt6989

# 5. 查看结果
ls -lh output_*.png
```

## 注意事项

1. **项目目录结构**: 确保项目目录结构与文档中描述的一致，特别是 `neuropilot-sdk` 目录的位置

2. **平台匹配**: 确保DLA模型文件与目标平台匹配（例如MT6989平台需要使用MT6989编译的模型）

3. **库路径**: 在设备上运行前，确保Neuron SDK的运行时库在系统库路径中，或使用`LD_LIBRARY_PATH`环境变量。`run_test.sh` 脚本会自动处理

4. **权限**: 确保程序有权限访问模型文件和读写输入输出文件

5. **输入格式**: 
   - 输入图片应该是RGB格式
   - 图片会被归一化到[0,1]范围
   - 模型期望输入格式: `[1, H, W, 3]` float32

6. **输出格式**: 
   - 输出图片是RGB格式
   - 输出值会被裁剪到[0,1]范围并转换为uint8保存

7. **Android NDK版本**: 建议使用NDK r21-r27版本，较新版本可能需要调整编译参数

## 环境变量说明

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `NEURON_SDK_PATH` | NeuroPilot SDK路径 | `../neuropilot-sdk` |
| `PLATFORM` | 目标平台（mt6989/mt6991/mt8371等） | `mt8371` |
| `NDK_ROOT` | Android NDK根目录 | `~/android-ndk-r27d` |

## 故障排除

### 编译问题

**问题**: 找不到NDK编译器
- **解决**: 设置正确的 `NDK_ROOT` 环境变量，或使用 `--help` 查看脚本支持的路径格式

**问题**: 链接失败，找不到 `libneuron_runtime.so`
- **解决**: 检查 `NEURON_SDK_PATH` 是否正确，确认 `neuron_sdk/<platform>/lib/` 目录存在

**问题**: stb库下载失败
- **解决**: 手动下载并放置到 `platform_test/` 目录，或检查网络连接

### 运行问题

**问题**: adb设备未连接
- **解决**: 确保设备已通过USB连接，启用USB调试，运行 `adb devices` 确认

**问题**: 运行时找不到库文件
- **解决**: `run_test.sh` 会自动推送库文件。如果手动运行，确保设置了 `LD_LIBRARY_PATH`

**问题**: 模型加载失败
- **解决**: 检查模型文件路径，确认模型与平台匹配，检查文件权限

### 模型加载失败

- 检查DLA文件路径是否正确
- 确认模型文件与平台匹配
- 检查文件权限

### 推理失败

- 检查输入图片尺寸是否符合模型要求
- 确认输入数据格式正确（RGB, float32, 归一化到[0,1]）
- 查看设备日志: `adb logcat | grep neuron`

### 链接错误

- 确认Neuron SDK库文件存在
- 检查库文件版本是否匹配
- 确认所有依赖库都已链接

## 参考资料

- [项目主README](../README.md)
- [Neuron SDK文档](../../neuropilot-sdk/)
- [stb库文档](https://github.com/nothings/stb)
- [Android NDK文档](https://developer.android.com/ndk/guides)
