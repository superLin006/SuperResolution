# AI超分辨率算法移植到MTK平台项目

本项目实现了EDSR和RCAN两个超分辨率算法的移植，支持使用NeuroPilot SDK转换为MTK平台可用的格式。

**项目仓库**: [https://github.com/iwwjlivecn/MTKSuperResolution](https://github.com/iwwjlivecn/MTKSuperResolution)

## 项目结构

本项目需要与上层的 `neuropilot-sdk` 和 `pyenv` 目录配合使用。完整的项目目录结构如下：

```
neuro/                          # 项目根目录
├── MTKSuperResolution/         # 本项目（主仓库）
│   ├── algorithms/             # 算法实现
│   │   ├── edsr/              # EDSR算法
│   │   │   ├── model.py       # 模型定义
│   │   │   ├── test_local.py  # 本地测试脚本
│   │   │   ├── convert.py     # 模型转换脚本
│   │   │   └── config.yaml    # mlkits配置
│   │   └── rcan/               # RCAN算法
│   │       ├── model.py       # 模型定义
│   │       ├── test_local.py  # 本地测试脚本
│   │       ├── convert.py     # 模型转换脚本
│   │       └── config.yaml    # mlkits配置
│   ├── scripts/                # 工具脚本
│   │   ├── download_models.sh  # 模型下载脚本
│   │   ├── test_platform.sh    # 平台测试脚本
│   │   └── test_platform_mt6989.py  # 平台测试Python脚本
│   ├── platform_test/          # 平台测试程序（C++）
│   │   ├── sr_test.cpp        # 测试程序源码
│   │   ├── build.sh           # 编译脚本
│   │   ├── run_test.sh        # 运行测试脚本
│   │   └── README.md          # 测试程序说明文档
│   ├── data/                   # 数据目录
│   │   ├── test_images/        # 测试图像
│   │   └── models/             # 预训练模型
│   └── results/                 # 结果目录
│       ├── edsr/               # EDSR结果
│       └── rcan/               # RCAN结果
├── neuropilot-sdk/             # NeuroPilot SDK（必需，需单独获取）
│   ├── neuron_sdk/             # Neuron SDK运行时库
│   │   ├── mt6989/            # MT6989平台支持
│   │   ├── mt6991/            # MT6991平台支持
│   │   └── ...                # 其他平台
│   └── offline_tool/           # 离线转换工具
│       ├── mlkits-*.whl       # mlkits wheel文件
│       └── mtk_converter-*.whl # mtk_converter wheel文件
└── pyenv/                      # Python虚拟环境（包含依赖和构建脚本）
    ├── venv/                   # Python虚拟环境目录
    ├── install.sh              # 环境安装脚本
    └── requirements.txt        # Python依赖列表
```

**重要说明**：
- `neuropilot-sdk` 目录需要单独获取，包含Neuron SDK运行时库和离线转换工具
- `pyenv` 目录包含Python虚拟环境构建脚本和依赖管理
- 本项目（`MTKSuperResolution`）是主仓库，可以独立克隆

## 环境要求

- Python 3.6+ (推荐 3.9)
- TensorFlow 2.x
- NeuroPilot SDK 8.0.10+
- mlkits (从neuropilot-sdk/offline_tool安装)
- Android NDK (用于编译platform_test，可选)
- 其他依赖: numpy, Pillow, scikit-image (可选，用于评估指标)

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/iwwjlivecn/MTKSuperResolution.git
cd MTKSuperResolution
```

### 2. 准备依赖目录

确保项目目录结构与上述结构一致。`neuropilot-sdk` 需要单独获取并放置在项目上层目录：

```
neuro/
├── MTKSuperResolution/  # 本项目
├── neuropilot-sdk/      # 需要单独获取
└── pyenv/              # 将在此目录创建
```

### 3. 构建Python虚拟环境（pyenv）

使用提供的安装脚本自动构建Python环境：

```bash
# 方法1: 使用系统Python（如果已安装Python 3.9）
cd ../pyenv
bash install.sh

# 方法2: 使用pyenv安装指定Python版本（推荐）
bash install.sh --use-pyenv --python-version 3.9.18
```

安装脚本会自动：
- 安装/配置pyenv（如果使用 `--use-pyenv`）
- 创建Python虚拟环境
- 安装所有Python依赖（TensorFlow、PyTorch等）
- 安装mlkits和mtk_converter（从neuropilot-sdk/offline_tool）

**详细说明**：
- 脚本会检测Python版本并选择匹配的mlkits wheel文件
- 支持Python 3.6、3.7、3.9
- 如果neuropilot-sdk目录存在，会自动安装mlkits相关依赖

### 4. 激活虚拟环境

```bash
source ../pyenv/venv/bin/activate
```

### 5. 验证安装

```bash
# 验证TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# 验证mlkits
python -c "from mlkits import api; print('mlkits可用')"
```

### 6. 准备测试数据

- 将测试图像放入 `data/test_images/` 目录
- 支持的格式: PNG, JPG

## 使用方法

### EDSR算法

#### 1. 本地测试

使用预训练模型（如果有）或随机初始化权重进行测试：

```bash
cd algorithms/edsr
python test_local.py \
    --model_path /path/to/model \
    --test_image_dir ../../data/test_images \
    --output_dir ../../results/edsr \
    --scale 2 \
    --filters 256 \
    --num_blocks 32
```

参数说明：
- `--model_path`: 模型权重路径（可选，如果不提供则使用随机权重）
- `--test_image_dir`: 测试图像目录
- `--output_dir`: 结果输出目录
- `--scale`: 超分辨率缩放因子（默认: 2）
- `--filters`: 滤波器数量（默认: 256）
- `--num_blocks`: 残差块数量（默认: 32）

#### 2. 模型转换

使用mlkits将模型转换为MT6989平台格式：

```bash
cd algorithms/edsr
python convert.py \
    --model_path /path/to/model \
    --config config.yaml \
    --output_dir ../../results/edsr/converted \
    --filters 256 \
    --num_blocks 32 \
    --scale 2
```

### RCAN算法

#### 1. 本地测试

```bash
cd algorithms/rcan
python test_local.py \
    --model_path /path/to/model \
    --test_image_dir ../../data/test_images \
    --output_dir ../../results/rcan \
    --scale 2 \
    --channels 64 \
    --num_groups 10 \
    --num_blocks 20
```

参数说明：
- `--model_path`: 模型权重路径（可选）
- `--test_image_dir`: 测试图像目录
- `--output_dir`: 结果输出目录
- `--scale`: 超分辨率缩放因子（默认: 2）
- `--channels`: 特征通道数（默认: 64）
- `--num_groups`: 残差组数量（默认: 10）
- `--num_blocks`: 每组残差块数量（默认: 20）

#### 2. 模型转换

```bash
cd algorithms/rcan
python convert.py \
    --model_path /path/to/model \
    --config config.yaml \
    --output_dir ../../results/rcan/converted \
    --channels 64 \
    --num_groups 10 \
    --num_blocks 20 \
    --scale 2
```

## 模型下载

预训练模型需要从官方仓库下载：

- **EDSR**: https://github.com/thstkdgus35/EDSR-PyTorch
- **RCAN**: https://github.com/yulunzhang/RCAN

注意：官方仓库提供的模型通常是PyTorch格式，如果需要TensorFlow格式，需要进行格式转换。

### 模型格式转换

项目提供了自动转换脚本，可以将PyTorch模型转换为TensorFlow格式：

```bash
# 检查模型状态
python scripts/check_pytorch_models.py

# 转换EDSR模型
python scripts/convert_pytorch_to_tensorflow.py \
    --model_dir data/models/edsr \
    --algorithm edsr

# 转换RCAN模型
python scripts/convert_pytorch_to_tensorflow.py \
    --model_dir data/models/rcan \
    --algorithm rcan
```

转换后的模型会保存在原模型目录下：
- `saved_model/`: TensorFlow SavedModel格式
- `model.weights.h5`: TensorFlow H5权重文件

可以使用提供的脚本查看下载说明：

```bash
bash scripts/download_models.sh
```

## 配置说明

### mlkits配置 (config.yaml)

主要配置项：

- `model.inputs`: 输入形状，例如 `[1, 256, 256, 3]` 表示批次大小为1，256x256像素，3通道
- `quant.quantizer.init_bitwidth`: 量化位数（8或16）
- `quant.input_ranges`: 输入数据范围

根据实际应用场景调整输入形状和量化参数。

## 平台测试

模型转换后，可以在MTK平台上进行测试。项目提供了两种测试方式：

### 方式1: Python测试脚本（用于验证转换后的模型）

```bash
python scripts/test_platform_mt6989.py \
    --model_path results/edsr/converted/saved_model \
    --test_image_dir data/test_images \
    --output_dir results/platform_test \
    --algorithm edsr
```

### 方式2: C++测试程序（在Android设备上运行，推荐）

`platform_test` 目录包含一个C++测试程序，可以在Android设备上直接运行DLA模型。

**详细使用方法请参考**: [platform_test/README.md](platform_test/README.md)

**快速开始**：

1. **编译测试程序**：
```bash
cd platform_test
# 设置环境变量
export NEURON_SDK_PATH=../../neuropilot-sdk
export PLATFORM=mt6989  # 或 mt6991, mt8371 等
export NDK_ROOT=/path/to/android-ndk  # 可选，如果未设置会尝试默认路径

# 编译
bash build.sh
```

2. **运行测试**：
```bash
# 使用提供的运行脚本（自动推送文件到设备并运行）
bash run_test.sh --algorithm edsr --platform mt6989

# 或手动运行
adb push build/sr_test /data/local/tmp/
adb push ../data/models/edsr/dla/model_mt6989.dla /data/local/tmp/
adb push ../data/test_images/test.png /data/local/tmp/
adb shell /data/local/tmp/sr_test \
    --model /data/local/tmp/model_mt6989.dla \
    --input /data/local/tmp/test.png \
    --output /data/local/tmp/output.png
```

**注意**：平台测试需要：
1. MTK开发板或设备（MT6989/MT6991/MT8371等）
2. Neuron SDK运行时库（在neuropilot-sdk/neuron_sdk/目录下）
3. Android设备已连接并启用USB调试
4. Android NDK（用于编译C++测试程序）

## 评估指标

测试脚本会计算以下指标：

- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，单位dB，值越大越好
- **SSIM** (Structural Similarity Index): 结构相似性指数，范围[0,1]，值越大越好

## 结果说明

测试结果保存在 `results/` 目录下：

- `sr_*.png`: 超分辨率结果图像
- `comparison_*.png`: 对比图像（LR、SR、HR并排显示）

## 注意事项

1. **模型格式**: 如果使用PyTorch预训练模型，需要先转换为TensorFlow格式
   - 使用 `scripts/convert_pytorch_to_tensorflow.py` 进行自动转换
   - 使用 `scripts/check_pytorch_models.py` 检查转换状态
2. **输入形状**: 转换前确保模型输入形状与config.yaml中配置一致
3. **依赖安装**: 转换需要PyTorch支持，已在 `pyenv/requirements.txt` 中添加
3. **量化**: 量化可能会影响模型精度，需要根据实际需求调整量化参数
4. **算子兼容性**: 确保使用的TensorFlow算子被mlkits支持

## 参考资源

- EDSR论文: "Enhanced Deep Residual Networks for Single Image Super-Resolution"
- RCAN论文: "Image Super-Resolution Using Very Deep Residual Channel Attention Networks"
- NeuroPilot SDK文档: 参考neuropilot-sdk目录下的文档
- EDSR代码: https://github.com/thstkdgus35/EDSR-PyTorch
- RCAN代码: https://github.com/yulunzhang/RCAN
