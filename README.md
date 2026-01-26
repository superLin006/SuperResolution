# SuperResolution

EDSR和RCAN超分辨率模型实现，支持MTK NPU和RKNN平台。

## 项目结构

```
SuperResolution/
├── edsr/                          # EDSR超分辨率
│   ├── mtk/                       # MTK NPU实现
│   │   ├── python/                # Python模型转换
│   │   ├── cpp/                   # C++推理实现
│   │   └── models/                # DLA模型文件（使用Git LFS）
│   └── rknn/                      # RKNN实现
│       ├── python/                # Python模型转换
│       └── cpp/                   # C++推理实现
├── rcan/                         # RCAN超分辨率
│   ├── mtk/                       # MTK NPU实现
│   │   ├── python/                # Python模型转换
│   │   ├── cpp/                   # C++推理实现
│   │   └── models/                # DLA模型文件（使用Git LFS）
│   └── rknn/                      # RKNN实现
│       ├── python/                # Python模型转换
│       └── cpp/                   # C++推理实现
└── data/                          # 数据和预训练模型
    ├── models/                    # 预训练模型
    └── test_images/               # 测试图像
```

## 支持的平台

- **MTK NPU**: MT8371, MT8189 (MDLA 5.3)
- **RKNN**: RK3588, RK3566, RK3568

## 模型说明

### EDSR (Enhanced Deep Residual Networks)
- **输入**: 256x256 RGB
- **输出**: 1024x1024 RGB (4x超分)
- **论文**: [EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

### RCAN (Residual Channel Attention Networks)
- **输入**: 510x339 RGB
- **输出**: 2040x1356 RGB (4x超分)
- **论文**: [RCAN: Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758)

## 快速开始

### MTK平台

```bash
# EDSR
cd edsr/mtk/cpp
./build.sh
./deploy_with_sdk_lib.sh --test

# RCAN
cd rcan/mtk/cpp
./build.sh
./deploy_with_sdk_lib.sh --test
```

### RKNN平台

```bash
# EDSR
cd edsr/rknn/python
python convert.py

# RCAN
cd rcan/rknn/python
python convert.py
```

## 性能

### MTK MT8189平台

| 模型 | 输入 | 输出 | 推理时间 | FPS |
|------|------|------|----------|-----|
| EDSR | 256x256 | 1024x1024 | ~7000ms | 0.14 |
| RCAN | 510x339 | 2040x1356 | ~4000ms | 0.25 |

## 依赖

### Python端
```bash
pip install torch torchvision numpy pillow mtk-converter rknn-toolkit2
```

### C++端
- Android NDK r25c+
- MTK NeuroPilot SDK 8.0.10
- RKNN Toolkit2

## 参考资料

- [MTK NeuroPilot SDK](https://github.com/MediaTek-NeuronPilot)
- [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)

## License

MIT License
