"""测试 TensorFlow 模型文件（EDSR 和 RCAN）进行图片超分辨率测试"""

import os
import json
import pathlib
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

# 添加项目根目录到路径
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'algorithms' / 'edsr'))
sys.path.insert(0, str(project_root / 'algorithms' / 'rcan'))

from algorithms.edsr.model import EDSR
from algorithms.rcan.model import RCAN
from algorithms.edsr.utils import calc_psnr, calc_ssim


def load_image(image_path, scale=2):
    """加载并预处理图片
    
    Args:
        image_path: 输入图片路径
        scale: 下采样因子
    
    Returns:
        lr: 低分辨率图片 (归一化到 [0, 1])
        hr: 高分辨率图片 (归一化到 [0, 1])
    """
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    
    # 通过下采样创建 LR
    h, w = img.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    
    # 使用 PIL 进行下采样（模拟双三次下采样）
    lr_img = Image.fromarray((img * 255).astype(np.uint8))
    lr_img = lr_img.resize((lr_w, lr_h), Image.BICUBIC)
    lr = np.array(lr_img).astype(np.float32) / 255.0
    
    # 裁剪 HR 以匹配 scale
    hr_h, hr_w = lr_h * scale, lr_w * scale
    hr = img[:hr_h, :hr_w]
    
    return lr, hr


def load_config(config_path):
    """加载模型配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def test_edsr_model(model_dir, test_image_dir, output_dir, scale=None):
    """测试 EDSR 模型
    
    Args:
        model_dir: 模型目录路径
        test_image_dir: 测试图片目录
        output_dir: 输出目录
        scale: 超分辨率倍数，如果为 None 则从配置读取
    """
    model_dir = pathlib.Path(model_dir)
    weights_path = model_dir / 'model.weights.h5'
    config_path = model_dir / 'config.json'
    
    print(f"\n{'='*60}")
    print(f"测试 EDSR 模型")
    print(f"{'='*60}")
    print(f"模型目录: {model_dir.absolute()}")
    
    # 检查文件
    if not weights_path.exists():
        print(f"❌ 权重文件不存在: {weights_path}")
        return False
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    # 加载配置
    config = load_config(config_path)
    print(f"\n模型配置:")
    print(f"  n_feats: {config.get('n_feats', 64)}")
    print(f"  n_resblocks: {config.get('n_resblocks', 16)}")
    print(f"  upscale: {config.get('upscale', 4)}")
    
    # 创建模型
    filters = config.get('n_feats', 64)
    num_blocks = config.get('n_resblocks', 16)
    if scale is None:
        scale = config.get('upscale', 4)
    
    print(f"\n创建模型...")
    model = EDSR(filters=filters, num_blocks=num_blocks, scale=scale)
    
    # 使用虚拟输入构建模型
    dummy_input = tf.zeros([1, 64, 64, 3])
    _ = model(dummy_input)
    
    # 加载权重
    print(f"加载权重: {weights_path}")
    try:
        model.load_weights(str(weights_path))
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return False
    
    # 创建输出目录
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取测试图片
    test_image_dir = pathlib.Path(test_image_dir)
    if not test_image_dir.exists():
        print(f"❌ 测试图片目录不存在: {test_image_dir}")
        return False
    
    image_files = list(test_image_dir.glob('*.png')) + list(test_image_dir.glob('*.jpg')) + \
                  list(test_image_dir.glob('*.jpeg'))
    
    if not image_files:
        print(f"❌ 未找到测试图片")
        return False
    
    print(f"\n找到 {len(image_files)} 张测试图片")
    
    # 测试每张图片
    psnr_values = []
    ssim_values = []
    
    for img_path in image_files:
        print(f"\n处理: {img_path.name}")
        
        try:
            # 加载图片
            lr, hr = load_image(img_path, scale=scale)
            
            # 预处理（添加批次维度）
            lr_tensor = tf.expand_dims(tf.constant(lr), 0)
            
            # 推理
            sr_tensor = model(lr_tensor, training=False)
            sr = sr_tensor[0].numpy()
            
            # 计算指标
            psnr = calc_psnr(sr, hr)
            ssim = calc_ssim(sr, hr)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
            # 保存结果
            sr_img = (np.clip(sr, 0, 1) * 255).astype(np.uint8)
            sr_pil = Image.fromarray(sr_img)
            sr_pil.save(output_dir / f"edsr_sr_{img_path.name}")
            
            # 保存对比图（LR, SR, HR 并排）
            lr_up = (np.clip(tf.image.resize(lr, hr.shape[:2], method='bicubic').numpy(), 0, 1) * 255).astype(np.uint8)
            comparison = np.concatenate([lr_up, sr_img, (hr * 255).astype(np.uint8)], axis=1)
            Image.fromarray(comparison).save(output_dir / f"edsr_comparison_{img_path.name}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    # 打印总结
    if psnr_values:
        print(f"\n{'='*60}")
        print("EDSR 测试总结:")
        print(f"  平均 PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"  平均 SSIM: {np.mean(ssim_values):.4f}")
        print(f"  结果保存至: {output_dir}")
        print(f"{'='*60}")
        return True
    else:
        print("❌ 没有成功处理的图片")
        return False


def test_rcan_model(model_dir, test_image_dir, output_dir, scale=None):
    """测试 RCAN 模型
    
    Args:
        model_dir: 模型目录路径
        test_image_dir: 测试图片目录
        output_dir: 输出目录
        scale: 超分辨率倍数，如果为 None 则默认使用 2
    """
    model_dir = pathlib.Path(model_dir)
    weights_path = model_dir / 'model.weights.h5'
    config_path = model_dir / 'config.json'
    
    print(f"\n{'='*60}")
    print(f"测试 RCAN 模型")
    print(f"{'='*60}")
    print(f"模型目录: {model_dir.absolute()}")
    
    # 检查文件
    if not weights_path.exists():
        print(f"❌ 权重文件不存在: {weights_path}")
        return False
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    # 加载配置
    config = load_config(config_path)
    print(f"\n模型配置:")
    print(f"  n_feats: {config.get('n_feats', 64)}")
    print(f"  n_resgroups: {config.get('n_resgroups', 10)}")
    print(f"  n_resblocks: {config.get('n_resblocks', 20)}")
    print(f"  reduction: {config.get('reduction', 16)}")
    
    # 创建模型
    channels = config.get('n_feats', 64)
    num_groups = config.get('n_resgroups', 10)
    num_blocks = config.get('n_resblocks', 20)
    reduction = config.get('reduction', 16)
    # RCAN 配置中没有直接指定 scale，通常为 2 或 4
    if scale is None:
        scale = config.get('upscale', 2)  # 默认使用 2
    
    print(f"\n创建模型...")
    model = RCAN(channels=channels, num_groups=num_groups, num_blocks=num_blocks, 
                 reduction=reduction, scale=scale)
    
    # 使用虚拟输入构建模型
    dummy_input = tf.zeros([1, 64, 64, 3])
    _ = model(dummy_input)
    
    # 加载权重
    print(f"加载权重: {weights_path}")
    try:
        model.load_weights(str(weights_path))
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return False
    
    # 创建输出目录
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取测试图片
    test_image_dir = pathlib.Path(test_image_dir)
    if not test_image_dir.exists():
        print(f"❌ 测试图片目录不存在: {test_image_dir}")
        return False
    
    image_files = list(test_image_dir.glob('*.png')) + list(test_image_dir.glob('*.jpg')) + \
                  list(test_image_dir.glob('*.jpeg'))
    
    if not image_files:
        print(f"❌ 未找到测试图片")
        return False
    
    print(f"\n找到 {len(image_files)} 张测试图片")
    
    # 测试每张图片
    psnr_values = []
    ssim_values = []
    
    for img_path in image_files:
        print(f"\n处理: {img_path.name}")
        
        try:
            # 加载图片
            lr, hr = load_image(img_path, scale=scale)
            
            # 预处理（添加批次维度）
            lr_tensor = tf.expand_dims(tf.constant(lr), 0)
            
            # 推理
            sr_tensor = model(lr_tensor, training=False)
            sr = sr_tensor[0].numpy()
            
            # 计算指标
            psnr = calc_psnr(sr, hr)
            ssim = calc_ssim(sr, hr)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
            # 保存结果
            sr_img = (np.clip(sr, 0, 1) * 255).astype(np.uint8)
            sr_pil = Image.fromarray(sr_img)
            sr_pil.save(output_dir / f"rcan_sr_{img_path.name}")
            
            # 保存对比图（LR, SR, HR 并排）
            lr_up = (np.clip(tf.image.resize(lr, hr.shape[:2], method='bicubic').numpy(), 0, 1) * 255).astype(np.uint8)
            comparison = np.concatenate([lr_up, sr_img, (hr * 255).astype(np.uint8)], axis=1)
            Image.fromarray(comparison).save(output_dir / f"rcan_comparison_{img_path.name}")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    # 打印总结
    if psnr_values:
        print(f"\n{'='*60}")
        print("RCAN 测试总结:")
        print(f"  平均 PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"  平均 SSIM: {np.mean(ssim_values):.4f}")
        print(f"  结果保存至: {output_dir}")
        print(f"{'='*60}")
        return True
    else:
        print("❌ 没有成功处理的图片")
        return False


def main():
    parser = argparse.ArgumentParser(description='测试 TensorFlow 模型进行图片超分辨率')
    parser.add_argument('--models_dir', type=str, 
                       default=str(project_root / 'data' / 'models'),
                       help='模型目录路径')
    parser.add_argument('--test_image_dir', type=str,
                       default=str(project_root / 'data' / 'test_images'),
                       help='测试图片目录')
    parser.add_argument('--output_dir', type=str,
                       default=str(project_root / 'results'),
                       help='输出目录')
    parser.add_argument('--algorithm', type=str, choices=['edsr', 'rcan', 'all'],
                       default='all', help='要测试的算法 (edsr, rcan, all)')
    parser.add_argument('--scale', type=int, default=None,
                       help='超分辨率倍数 (如果不指定，将从配置读取或使用默认值)')
    
    args = parser.parse_args()
    
    models_dir = pathlib.Path(args.models_dir)
    test_image_dir = pathlib.Path(args.test_image_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    print("="*60)
    print("TensorFlow 模型超分辨率测试")
    print("="*60)
    print(f"模型目录: {models_dir.absolute()}")
    print(f"测试图片目录: {test_image_dir.absolute()}")
    print(f"输出目录: {output_dir.absolute()}")
    
    results = {}
    
    # 测试 EDSR
    if args.algorithm in ['edsr', 'all']:
        edsr_dir = models_dir / 'edsr'
        if edsr_dir.exists():
            edsr_output = output_dir / 'edsr'
            results['edsr'] = test_edsr_model(edsr_dir, test_image_dir, edsr_output, scale=args.scale)
        else:
            print(f"\n⚠️  EDSR 模型目录不存在: {edsr_dir}")
            results['edsr'] = False
    
    # 测试 RCAN
    if args.algorithm in ['rcan', 'all']:
        rcan_dir = models_dir / 'rcan'
        if rcan_dir.exists():
            rcan_output = output_dir / 'rcan'
            results['rcan'] = test_rcan_model(rcan_dir, test_image_dir, rcan_output, scale=args.scale)
        else:
            print(f"\n⚠️  RCAN 模型目录不存在: {rcan_dir}")
            results['rcan'] = False
    
    # 最终总结
    print(f"\n{'='*60}")
    print("最终总结")
    print(f"{'='*60}")
    
    all_success = True
    for alg, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {alg.upper()}: {status}")
        if not success:
            all_success = False
    
    if all_success:
        print(f"\n✅ 所有模型测试完成！")
        print(f"结果保存在: {output_dir}")
    else:
        print(f"\n⚠️  部分模型测试失败，请检查错误信息")
    
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
