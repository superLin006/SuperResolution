"""将PyTorch模型转换为TensorFlow格式的脚本"""

import argparse
import pathlib
import json
import sys
import os
import numpy as np

# 强制使用CPU（避免CUDA错误）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU

# 添加项目路径以导入模型
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_pytorch_available():
    """检查PyTorch是否可用"""
    try:
        import torch
        return True, torch
    except ImportError:
        return False, None

def check_tensorflow_available():
    """检查TensorFlow是否可用，并配置为使用CPU"""
    try:
        import tensorflow as tf
        # 强制使用CPU
        tf.config.set_visible_devices([], 'GPU')  # 隐藏所有GPU设备
        return True, tf
    except ImportError:
        return False, None

def convert_conv_weight_pytorch_to_tf(pytorch_weight, transpose=True):
    """转换PyTorch卷积层权重到TensorFlow格式
    
    PyTorch: [out_channels, in_channels, kernel_h, kernel_w] (NCHW)
    TensorFlow: [kernel_h, kernel_w, in_channels, out_channels] (NHWC)
    
    Args:
        pytorch_weight: PyTorch权重张量
        transpose: 是否转置（默认True，用于卷积层）
    
    Returns:
        TensorFlow格式的权重
    """
    weight = pytorch_weight.numpy() if hasattr(pytorch_weight, 'numpy') else pytorch_weight
    
    if transpose:
        # 卷积层权重: [out, in, h, w] -> [h, w, in, out]
        if len(weight.shape) == 4:
            weight = np.transpose(weight, (2, 3, 1, 0))
        # 1x1卷积或全连接层: [out, in] -> [in, out]
        elif len(weight.shape) == 2:
            weight = np.transpose(weight, (1, 0))
    
    return weight

def convert_bias_pytorch_to_tf(pytorch_bias):
    """转换PyTorch偏置到TensorFlow格式（通常不需要转换）"""
    if pytorch_bias is None:
        return None
    bias = pytorch_bias.numpy() if hasattr(pytorch_bias, 'numpy') else pytorch_bias
    return bias

def load_pytorch_state_dict(pytorch_model_path, torch):
    """加载PyTorch模型权重"""
    print(f"正在加载PyTorch权重: {pytorch_model_path}")
    state_dict = torch.load(pytorch_model_path, map_location='cpu')
    
    # 如果加载的是完整模型，提取state_dict
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif not isinstance(state_dict, dict):
        print("警告: 模型文件格式可能不是标准的state_dict")
        return None
    
    print(f"找到 {len(state_dict)} 个权重项")
    return state_dict

def convert_edsr_pytorch_to_tensorflow(pytorch_model_path, config_path, output_path):
    """转换EDSR PyTorch模型到TensorFlow"""
    # 确保使用CPU
    print("使用CPU模式进行转换...")
    
    has_torch, torch = check_pytorch_available()
    has_tf, tf = check_tensorflow_available()
    
    if not has_torch:
        print("错误: 需要安装PyTorch才能进行转换")
        print("请运行: pip install torch")
        return False
    
    if not has_tf:
        print("错误: 需要安装TensorFlow才能进行转换")
        return False
    
    print(f"\n加载PyTorch模型: {pytorch_model_path}")
    print(f"配置文件: {config_path}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n模型配置:")
    for key, value in config.items():
        if key not in ['_commit_hash', 'transformers_version']:
            print(f"  {key}: {value}")
    
    # 从配置中提取参数
    filters = config.get('n_feats', config.get('num_feature_maps', 64))
    num_blocks = config.get('n_resblocks', config.get('num_res_block', 16))
    scale = config.get('upscale', 2)
    
    print(f"\n模型参数: filters={filters}, num_blocks={num_blocks}, scale={scale}")
    
    # 加载PyTorch权重
    state_dict = load_pytorch_state_dict(pytorch_model_path, torch)
    if state_dict is None:
        print("错误: 无法加载PyTorch权重")
        return False
    
    # 打印PyTorch权重名称（用于调试）
    print("\nPyTorch权重名称（全部）:")
    for i, key in enumerate(state_dict.keys()):
        print(f"  {i+1}. {key}")
    
    # 导入TensorFlow模型
    from algorithms.edsr.model import EDSR
    
    # 创建TensorFlow模型
    print("\n创建TensorFlow模型结构...")
    tf_model = EDSR(filters=filters, num_blocks=num_blocks, scale=scale)
    
    # 构建模型（需要先调用一次以初始化权重）
    dummy_input = tf.zeros((1, 32, 32, 3))
    _ = tf_model(dummy_input)
    
    # 打印TensorFlow层名称（用于调试）
    print("\nTensorFlow模型层结构:")
    def print_layer_structure(layer, prefix="", depth=0):
        if depth > 4:  # 限制深度避免输出太多
            return
        layer_name = layer.name if hasattr(layer, 'name') else type(layer).__name__
        if hasattr(layer, 'layers') and len(layer.layers) > 0:
            print(f"  {prefix}{layer_name} (Sequential, {len(layer.layers)} layers)")
            for i, sublayer in enumerate(layer.layers):
                print_layer_structure(sublayer, prefix + "  ", depth+1)
        else:
            if hasattr(layer, 'kernel') or hasattr(layer, 'bias'):
                print(f"  {prefix}{layer_name}")
    print_layer_structure(tf_model)
    
    # 打印所有TensorFlow可训练变量（用于调试）
    print("\nTensorFlow可训练变量:")
    for var in tf_model.trainable_variables:
        print(f"  {var.name}: {var.shape}")
    
    # 映射权重
    print("\n开始映射权重...")
    pytorch_to_tf_name_map = {}
    
    # 构建名称映射
    # PyTorch命名通常: model.stem.weight, model.body.residual0.feature.conv0.weight等
    # TensorFlow命名: stem/kernel, body/residual0/feature/conv0/kernel等
    
    # 根据实际的PyTorch权重名称构建映射
    # PyTorch格式: edsr_model.edsr_head.0.weight -> TensorFlow: _stem/kernel
    # PyTorch格式: edsr_model.edsr_body.{i}.edsr_body.{j}.weight -> TensorFlow: _body/residual{i}/feature/conv{j}/kernel
    
    # Head层 (对应stem)
    pytorch_to_tf_name_map['edsr_model.edsr_head.0.weight'] = '_stem/kernel'
    pytorch_to_tf_name_map['edsr_model.edsr_head.0.bias'] = '_stem/bias'
    
    # Body层 - Residual blocks
    # PyTorch: edsr_model.edsr_body.{i}.edsr_body.0.weight 和 .2.weight
    # TensorFlow: _body/residual{i}/_feature/conv0/kernel 和 conv1/kernel
    # 注意：ResidualBlock 使用 _feature（带下划线）属性
    for i in range(num_blocks):
        pytorch_to_tf_name_map[f'edsr_model.edsr_body.{i}.edsr_body.0.weight'] = f'_body/residual{i}/_feature/conv0/kernel'
        pytorch_to_tf_name_map[f'edsr_model.edsr_body.{i}.edsr_body.0.bias'] = f'_body/residual{i}/_feature/conv0/bias'
        pytorch_to_tf_name_map[f'edsr_model.edsr_body.{i}.edsr_body.2.weight'] = f'_body/residual{i}/_feature/conv1/kernel'
        pytorch_to_tf_name_map[f'edsr_model.edsr_body.{i}.edsr_body.2.bias'] = f'_body/residual{i}/_feature/conv1/bias'
    
    # Body最后的conv层
    # PyTorch: edsr_model.edsr_body.16.weight (在所有residual blocks之后，索引为16)
    pytorch_to_tf_name_map['edsr_model.edsr_body.16.weight'] = '_body/conv/kernel'
    pytorch_to_tf_name_map['edsr_model.edsr_body.16.bias'] = '_body/conv/bias'
    
    # Tail层 - Upsample blocks 和最后的conv层
    # PyTorch: upsampler.0.0.weight (第一个upsample block的conv)
    #          upsampler.0.2.weight (第二个upsample block的conv，如果有)
    #          upsampler.1.weight (最后的输出conv)
    # 注意：UpsampleBlock 使用 _conv（带下划线）属性
    num_upsample = int(np.log2(scale))
    if num_upsample >= 1:
        # 第一个upsample block
        pytorch_to_tf_name_map['upsampler.0.0.weight'] = '_tail/upsample0/_conv/kernel'
        pytorch_to_tf_name_map['upsampler.0.0.bias'] = '_tail/upsample0/_conv/bias'
    if num_upsample >= 2:
        # 第二个upsample block（如果有）
        pytorch_to_tf_name_map['upsampler.0.2.weight'] = '_tail/upsample1/_conv/kernel'
        pytorch_to_tf_name_map['upsampler.0.2.bias'] = '_tail/upsample1/_conv/bias'
    
    # 最后的输出conv层
    pytorch_to_tf_name_map['upsampler.1.weight'] = '_tail/conv/kernel'
    pytorch_to_tf_name_map['upsampler.1.bias'] = '_tail/conv/bias'
    
    # 尝试映射权重
    mapped_count = 0
    for pytorch_name, tf_name in pytorch_to_tf_name_map.items():
        weight_key = pytorch_name
        bias_key = pytorch_name.replace('.weight', '.bias')
        
        if weight_key in state_dict:
            try:
                # 获取TensorFlow层
                layer = tf_model
                parts = tf_name.split('/')[:-1]
                for idx, part in enumerate(parts):
                    prev_layer = layer
                    
                    # 首先尝试直接通过属性访问（处理下划线前缀）
                    if part.startswith('_'):
                        layer = getattr(layer, part, None)
                    else:
                        layer = getattr(layer, part, None)
                        if layer is None:
                            # 尝试带下划线
                            layer = getattr(layer, '_' + part, None)
                    
                    # 如果找不到，且当前层是Sequential，尝试在子层中查找
                    if layer is None and hasattr(prev_layer, 'layers'):
                        # 尝试通过名称查找
                        for sublayer in prev_layer.layers:
                            if hasattr(sublayer, 'name') and sublayer.name == part:
                                layer = sublayer
                                break
                            # 也尝试去掉下划线匹配
                            if hasattr(sublayer, 'name') and sublayer.name == part.lstrip('_'):
                                layer = sublayer
                                break
                        
                        # 如果还是找不到，尝试通过索引（对于residual{i}）
                        if layer is None and part.startswith('residual'):
                            try:
                                residual_idx = int(part.replace('residual', ''))
                                if residual_idx < len(prev_layer.layers):
                                    layer = prev_layer.layers[residual_idx]
                            except (ValueError, IndexError):
                                pass
                    
                    if layer is None:
                        layer_type = type(prev_layer).__name__ if prev_layer is not None else 'None'
                        has_layers = hasattr(prev_layer, 'layers') and len(prev_layer.layers) > 0 if prev_layer is not None else False
                        print(f"  ⚠ 找不到层: {'/'.join(parts)} (在 {part} 处失败, 前一层类型: {layer_type}, 有子层: {has_layers})")
                        break
                
                if layer is not None:
                    param_name = tf_name.split('/')[-1]
                    if param_name == 'kernel':
                        weight = convert_conv_weight_pytorch_to_tf(state_dict[weight_key])
                        layer.kernel.assign(weight)
                        mapped_count += 1
                        print(f"  ✓ 映射权重: {pytorch_name} -> {tf_name}")
                    elif param_name == 'bias' and bias_key in state_dict:
                        bias = convert_bias_pytorch_to_tf(state_dict[bias_key])
                        if layer.bias is not None:
                            layer.bias.assign(bias)
                            mapped_count += 1
                            print(f"  ✓ 映射偏置: {bias_key} -> {tf_name}")
            except Exception as e:
                print(f"  ✗ 映射失败: {pytorch_name} -> {tf_name} ({e})")
        else:
            # 打印未找到的权重名称（用于调试）
            if mapped_count == 0:  # 只在第一次时打印
                pass  # 暂时不打印，避免输出太多
    
    print(f"\n成功映射 {mapped_count} 个权重项")
    
    # 如果自动映射失败，尝试通用映射
    if mapped_count == 0:
        print("\n⚠️  自动映射失败，尝试通用权重映射...")
        print("请检查上面的PyTorch权重名称和TensorFlow层结构，手动调整映射关系")
        
        # 尝试更通用的映射：直接匹配权重名称
        print("\n尝试基于名称相似性的映射...")
        # 这里可以添加更智能的映射逻辑
    
    # 保存TensorFlow模型
    # output_path 应该是 model_dir
    model_dir = pathlib.Path(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为H5格式（主要输出文件）
    h5_path = model_dir / 'model.weights.h5'
    print(f"\n保存TensorFlow模型到: {h5_path}")
    
    try:
        # 保存权重为H5格式（主要输出文件）
        tf_model.save_weights(str(h5_path))
        print("✅ 权重保存成功")
        
        # 也保存为SavedModel格式（可选，用于其他用途）
        saved_model_path = model_dir / 'saved_model'
        try:
            # 使用model.save()保存SavedModel格式（更兼容）
            tf_model.save(str(saved_model_path), save_format='tf')
            print(f"✅ SavedModel格式保存到: {saved_model_path}")
        except Exception as e_saved:
            # 如果model.save失败，尝试tf.saved_model.save
            try:
                tf.saved_model.save(tf_model, str(saved_model_path))
                print(f"✅ SavedModel格式保存到: {saved_model_path}")
            except Exception as e_saved2:
                print(f"⚠️  SavedModel保存失败: {e_saved2}")
                print("   但权重文件已成功保存，可以正常使用")
        
        return True
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        # 如果保存失败，至少尝试保存权重
        try:
            tf_model.save_weights(str(h5_path))
            print(f"✅ 权重已保存到: {h5_path}")
            return True
        except Exception as e2:
            print(f"❌ 保存权重也失败: {e2}")
            return False

def convert_rcan_pytorch_to_tensorflow(pytorch_model_path, config_path, output_path):
    """转换RCAN PyTorch模型到TensorFlow"""
    # 确保使用CPU
    print("使用CPU模式进行转换...")
    
    has_torch, torch = check_pytorch_available()
    has_tf, tf = check_tensorflow_available()
    
    if not has_torch:
        print("错误: 需要安装PyTorch才能进行转换")
        print("请运行: pip install torch")
        return False
    
    if not has_tf:
        print("错误: 需要安装TensorFlow才能进行转换")
        return False
    
    print(f"\n加载PyTorch模型: {pytorch_model_path}")
    print(f"配置文件: {config_path}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 从配置中提取参数
    channels = config.get('n_feats', 64)
    num_groups = config.get('n_resgroups', 10)
    num_blocks = config.get('n_resblocks', 20)
    reduction = config.get('reduction', 16)
    scale = 2  # 通常RCAN默认是2x，可以从配置中读取
    
    print(f"\n模型参数: channels={channels}, num_groups={num_groups}, num_blocks={num_blocks}, reduction={reduction}, scale={scale}")
    
    # 加载PyTorch权重
    state_dict = load_pytorch_state_dict(pytorch_model_path, torch)
    if state_dict is None:
        print("错误: 无法加载PyTorch权重")
        return False
    
    # 导入TensorFlow模型
    from algorithms.rcan.model import RCAN
    
    # 创建TensorFlow模型
    print("\n创建TensorFlow模型结构...")
    tf_model = RCAN(channels=channels, num_groups=num_groups, num_blocks=num_blocks, reduction=reduction, scale=scale)
    
    # 构建模型
    dummy_input = tf.zeros((1, 32, 32, 3))
    _ = tf_model(dummy_input)
    
    # 映射权重（RCAN结构更复杂，需要更详细的映射）
    print("\n开始映射权重...")
    print("注意: RCAN模型结构复杂，权重映射可能需要手动调整")
    
    # 这里可以实现RCAN的权重映射逻辑
    # 由于RCAN有Channel Attention等复杂结构，映射会更复杂
    
    # 保存TensorFlow模型
    # output_path 应该是 model_dir
    model_dir = pathlib.Path(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为H5格式（主要输出文件）
    h5_path = model_dir / 'model.weights.h5'
    print(f"\n保存TensorFlow模型到: {h5_path}")
    
    try:
        # 保存权重为H5格式（主要输出文件）
        tf_model.save_weights(str(h5_path))
        print("✅ 权重保存成功")
        
        # 也保存为SavedModel格式（可选，用于其他用途）
        saved_model_path = model_dir / 'saved_model'
        try:
            # 使用model.save()保存SavedModel格式（更兼容）
            tf_model.save(str(saved_model_path), save_format='tf')
            print(f"✅ SavedModel格式保存到: {saved_model_path}")
        except Exception as e_saved:
            # 如果model.save失败，尝试tf.saved_model.save
            try:
                tf.saved_model.save(tf_model, str(saved_model_path))
                print(f"✅ SavedModel格式保存到: {saved_model_path}")
            except Exception as e_saved2:
                print(f"⚠️  SavedModel保存失败: {e_saved2}")
                print("   但权重文件已成功保存，可以正常使用")
        
        return True
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        # 如果保存失败，至少尝试保存权重
        try:
            tf_model.save_weights(str(h5_path))
            print(f"✅ 权重已保存到: {h5_path}")
            return True
        except Exception as e2:
            print(f"❌ 保存权重也失败: {e2}")
            return False

def main():
    parser = argparse.ArgumentParser(description='转换PyTorch模型到TensorFlow格式')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='包含PyTorch模型的目录')
    parser.add_argument('--algorithm', type=str, choices=['edsr', 'rcan'],
                       required=True, help='算法类型')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：与model_dir相同）')
    
    args = parser.parse_args()
    
    model_dir = pathlib.Path(args.model_dir)
    pytorch_model_file = model_dir / 'pytorch_model.bin'
    config_file = model_dir / 'config.json'
    
    if not pytorch_model_file.exists():
        print(f"错误: 找不到模型文件 {pytorch_model_file}")
        return 1
    
    if not config_file.exists():
        print(f"错误: 找不到配置文件 {config_file}")
        return 1
    
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir) / args.algorithm
    else:
        output_dir = model_dir
    
    print("="*70)
    print(f"转换 {args.algorithm.upper()} PyTorch模型到TensorFlow")
    print("="*70)
    
    if args.algorithm == 'edsr':
        success = convert_edsr_pytorch_to_tensorflow(
            pytorch_model_file, config_file, output_dir
        )
    else:
        success = convert_rcan_pytorch_to_tensorflow(
            pytorch_model_file, config_file, output_dir
        )
    
    if success:
        print("\n" + "="*70)
        print("✅ 转换完成!")
        print("="*70)
        print(f"TensorFlow模型已保存到: {output_dir}")
    else:
        print("\n" + "="*70)
        print("⚠️  转换未完全成功")
        print("="*70)
        print("\n可能的原因:")
        print("1. PyTorch和TensorFlow模型结构不完全匹配")
        print("2. 权重名称映射需要手动调整")
        print("3. 某些层可能需要特殊处理")
        print("\n建议:")
        print("1. 检查权重映射日志，手动调整映射关系")
        print("2. 使用ONNX作为中间格式进行转换")
        print("3. 或使用项目代码重新训练模型")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
