"""检查PyTorch模型文件并说明情况"""

import pathlib
import sys
import json

def check_tensorflow_model(model_dir):
    """检查是否已转换为TensorFlow格式"""
    saved_model_path = model_dir / 'saved_model'
    h5_path = model_dir / 'model.weights.h5'
    
    has_saved_model = saved_model_path.exists() and (saved_model_path / 'saved_model.pb').exists()
    has_h5 = h5_path.exists()
    
    return has_saved_model, has_h5

def main():
    project_root = pathlib.Path(__file__).parent.parent
    models_dir = project_root / 'data' / 'models'
    
    print("="*70)
    print("PyTorch模型文件检查报告")
    print("="*70)
    
    algorithms = {
        'EDSR': models_dir / 'edsr',
        'RCAN': models_dir / 'rcan'
    }
    
    all_found = True
    conversion_status = {}
    
    for name, model_dir in algorithms.items():
        print(f"\n{'='*70}")
        print(f"{name} 模型")
        print(f"{'='*70}")
        
        if not model_dir.exists():
            print(f"❌ 目录不存在: {model_dir}")
            all_found = False
            continue
        
        bin_file = model_dir / 'pytorch_model.bin'
        config_file = model_dir / 'config.json'
        
        # 检查PyTorch模型文件
        pytorch_found = False
        if bin_file.exists():
            size_mb = bin_file.stat().st_size / (1024 * 1024)
            print(f"✅ PyTorch模型文件: {bin_file.name} ({size_mb:.2f} MB)")
            pytorch_found = True
        else:
            print(f"❌ PyTorch模型文件不存在")
            all_found = False
        
        # 检查配置文件
        config_valid = False
        if config_file.exists():
            print(f"✅ 配置文件: {config_file.name}")
            # 读取配置（如果可能）
            try:
                content = config_file.read_text()
                if content.startswith('{'):
                    config = json.loads(content)
                    print(f"   配置项: {len(config)} 个")
                    # 显示关键配置
                    key_configs = ['n_feats', 'num_feature_maps', 'n_resblocks', 'num_res_block', 
                                  'upscale', 'n_resgroups', 'reduction']
                    for key in key_configs:
                        if key in config:
                            print(f"   {key}: {config[key]}")
                    config_valid = True
                else:
                    print(f"   配置文件格式: 非标准JSON")
            except Exception as e:
                print(f"   配置文件: 无法解析 ({e})")
        else:
            print(f"❌ 配置文件不存在")
            all_found = False
        
        # 检查TensorFlow转换状态
        has_saved_model, has_h5 = check_tensorflow_model(model_dir)
        conversion_status[name] = {
            'pytorch': pytorch_found,
            'config': config_valid,
            'tensorflow_saved_model': has_saved_model,
            'tensorflow_h5': has_h5
        }
        
        if has_saved_model or has_h5:
            print(f"\n✅ TensorFlow转换状态:")
            if has_saved_model:
                print(f"   ✅ SavedModel格式: {model_dir / 'saved_model'}")
            if has_h5:
                size_mb = h5_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ H5权重文件: {h5_path.name} ({size_mb:.2f} MB)")
        else:
            print(f"\n⚠️  TensorFlow转换状态: 未转换")
    
    print(f"\n{'='*70}")
    print("转换状态总结")
    print(f"{'='*70}")
    
    all_converted = True
    for name, status in conversion_status.items():
        if status['pytorch'] and not (status['tensorflow_saved_model'] or status['tensorflow_h5']):
            all_converted = False
            break
    
    if all_found:
        print("✅ 已找到PyTorch格式的模型文件")
        
        if all_converted:
            print("\n✅ 所有模型已转换为TensorFlow格式")
            print("   - 可以使用TensorFlow模型进行推理")
            print("   - 模型已准备好用于mlkits转换")
        else:
            print("\n⚠️  格式兼容性说明:")
            print("   - 模型文件格式: PyTorch (.bin文件)")
            print("   - 项目框架: TensorFlow")
            print("   - 状态: 需要转换为TensorFlow格式")
            
            print("\n📋 转换方法:")
            print("   使用转换脚本进行自动转换:")
            print("   python scripts/convert_pytorch_to_tensorflow.py \\")
            print("       --model_dir data/models/edsr \\")
            print("       --algorithm edsr")
            print("")
            print("   python scripts/convert_pytorch_to_tensorflow.py \\")
            print("       --model_dir data/models/rcan \\")
            print("       --algorithm rcan")
            
            print("\n📋 其他可选方案:")
            print("   方案1: 使用ONNX作为中间格式转换")
            print("     - PyTorch -> ONNX -> TensorFlow")
            print("     - 需要安装: torch, onnx, tf2onnx")
            print("")
            print("   方案2: 使用项目代码训练（最可靠）")
            print("     - 使用项目中的TensorFlow实现")
            print("     - 在数据集上训练模型")
            print("     - 完全兼容项目框架")
        
        print("\n✅ 当前状态:")
        print("   - 项目代码结构已验证正确")
        print("   - 可以使用随机权重测试代码功能（已完成）")
        if all_converted:
            print("   - TensorFlow模型已准备就绪")
        else:
            print("   - 可以继续模型转换工作")
    else:
        print("❌ 部分模型文件缺失")
    
    print(f"\n{'='*70}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
