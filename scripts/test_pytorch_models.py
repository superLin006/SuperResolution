"""测试PyTorch模型文件（检查文件是否存在和格式）"""

import os
import json
import pathlib
import sys

def check_model_files(model_dir, algorithm):
    """检查模型文件"""
    model_dir = pathlib.Path(model_dir)
    
    print(f"\n{'='*60}")
    print(f"检查 {algorithm.upper()} 模型文件")
    print(f"{'='*60}")
    print(f"目录: {model_dir.absolute()}")
    
    if not model_dir.exists():
        print(f"❌ 目录不存在: {model_dir}")
        return False
    
    files = list(model_dir.iterdir())
    if not files:
        print("❌ 目录为空")
        return False
    
    print(f"\n找到 {len(files)} 个文件:")
    has_bin = False
    has_json = False
    
    for f in files:
        size = f.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.2f} MB ({size} bytes)")
        
        if f.name.endswith('.bin') or f.name.endswith('.pth'):
            has_bin = True
        if f.name == 'config.json':
            has_json = True
            # 读取配置
            try:
                with open(f, 'r') as cfg:
                    config = json.load(cfg)
                print(f"    配置内容:")
                for key, value in list(config.items())[:5]:
                    print(f"      {key}: {value}")
            except Exception as e:
                print(f"    无法读取配置: {e}")
    
    print(f"\n文件检查:")
    print(f"  {'✅' if has_bin else '❌'} 模型文件 (.bin/.pth): {has_bin}")
    print(f"  {'✅' if has_json else '❌'} 配置文件 (config.json): {has_json}")
    
    return has_bin and has_json

def main():
    project_root = pathlib.Path(__file__).parent.parent
    models_dir = project_root / 'data' / 'models'
    
    print("="*60)
    print("PyTorch模型文件检查")
    print("="*60)
    
    algorithms = ['edsr', 'rcan']
    all_ok = True
    
    for alg in algorithms:
        model_dir = models_dir / alg
        ok = check_model_files(model_dir, alg)
        all_ok = all_ok and ok
    
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    
    if all_ok:
        print("✅ 所有模型文件已找到")
        print("\n注意:")
        print("- 这些是PyTorch格式的模型文件")
        print("- 项目使用TensorFlow框架")
        print("- 需要进行格式转换才能使用")
        print("\n选项:")
        print("1. 使用ONNX进行转换（推荐）")
        print("2. 使用项目代码重新训练")
        print("3. 当前可以继续使用随机权重测试代码功能")
    else:
        print("❌ 部分模型文件缺失")
        print("请确保模型文件已正确放置")
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())
