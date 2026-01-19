"""Model conversion script for EDSR using mlkits."""

import argparse
import pathlib
import yaml
import tensorflow as tf
import sys
import os

# Add parent directory to path to import mlkits
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent / 'neuropilot-sdk' / 'offline_tool'))

try:
    from mlkits import api
    HAS_MLKITS = True
except ImportError:
    HAS_MLKITS = False
    print("Warning: mlkits not found. Please install mlkits package.")
    print("You may need to install: pip install mlkits-8.6.4+apu7.apu8.2521.2-*.whl")

try:
    from mtk_converter.python.converters.tensorflow.converter import TensorFlowConverter
    HAS_MTK_CONVERTER = True
except ImportError:
    HAS_MTK_CONVERTER = False
    print("Warning: mtk_converter not found. DLA conversion will be skipped.")

from model import EDSR


def compile_tflite_to_dla(tflite_path, dla_path, platform='MT6989', neuron_sdk_path=None):
    """使用 ncc-tflite 将 TFLite 编译为 DLA 格式
    
    Args:
        tflite_path: TFLite 模型文件路径（字符串或 Path 对象）
        dla_path: 输出 DLA 文件路径（字符串或 Path 对象）
        platform: 目标平台 (MT6989, MT6991, MT6899, MT8371 等)
        neuron_sdk_path: Neuron SDK 路径（字符串或 Path 对象）
    """
    import subprocess
    import shutil
    
    # 转换为 Path 对象
    tflite_path = pathlib.Path(tflite_path)
    dla_path = pathlib.Path(dla_path)
    
    # 查找 ncc-tflite 工具
    if neuron_sdk_path is None:
        neuron_sdk_path = pathlib.Path(__file__).parent.parent.parent.parent / 'neuropilot-sdk'
    else:
        neuron_sdk_path = pathlib.Path(neuron_sdk_path)
    
    # 可能的 ncc-tflite 路径
    possible_paths = [
        neuron_sdk_path / 'neuron_sdk' / 'host' / 'bin' / 'ncc-tflite',
        neuron_sdk_path / 'host' / 'bin' / 'ncc-tflite',
        pathlib.Path('/usr/local/bin/ncc-tflite'),
        pathlib.Path('/usr/bin/ncc-tflite'),
    ]
    
    ncc_tflite = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            ncc_tflite = path
            break
    
    if ncc_tflite is None:
        # 尝试在 PATH 中查找
        ncc_tflite = shutil.which('ncc-tflite')
        if ncc_tflite:
            ncc_tflite = pathlib.Path(ncc_tflite)
    
    if ncc_tflite is None or not ncc_tflite.exists():
        print(f"⚠️  未找到 ncc-tflite 工具")
        print(f"  查找路径: {[str(p) for p in possible_paths]}")
        print(f"  请确保 Neuron SDK 已正确安装，或手动设置 NEURON_SDK_PATH 环境变量")
        print(f"  手动编译命令示例:")
        print(f"    ncc-tflite --arch=mdla5.5 -O3 --l1-size-kb=2048 --num-mdla=2 \\")
        print(f"      --relax-fp32 --opt-accuracy --opt-footprint -d {dla_path} {tflite_path}")
        return False
    
    print(f"  找到 ncc-tflite: {ncc_tflite}")
    
    # 平台配置
    platform_configs = {
        'MT6989': {
            'arch': 'mdla5.5',
            'l1_size': '2048',
            'num_mdla': '2'
        },
        'MT6991': {
            'arch': 'mdla5.5',
            'l1_size': '7168',
            'num_mdla': '4'
        },
        'MT6899': {
            'arch': 'mdla5.5',
            'l1_size': '2048',
            'num_mdla': '2'
        },
        'MT8371': {
            'arch': 'mdla5.3,edma3.6',
            'l1_size': '256',
            'num_mdla': '1'
        }
    }
    
    config = platform_configs.get(platform, platform_configs['MT6989'])
    print(f"  平台配置: {platform}")
    print(f"    ARCH: {config['arch']}")
    print(f"    L1 Size: {config['l1_size']}KB")
    print(f"    NUM MDLA: {config['num_mdla']}")
    
    # 设置环境变量
    env = os.environ.copy()
    neuron_sdk_host = neuron_sdk_path / 'neuron_sdk' / 'host'
    if neuron_sdk_host.exists():
        env['PATH'] = str(neuron_sdk_host / 'bin') + ':' + env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(neuron_sdk_host / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    
    # 构建编译命令
    log_path = dla_path.parent / f'compile_{platform}.log'
    cmd = [
        str(ncc_tflite),
        f"--arch={config['arch']}",
        '-O3',
        f"--l1-size-kb={config['l1_size']}",
        f"--num-mdla={config['num_mdla']}",
        '--show-memory-summary',
        '--relax-fp32',
        '--opt-accuracy',
        '--opt-footprint',
        '--fc-to-conv',
        '-d', str(dla_path),
        str(tflite_path)
    ]
    
    print(f"  执行编译命令...")
    print(f"  {' '.join(cmd)}")
    
    try:
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True
            )
        
        if dla_path.exists():
            print(f"✅ DLA 文件已生成: {dla_path}")
            print(f"  编译日志: {log_path}")
            return True
        else:
            print(f"⚠️  DLA 文件未生成，请查看日志: {log_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 编译失败 (退出码: {e.returncode})")
        print(f"  请查看日志: {log_path}")
        if log_path.exists():
            with open(log_path, 'r') as f:
                print("  最后 20 行日志:")
                lines = f.readlines()
                for line in lines[-20:]:
                    print(f"    {line.rstrip()}")
        return False
    except FileNotFoundError:
        print(f"❌ ncc-tflite 工具未找到")
        return False


def convert_model(model_path, config_path, output_dir, filters=256, num_blocks=32, scale=2, platform='MT6989', neuron_sdk_path=None):
    """Convert EDSR model using mlkits.
    
    Args:
        model_path: Path to saved model checkpoint
        config_path: Path to mlkits config YAML file
        output_dir: Output directory for converted model
        filters: Number of filters
        num_blocks: Number of residual blocks
        scale: Super-resolution scale factor
        platform: Target platform (MT6989, MT6991, MT6899, MT8371)
        neuron_sdk_path: Path to Neuron SDK (auto-detected if None)
    """
    if not HAS_MLKITS:
        print("Error: mlkits is required for model conversion")
        return
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup mlkits
    api.setup_mlkits(config)
    
    # Create model
    model = EDSR(filters=filters, num_blocks=num_blocks, scale=scale)
    
    # Build model with input shape from config
    input_shape = config['model']['inputs'][0]['shape']
    dummy_input = tf.zeros(input_shape)
    _ = model(dummy_input)
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        if os.path.isdir(model_path):
            ckpt_path = os.path.join(model_path, 'ckpt')
            if os.path.exists(ckpt_path + '.index'):
                print(f"Loading weights from {ckpt_path}")
                model.load_weights(ckpt_path)
        elif os.path.isfile(model_path):
            print(f"Loading weights from {model_path}")
            try:
                model.load_weights(model_path)
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
    
    # Build supernet (for NAS) or convert directly
    print("Building model for conversion...")
    try:
        # For direct conversion, analyze the model first
        api.analyze_reference(model)
        
        # Convert model
        print("Converting model...")
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model in SavedModel format first
        saved_model_path = output_dir / 'saved_model'
        model.save(saved_model_path.as_posix())
        print(f"✅ SavedModel 已保存到: {saved_model_path}")
        
        # 转换为 TFLite，然后编译为 DLA
        if HAS_MTK_CONVERTER:
            print("\n开始转换为 MTK DLA 格式...")
            try:
                # 步骤 1: 使用 mtk_converter 将 SavedModel 转换为 TFLite
                print(f"  步骤 1: 加载 SavedModel: {saved_model_path}")
                # 不提供 input_shapes，让 converter 从 SavedModel 自动获取
                converter = TensorFlowConverter.from_saved_model_dir(
                    str(saved_model_path)
                )
                
                # 转换为 TFLite
                tflite_path = output_dir / 'model.tflite'
                print(f"  步骤 2: 转换为 TFLite: {tflite_path}")
                converter.convert_to_tflite(output_file=str(tflite_path))
                print(f"✅ TFLite 已保存到: {tflite_path}")
                
                # 步骤 3: 使用 ncc-tflite 编译为 DLA
                print(f"  步骤 3: 编译 TFLite 为 DLA...")
                dla_path = output_dir / f'model_{platform.lower()}.dla'
                if neuron_sdk_path is None:
                    neuron_sdk_path = pathlib.Path(__file__).parent.parent.parent.parent / 'neuropilot-sdk'
                else:
                    neuron_sdk_path = pathlib.Path(neuron_sdk_path)
                
                compile_tflite_to_dla(
                    tflite_path=str(tflite_path),
                    dla_path=str(dla_path),
                    platform=platform,
                    neuron_sdk_path=neuron_sdk_path
                )
                
            except Exception as e:
                print(f"⚠️  DLA 转换失败: {e}")
                print("  但 SavedModel 已成功保存")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️  mtk_converter 未安装，跳过 DLA 转换")
            print("  可以使用以下命令手动转换:")
            print(f"  mtk_tensorflow_converter --input_saved_model_dir {saved_model_path} --output_file_format tflite --output_file {output_dir}/model.tflite")
        
        print("\n✅ 转换完成！")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert EDSR model using mlkits')
    parser.add_argument('--model_path', type=str, default='',
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to mlkits config YAML file')
    parser.add_argument('--output_dir', type=str, default='../../results/edsr/converted',
                       help='Output directory for converted model')
    parser.add_argument('--filters', type=int, default=256, help='Number of filters')
    parser.add_argument('--num_blocks', type=int, default=32, help='Number of residual blocks')
    parser.add_argument('--scale', type=int, default=2, help='Super-resolution scale factor')
    parser.add_argument('--platform', type=str, default='MT6989',
                       help='Target platform (MT6989, MT6991, MT6899, MT8371)')
    parser.add_argument('--neuron_sdk_path', type=str, default=None,
                       help='Path to Neuron SDK (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    convert_model(
        model_path=args.model_path,
        config_path=args.config,
        output_dir=args.output_dir,
        filters=args.filters,
        num_blocks=args.num_blocks,
        scale=args.scale,
        platform=args.platform,
        neuron_sdk_path=args.neuron_sdk_path
    )
