"""创建测试图像的脚本"""

import numpy as np
from PIL import Image
import pathlib

def create_test_images(output_dir, num_images=2):
    """创建测试图像
    
    Args:
        output_dir: 输出目录
        num_images: 创建的图像数量
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建几个简单的测试图像
    images = [
        ("test_pattern.png", create_pattern_image()),
        ("test_gradient.png", create_gradient_image()),
    ]
    
    for name, img_array in images[:num_images]:
        img = Image.fromarray(img_array)
        img.save(output_dir / name)
        print(f"创建测试图像: {output_dir / name} ({img.size})")

def create_pattern_image(size=(256, 256)):
    """创建棋盘图案测试图像"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    block_size = 32
    for i in range(0, size[0], block_size):
        for j in range(0, size[1], block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i:i+block_size, j:j+block_size] = [255, 255, 255]
            else:
                img[i:i+block_size, j:j+block_size] = [128, 128, 128]
    return img

def create_gradient_image(size=(256, 256)):
    """创建渐变测试图像"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = [
                int(255 * i / size[0]),
                int(255 * j / size[1]),
                int(255 * (i + j) / (size[0] + size[1]))
            ]
    return img

if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/test_images'
    create_test_images(output_dir)
