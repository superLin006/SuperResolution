"""Local testing script for EDSR model."""

import argparse
import pathlib
import numpy as np
import tensorflow as tf
from PIL import Image
import os

from model import EDSR
from utils import calc_psnr, calc_ssim


def load_image(image_path, scale=2):
    """Load and preprocess image.
    
    Args:
        image_path: Path to input image
        scale: Downscaling factor
    
    Returns:
        lr: Low-resolution image (normalized to [0, 1])
        hr: High-resolution image (normalized to [0, 1])
    """
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    
    # Create LR by downsampling
    h, w = img.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    
    # Use PIL for downsampling (simulates bicubic downsampling)
    lr_img = Image.fromarray((img * 255).astype(np.uint8))
    lr_img = lr_img.resize((lr_w, lr_h), Image.BICUBIC)
    lr = np.array(lr_img).astype(np.float32) / 255.0
    
    # Crop HR to match scale
    hr_h, hr_w = lr_h * scale, lr_w * scale
    hr = img[:hr_h, :hr_w]
    
    return lr, hr


def test_model(model_path, test_image_dir, output_dir, scale=2, filters=256, num_blocks=32):
    """Test EDSR model on test images.
    
    Args:
        model_path: Path to saved model or checkpoint
        test_image_dir: Directory containing test images
        output_dir: Directory to save results
        scale: Super-resolution scale factor
        filters: Number of filters
        num_blocks: Number of residual blocks
    """
    # Create model
    model = EDSR(filters=filters, num_blocks=num_blocks, scale=scale)
    
    # Build model with dummy input
    dummy_input = tf.zeros([1, 64, 64, 3])
    _ = model(dummy_input)
    
    # Load weights
    if os.path.isdir(model_path):
        # Assume checkpoint directory
        ckpt_path = os.path.join(model_path, 'ckpt')
        if os.path.exists(ckpt_path + '.index'):
            model.load_weights(ckpt_path)
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using random weights")
    elif os.path.isfile(model_path):
        try:
            model.load_weights(model_path)
        except:
            print(f"Warning: Could not load weights from {model_path}, using random weights")
    else:
        print(f"Warning: Model path {model_path} not found, using random weights")
    
    # Create output directory
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test images
    test_image_dir = pathlib.Path(test_image_dir)
    if not test_image_dir.exists():
        print(f"Test image directory {test_image_dir} does not exist")
        print("Creating a simple test by generating a random image...")
        # Create a simple test image
        test_image_dir.mkdir(parents=True, exist_ok=True)
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(test_img).save(test_image_dir / 'test.png')
    
    image_files = list(test_image_dir.glob('*.png')) + list(test_image_dir.glob('*.jpg'))
    
    if not image_files:
        print("No test images found. Please add images to test_image_dir.")
        return
    
    # Test on each image
    psnr_values = []
    ssim_values = []
    
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        
        # Load image
        lr, hr = load_image(img_path, scale=scale)
        
        # Preprocess for model (add batch dimension)
        lr_tensor = tf.expand_dims(tf.constant(lr), 0)
        
        # Inference
        sr_tensor = model(lr_tensor, training=False)
        sr = sr_tensor[0].numpy()
        
        # Calculate metrics
        psnr = calc_psnr(sr, hr)
        ssim = calc_ssim(sr, hr)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        
        print(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        
        # Save results
        sr_img = (np.clip(sr, 0, 1) * 255).astype(np.uint8)
        sr_pil = Image.fromarray(sr_img)
        sr_pil.save(output_dir / f"sr_{img_path.name}")
        
        # Save comparison (LR, SR, HR side by side)
        lr_up = (np.clip(tf.image.resize(lr, hr.shape[:2], method='bicubic').numpy(), 0, 1) * 255).astype(np.uint8)
        comparison = np.concatenate([lr_up, sr_img, (hr * 255).astype(np.uint8)], axis=1)
        Image.fromarray(comparison).save(output_dir / f"comparison_{img_path.name}")
    
    # Print summary
    if psnr_values:
        print("\n" + "="*50)
        print("Summary:")
        print(f"  Average PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"  Average SSIM: {np.mean(ssim_values):.4f}")
        print(f"  Results saved to: {output_dir}")
        print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test EDSR model locally')
    parser.add_argument('--model_path', type=str, default='', 
                       help='Path to model checkpoint or weights')
    parser.add_argument('--test_image_dir', type=str, default='../../data/test_images',
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='../../results/edsr',
                       help='Directory to save results')
    parser.add_argument('--scale', type=int, default=2, help='Super-resolution scale factor')
    parser.add_argument('--filters', type=int, default=256, help='Number of filters')
    parser.add_argument('--num_blocks', type=int, default=32, help='Number of residual blocks')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model_path,
        test_image_dir=args.test_image_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        filters=args.filters,
        num_blocks=args.num_blocks
    )
