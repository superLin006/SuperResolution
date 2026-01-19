"""Utility functions for RCAN."""

import numpy as np
import tensorflow as tf
try:
    import skimage.metrics
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def calc_psnr(sr, hr, rgb_range=1.0, shave=8):
    """Calculate PSNR for given SR and HR images.
    
    Args:
        sr: Super-resolved image tensor
        hr: High-resolution ground truth image tensor
        rgb_range: RGB value range (default: 1.0 for normalized images)
        shave: Border pixels to shave (default: 8)
    
    Returns:
        PSNR value
    """
    if isinstance(sr, tf.Tensor):
        sr = sr.numpy()
    if isinstance(hr, tf.Tensor):
        hr = hr.numpy()
    
    if len(sr.shape) == 4:
        sr = sr[0]
    if len(hr.shape) == 4:
        hr = hr[0]
    
    sr, hr = sr[shave:-shave, shave:-shave], hr[shave:-shave, shave:-shave]

    scale = 255 / rgb_range
    sr = np.clip(sr * scale, 0, 255).round().astype(np.uint8)
    hr = np.clip(hr * scale, 0, 255).round().astype(np.uint8)

    if HAS_SKIMAGE:
        return skimage.metrics.peak_signal_noise_ratio(hr, sr, data_range=255)
    else:
        # Fallback implementation
        mse = np.mean((hr - sr) ** 2)
        if mse == 0:
            return 100
        return 20 * np.log10(255.0 / np.sqrt(mse))


def calc_ssim(sr, hr, rgb_range=1.0, shave=8):
    """Calculate SSIM for given SR and HR images.
    
    Args:
        sr: Super-resolved image tensor
        hr: High-resolution ground truth image tensor
        rgb_range: RGB value range (default: 1.0 for normalized images)
        shave: Border pixels to shave (default: 8)
    
    Returns:
        SSIM value
    """
    if HAS_SKIMAGE:
        if isinstance(sr, tf.Tensor):
            sr = sr.numpy()
        if isinstance(hr, tf.Tensor):
            hr = hr.numpy()
        
        if len(sr.shape) == 4:
            sr = sr[0]
        if len(hr.shape) == 4:
            hr = hr[0]
        
        sr, hr = sr[shave:-shave, shave:-shave], hr[shave:-shave, shave:-shave]

        scale = 255 / rgb_range
        sr = np.clip(sr * scale, 0, 255).round().astype(np.uint8)
        hr = np.clip(hr * scale, 0, 255).round().astype(np.uint8)

        return skimage.metrics.structural_similarity(hr, sr, data_range=255, multichannel=True, channel_axis=2)
    else:
        return 0.0
