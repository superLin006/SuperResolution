"""RCAN (Residual Channel Attention Network) model implementation."""

import numpy as np
import tensorflow as tf


class ChannelAttention(tf.keras.layers.Layer):
    """Channel Attention module.
    
    Uses global average pooling followed by MLP to generate channel-wise attention weights.
    """
    
    def __init__(self, channels, reduction=16, name='ca'):
        """Initialize Channel Attention module.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for MLP (default: 16)
            name: Layer name
        """
        super().__init__(name=name)
        self._channels = channels
        self._reduction = reduction
        
        # MLP layers for channel attention
        self._global_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='global_pool')
        self._mlp = tf.keras.Sequential([
            tf.keras.layers.Conv2D(channels // reduction, 1, name='conv1'),
            tf.keras.layers.ReLU(name='relu'),
            tf.keras.layers.Conv2D(channels, 1, name='conv2'),
            tf.keras.layers.Activation('sigmoid', name='sigmoid')
        ], name='mlp')
        
    def call(self, inputs):
        """Apply channel attention.
        
        Args:
            inputs: Input tensor of shape [B, H, W, C]
        
        Returns:
            Output tensor with channel attention applied
        """
        ca = self._global_pool(inputs)
        ca = self._mlp(ca)
        return inputs * ca


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with Channel Attention.
    
    Structure: Conv -> ReLU -> Conv -> CA -> Add
    """
    
    def __init__(self, channels, reduction=16, name='resblock'):
        """Initialize Residual Block.
        
        Args:
            channels: Number of channels
            reduction: Reduction ratio for CA (default: 16)
            name: Layer name
        """
        super().__init__(name=name)
        self._conv1 = tf.keras.layers.Conv2D(channels, 3, padding='same', name='conv1')
        self._relu = tf.keras.layers.ReLU(name='relu')
        self._conv2 = tf.keras.layers.Conv2D(channels, 3, padding='same', name='conv2')
        self._ca = ChannelAttention(channels, reduction=reduction, name='ca')
        self._add = tf.keras.layers.Add(name='add')
        
    def call(self, inputs):
        """Forward pass.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Output tensor
        """
        out = self._conv1(inputs)
        out = self._relu(out)
        out = self._conv2(out)
        out = self._ca(out)
        out = self._add([out, inputs])
        return out


class ResidualGroup(tf.keras.layers.Layer):
    """Residual Group containing multiple Residual Blocks.
    
    This is part of the Residual in Residual (RIR) structure.
    """
    
    def __init__(self, channels, num_blocks, reduction=16, name='rg'):
        """Initialize Residual Group.
        
        Args:
            channels: Number of channels
            num_blocks: Number of residual blocks in the group
            reduction: Reduction ratio for CA (default: 16)
            name: Layer name
        """
        super().__init__(name=name)
        self._blocks = tf.keras.Sequential([
            ResidualBlock(channels, reduction=reduction, name=f'resblock_{i}')
            for i in range(num_blocks)
        ], name='blocks')
        self._conv = tf.keras.layers.Conv2D(channels, 3, padding='same', name='conv')
        self._add = tf.keras.layers.Add(name='add')
        
    def call(self, inputs):
        """Forward pass with long skip connection.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Output tensor
        """
        out = self._blocks(inputs)
        out = self._conv(out)
        out = self._add([out, inputs])
        return out


class UpsampleBlock(tf.keras.layers.Layer):
    """Upsampling block using PixelShuffle (depth_to_space).
    
    For 2x upsampling: Conv -> PixelShuffle
    """
    
    def __init__(self, channels, scale=2, name='upsample'):
        """Initialize Upsample Block.
        
        Args:
            channels: Number of input channels
            scale: Upsampling scale factor (default: 2)
            name: Layer name
        """
        super().__init__(name=name)
        self._scale = scale
        self._conv = tf.keras.layers.Conv2D(channels * scale * scale, 3, padding='same', name='conv')
        
    def call(self, inputs):
        """Upsample input.
        
        Args:
            inputs: Input tensor
        
        Returns:
            Upsampled tensor
        """
        out = self._conv(inputs)
        return tf.nn.depth_to_space(out, block_size=self._scale, name='pixelshuffle')


class RCAN(tf.keras.Model):
    """RCAN (Residual Channel Attention Network) model.
    
    Paper: "Image Super-Resolution Using Very Deep Residual Channel Attention Networks"
    """
    
    def __init__(self, channels=64, num_groups=10, num_blocks=20, reduction=16, scale=2):
        """Initialize RCAN model.
        
        Args:
            channels: Number of feature channels (default: 64)
            num_groups: Number of Residual Groups (default: 10)
            num_blocks: Number of Residual Blocks per group (default: 20)
            reduction: Reduction ratio for Channel Attention (default: 16)
            scale: Super-resolution scale factor (default: 2)
        """
        super().__init__()
        
        self._scale = scale
        
        # Shallow feature extraction
        self._shallow_conv = tf.keras.layers.Conv2D(channels, 3, padding='same', name='shallow_conv')
        
        # Residual in Residual (RIR) structure
        self._rgs = tf.keras.Sequential([
            ResidualGroup(channels, num_blocks, reduction=reduction, name=f'rg_{i}')
            for i in range(num_groups)
        ], name='rgs')
        
        # Long skip connection
        self._long_skip_conv = tf.keras.layers.Conv2D(channels, 3, padding='same', name='long_skip_conv')
        self._long_skip_add = tf.keras.layers.Add(name='long_skip_add')
        
        # Upsampling (assuming scale is power of 2)
        num_upsample = int(np.log2(scale))
        self._upsample = tf.keras.Sequential([
            UpsampleBlock(channels, scale=2, name=f'upsample_{i}')
            for i in range(num_upsample)
        ], name='upsample')
        
        # Output projection
        self._output_conv = tf.keras.layers.Conv2D(3, 3, padding='same', name='output_conv')
        
    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: Input LR image tensor [B, H, W, 3]
            training: Whether in training mode
        
        Returns:
            Super-resolved image tensor [B, H*scale, W*scale, 3]
        """
        # Shallow feature extraction
        shallow_feat = self._shallow_conv(inputs)
        
        # Residual in Residual structure
        deep_feat = self._rgs(shallow_feat)
        
        # Long skip connection
        deep_feat = self._long_skip_conv(deep_feat)
        deep_feat = self._long_skip_add([deep_feat, shallow_feat])
        
        # Upsampling
        up_feat = self._upsample(deep_feat)
        
        # Output projection
        output = self._output_conv(up_feat)
        
        return output
