# Copyright (C) 2023 MediaTek Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Define EDSR model."""

import numpy as np
import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block."""

    def __init__(self, filters, name):
        """Initialize."""
        super().__init__(name=name)

        self._feature = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters, 3, padding='same', name='conv0'),
                tf.keras.layers.ReLU(name='relu'),
                tf.keras.layers.Conv2D(filters, 3, padding='same', name='conv1')
            ],
            name='feature'
        )
        self._residual = tf.keras.layers.Add(name='residual')

    def call(self, inputs):  # pylint: disable=arguments-differ
        """Call."""
        return self._residual([self._feature(inputs), inputs])


class UpsampleBlock(tf.keras.layers.Layer):
    """Upsample block."""

    def __init__(self, filters, name):
        """Initialize."""
        super().__init__(name=name)
        self._conv = tf.keras.layers.Conv2D(filters * 4, 3, padding='same', name='conv')

    def call(self, inputs):  # pylint: disable=arguments-differ
        """Call."""
        return tf.nn.depth_to_space(self._conv(inputs), block_size=2, name='d2s')


class EDSR(tf.keras.Model):  # pylint: disable=too-many-ancestors,abstract-method
    """EDSR model."""

    def __init__(self, filters=256, num_blocks=32, scale=2):
        """Initialize.
        
        Args:
            filters: Number of filters (default: 256 for EDSR, 64 for small version)
            num_blocks: Number of residual blocks (default: 32 for EDSR, 16 for small)
            scale: Super-resolution scale factor (default: 2)
        """
        super().__init__()

        self._scale = scale
        self._stem = tf.keras.layers.Conv2D(filters, 3, padding='same', name='stem')
        self._body = tf.keras.Sequential(
            [
                *[ResidualBlock(filters, name=f'residual{idx}') for idx in range(num_blocks)],
                tf.keras.layers.Conv2D(filters, 3, padding='same', name='conv')
            ],
            name='body'
        )
        self._body_residual = tf.keras.layers.Add(name='body_residual')
        
        # Upsampling blocks (assuming scale is power of 2)
        upsample_layers = []
        num_upsample = int(np.log2(scale))
        for i in range(num_upsample):
            upsample_layers.append(UpsampleBlock(filters, name=f'upsample{i}'))
        upsample_layers.append(tf.keras.layers.Conv2D(3, 3, padding='same', name='conv'))
        self._tail = tf.keras.Sequential(upsample_layers, name='tail')

    def call(self, inputs):  # pylint: disable=arguments-differ
        """Call."""
        output = self._stem(inputs)
        output = self._body_residual([self._body(output), output])
        output = self._tail(output)

        return output
