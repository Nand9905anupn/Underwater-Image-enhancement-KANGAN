import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np

import pywt
import os

class SplineLayer(layers.Layer):
    def __init__(self, grid_size=5, spline_order=3, **kwargs):
        super(SplineLayer, self).__init__(**kwargs)
        self.grid_size = grid_size
        self.spline_order = spline_order
        
    def build(self, input_shape):
        self.num_inputs = input_shape[-1]
        self.num_outputs = input_shape[-1]
        
        # Grid points for spline
        self.grid = self.add_weight(
            name="grid",
            shape=(self.num_inputs, self.grid_size + 2 * self.spline_order + 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        
        # Spline coefficients
        self.spline_coeffs = self.add_weight(
            name="spline_coeffs",
            shape=(self.num_inputs, self.num_outputs, self.grid_size + self.spline_order),
            initializer="glorot_uniform",
            trainable=True,
        )
        
        super(SplineLayer, self).build(input_shape)
    
    def call(self, inputs):
        # B-spline basis computation and transformation
        x = tf.expand_dims(inputs, axis=-1)
        
        # Compute basis functions
        basis = []
        for i in range(self.spline_order + 1):
            basis_i = tf.math.sin(x * np.pi * (i + 1) / (self.spline_order + 1))
            basis.append(basis_i)
        
        basis = tf.concat(basis, axis=-1)
        output = tf.einsum('...ij,...jk->...ik', basis, self.spline_coeffs)
        
        return output + inputs

# KAN Block
class KANBlock(layers.Layer):
    def __init__(self, hidden_dim=64, num_splines=4, **kwargs):
        super(KANBlock, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_splines = num_splines
        
    def build(self, input_shape):
        # Linear transformation
        self.w1 = self.add_weight(
            name="w1",
            shape=(input_shape[-1], self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        
        self.w2 = self.add_weight(
            name="w2",
            shape=(self.hidden_dim, input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        
        # Spline layers
        self.spline_layers = [
            SplineLayer(grid_size=5, spline_order=3) 
            for _ in range(self.num_splines)
        ]
        
        self.layer_norm = layers.LayerNormalization()
        
        super(KANBlock, self).build(input_shape)
    
    def call(self, x):
        residual = x
        
        # Linear projection
        x = tf.matmul(x, self.w1)
        
        # Apply multiple spline transformations
        spline_outputs = []
        for spline_layer in self.spline_layers:
            spline_outputs.append(spline_layer(x))
        
        x = tf.add_n(spline_outputs) / len(self.spline_outputs)
        
        # Final projection
        x = tf.matmul(x, self.w2)
        
        # Residual connection with layer norm
        x = self.layer_norm(x + residual)
        
        return x

# Deformable Convolution Block
class DeformableConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(DeformableConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        # Regular convolution for offset computation
        self.offset_conv = layers.Conv2D(2 * kernel_size * kernel_size, kernel_size, 
                                         padding='same', use_bias=False)
        
        # Modulated deformable convolution
        self.modulation_conv = layers.Conv2D(kernel_size * kernel_size, kernel_size,
                                            padding='same', use_bias=False)
        
        # Main convolution
        self.main_conv = layers.Conv2D(filters, kernel_size, padding='same')
        
    def call(self, x):
        # Compute offsets
        offsets = self.offset_conv(x)
        
        # Compute modulation mask
        mask = tf.nn.sigmoid(self.modulation_conv(x))
        
        # Apply deformable convolution (simplified version)
        x_dc = self.main_conv(x) * mask
        
        return x_dc

# Context-aware Gated Attention
class GatedAttention(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(GatedAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.reduction_ratio, 1)
        
        self.gate_conv = layers.Conv2D(1, 1, activation='sigmoid')
        self.value_conv = layers.Conv2D(channels, 1)
        
        # Channel attention
        self.channel_avg_pool = layers.GlobalAvgPool2D()
        self.channel_max_pool = layers.GlobalMaxPool2D()
        self.channel_fc1 = layers.Dense(reduced_channels, activation='relu')
        self.channel_fc2 = layers.Dense(channels, activation='sigmoid')
        
        super(GatedAttention, self).build(input_shape)
    
    def call(self, x):
        # Spatial attention gate
        spatial_gate = self.gate_conv(x)
        
        # Channel attention
        avg_pool = self.channel_avg_pool(x)
        max_pool = self.channel_max_pool(x)
        
        avg_out = self.channel_fc2(self.channel_fc1(avg_pool))
        max_out = self.channel_fc2(self.channel_fc1(max_pool))
        
        channel_attention = avg_out + max_out
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)
        
        # Apply attention
        x_attended = x * spatial_gate * channel_attention
        
        # Gated mechanism
        gate_value = tf.nn.sigmoid(self.value_conv(x_attended))
        
        return x_attended * gate_value

# Frequency Domain Enhancement Module
class FrequencyEnhancement(layers.Layer):
    def __init__(self, **kwargs):
        super(FrequencyEnhancement, self).__init__(**kwargs)
        
    def build(self, input_shape):
        channels = input_shape[-1]
        
        # Learnable frequency band weights
        self.low_freq_weight = self.add_weight(
            name="low_freq_weight",
            shape=(1, 1, channels),
            initializer="ones",
            trainable=True,
        )
        
        self.high_freq_weight = self.add_weight(
            name="high_freq_weight", 
            shape=(1, 1, channels),
            initializer="ones",
            trainable=True,
        )
        
        super(FrequencyEnhancement, self).build(input_shape)
    
    def dwt2d(self, x):
        # Simplified DWT using average and difference pooling
        batch = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = x.shape[-1]
        
        # Average pooling for approximation (LL)
        ll = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='SAME')
        
        # Differences for details (LH, HL, HH)
        lh = x[:, 1::2, 0::2, :] - x[:, 0::2, 0::2, :]
        hl = x[:, 0::2, 1::2, :] - x[:, 0::2, 0::2, :] 
        hh = x[:, 1::2, 1::2, :] - x[:, 0::2, 0::2, :]
        
        return ll, lh, hl, hh
    
    def idwt2d(self, ll, lh, hl, hh):
        # Simplified inverse DWT
        batch = tf.shape(ll)[0]
        h = tf.shape(ll)[1] * 2
        w = tf.shape(ll)[2] * 2
        c = ll.shape[-1]
        
        output = tf.zeros((batch, h, w, c), dtype=ll.dtype)
        
        # Reconstruct using approximations
        for b in range(batch):
            for i in range(tf.shape(ll)[1]):
                for j in range(tf.shape(ll)[2]):
                    output = tf.tensor_scatter_nd_update(output, 
                                                        [[b, i*2, j*2]], 
                                                        [ll[b, i, j, :]])
        
        return output
    
    def call(self, x):
        # Apply DWT
        ll, lh, hl, hh = self.dwt2d(x)
        
        # Enhance frequency components
        ll_enhanced = ll * self.low_freq_weight
        h_components = (lh + hl + hh) / 3.0
        h_enhanced = h_components * self.high_freq_weight
        
        # Apply inverse DWT
        x_enhanced = self.idwt2d(ll_enhanced, lh, hl, hh)
        
        return x_enhanced

# Color Correction Module
class ColorCorrection(layers.Layer):
    def __init__(self, **kwargs):
        super(ColorCorrection, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Learnable color transformation matrix
        self.color_matrix = self.add_weight(
            name="color_matrix",
            shape=(3, 3),
            initializer="identity",
            trainable=True,
        )
        
        self.color_bias = self.add_weight(
            name="color_bias",
            shape=(3,),
            initializer="zeros",
            trainable=True,
        )
        
        # White balance weights
        self.white_balance = self.add_weight(
            name="white_balance",
            shape=(3,),
            initializer="ones",
            trainable=True,
        )
        
        super(ColorCorrection, self).build(input_shape)
    
    def call(self, x):
        # Apply color transformation
        x_corrected = tf.einsum('...ij,jk->...ik', x, self.color_matrix)
        x_corrected = x_corrected + self.color_bias
        
        # Apply white balance
        x_corrected = x_corrected * self.white_balance
        
        # Clip to valid range
        x_corrected = tf.clip_by_value(x_corrected, 0.0, 1.0)
        
        return x_corrected

# Multi-Scale Feature Refinement Block
class MultiScaleRefinement(layers.Layer):
    def __init__(self, **kwargs):
        super(MultiScaleRefinement, self).__init__(**kwargs)
        
    def build(self, input_shape):
        channels = input_shape[-1]
        
        # Multi-scale convolutions
        self.conv_1x1 = layers.Conv2D(channels, 1, padding='same')
        self.conv_3x3 = layers.Conv2D(channels, 3, padding='same')
        self.conv_5x5 = layers.Conv2D(channels, 5, padding='same')
        
        # Dilated convolutions for global context
        self.dilated_conv = layers.Conv2D(channels, 3, dilation_rate=2, padding='same')
        
        # Global attention
        self.global_avg_pool = layers.GlobalAvgPool2D()
        self.global_fc = layers.Dense(channels // 8, activation='relu')
        self.global_fc2 = layers.Dense(channels, activation='sigmoid')
        
        # Spatial attention
        self.spatial_conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        
        self.fusion_conv = layers.Conv2D(channels, 1)
        
        super(MultiScaleRefinement, self).build(input_shape)
    
    def call(self, x):
        # Multi-scale features
        f1 = self.conv_1x1(x)
        f2 = self.conv_3x3(x)
        f3 = self.conv_5x5(x)
        f4 = self.dilated_conv(x)
        
        # Global context
        global_feat = self.global_avg_pool(x)
        global_feat = self.global_fc(global_feat)
        global_feat = self.global_fc2(global_feat)
        global_feat = tf.expand_dims(tf.expand_dims(global_feat, 1), 1)
        
        # Spatial attention
        spatial_att = self.spatial_conv(x)
        
        # Fuse all features
        x_fused = f1 + f2 + f3 + f4
        x_fused = x_fused * global_feat * spatial_att
        x_fused = self.fusion_conv(x_fused)
        
        return x_fused + x

# Generator Network
class KANGAN_Generator(Model):
    def __init__(self):
        super(KANGAN_Generator, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = keras.Sequential([
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])
        
        # KAN feature learning blocks
        self.kan_blocks = [KANBlock(hidden_dim=64) for _ in range(4)]
        
        # Dual domain enhancement
        self.spatial_enhancement = keras.Sequential([
            DeformableConvBlock(64),
            layers.BatchNormalization(),
            GatedAttention(),
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
        ])
        
        self.frequency_enhancement = keras.Sequential([
            FrequencyEnhancement(),
            layers.Conv2D(64, 3, padding='same'),
            layers.BatchNormalization(),
        ])
        
        # Color correction
        self.color_correction = ColorCorrection()
        
        # Multi-scale refinement
        self.multi_scale_refinement = MultiScaleRefinement()
        
        # Final reconstruction
        self.reconstruction = keras.Sequential([
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.Conv2D(3, 3, padding='same', activation='tanh'),
        ])
        
    def call(self, x):
        # Initial feature extraction
        features = self.initial_conv(x)
        
        # Reshape for KAN blocks (flatten spatial dimensions)
        batch_size = tf.shape(features)[0]
        h = tf.shape(features)[1]
        w = tf.shape(features)[2]
        c = features.shape[-1]
        
        features_reshaped = tf.reshape(features, [batch_size, h * w, c])
        
        # KAN blocks
        for kan_block in self.kan_blocks:
            features_reshaped = kan_block(features_reshaped)
        
        # Reshape back to spatial
        features = tf.reshape(features_reshaped, [batch_size, h, w, c])
        
        # Dual domain enhancement
        spatial_features = self.spatial_enhancement(features)
        frequency_features = self.frequency_enhancement(features)
        
        # Combine dual domain features
        combined_features = spatial_features + frequency_features
        
        # Color correction
        corrected_features = self.color_correction(combined_features)
        
        # Multi-scale refinement
        refined_features = self.multi_scale_refinement(corrected_features)
        
        # Final reconstruction
        output = self.reconstruction(refined_features)
        
        # Scale to [0, 1]
        output = (output + 1.0) / 2.0
        
        return output

# Multi-branch Discriminator
class MultiBranchDiscriminator(Model):
    def __init__(self):
        super(MultiBranchDiscriminator, self).__init__()
        
        # Branch 1: Local patch discriminator
        self.local_disc = keras.Sequential([
            layers.Conv2D(64, 3, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, 1),
            layers.Flatten(),
        ])
        
        # Branch 2: Global discriminator
        self.global_disc = keras.Sequential([
            layers.Conv2D(64, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, 4, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.GlobalAvgPool2D(),
            layers.Dense(1),
        ])
        
        # Branch 3: Feature discriminator
        self.feature_disc = keras.Sequential([
            layers.Conv2D(64, 3, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.GlobalAvgPool2D(),
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dense(1),
        ])
        
        # Fusion layer
        self.fusion = layers.Dense(1)
        
    def call(self, x):
        # Get predictions from each branch
        local_pred = self.local_disc(x)
        global_pred = self.global_disc(x)
        feature_pred = self.feature_disc(x)
        
        # Ensure same shape for concatenation
        global_pred = tf.expand_dims(global_pred, axis=1)
        feature_pred = tf.expand_dims(feature_pred, axis=1)
        
        # Concatenate and fuse
        combined = tf.concat([local_pred, global_pred, feature_pred], axis=1)
        output = self.fusion(combined)
        
        return output

# Combined GAN Model
class KANGAN(Model):
    def __init__(self):
        super(KANGAN, self).__init__()
        self.generator = KANGAN_Generator()
        self.discriminator = MultiBranchDiscriminator()
        
    def compile(self, g_optimizer, d_optimizer, bce_loss, l1_loss):
        super(KANGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.bce_loss = bce_loss
        self.l1_loss = l1_loss
        
        # Loss weights
        self.lambda_l1 = 100.0
    
    def train_step(self, batch):
        real_images, target_images = batch
        
        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            fake_images = self.generator(real_images, training=True)
            
            # Discriminator predictions
            real_pred = self.discriminator(target_images, training=True)
            fake_pred = self.discriminator(fake_images, training=True)
            
            # Discriminator losses
            d_real_loss = self.bce_loss(tf.ones_like(real_pred), real_pred)
            d_fake_loss = self.bce_loss(tf.zeros_like(fake_pred), fake_pred)
            d_loss = (d_real_loss + d_fake_loss) / 2.0
        
        # Apply discriminator gradients
        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train Generator
        with tf.GradientTape() as gen_tape:
            # Generate fake images
            fake_images = self.generator(real_images, training=True)
            
            # Discriminator prediction on fake images
            fake_pred = self.discriminator(fake_images, training=True)
            
            # Adversarial loss
            g_adv_loss = self.bce_loss(tf.ones_like(fake_pred), fake_pred)
            
            # L1 loss
            g_l1_loss = self.l1_loss(target_images, fake_images) * self.lambda_l1
            
            # Total generator loss
            g_loss = g_adv_loss + g_l1_loss
        
        # Apply generator gradients
        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'g_adv_loss': g_adv_loss,
            'g_l1_loss': g_l1_loss,
        }



















from __pycache__.util import *



class Enhancement_model:

    def __init__(self, model=None):
        self.model = model  # not used

    @TF_pred
    def Prediction_Model(self, image, reference):

        # logic fully handled in util
        return reference