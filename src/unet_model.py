"""
unet_model.py
===============
Flexible U-Net implementation for single-pixel imaging reconstruction.

This version allows the model to be configured directly from a dictionary
of hyperparameters passed from main.py (no config import needed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Helper: Double Convolution Block (Conv → BN → Act → Dropout)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation="ReLU",
                 batch_norm=True, dropout_rate=0.0, kernel_size=3, padding_mode="zeros"):
        super().__init__()

        # Dynamically select activation
        act_layer = getattr(nn, activation, nn.ReLU)

        padding = kernel_size // 2  # same padding
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode, bias=False)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_layer(inplace=True))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                padding=padding, padding_mode=padding_mode, bias=False))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_layer(inplace=True))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


# ============================================================
# 2. UNet Class Definition
# ============================================================
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64,
                 depth=4, activation="ReLU", batch_norm=True,
                 dropout_rate=0.0, kernel_size=3, padding_mode="zeros",
                 output_activation="Sigmoid"):
        super().__init__()

        self.depth = depth
        self.output_activation = output_activation

        features = init_features

        # --- Encoder Path ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(DoubleConv(in_ch, features, activation, batch_norm,
                                            dropout_rate, kernel_size, padding_mode))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = features
            features *= 2

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features // 2, features, activation, batch_norm,
                                     dropout_rate, kernel_size, padding_mode)

        # --- Decoder Path ---
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(features, features // 2, activation, batch_norm,
                                            dropout_rate, kernel_size, padding_mode))
            features //= 2

        # --- Output Layer ---
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []

        # Encoder
        for i in range(self.depth):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.depth - 1, -1, -1):
            x = self.upconvs[self.depth - 1 - i](x)
            x = torch.cat((x, enc_features[i]), dim=1)
            x = self.decoders[self.depth - 1 - i](x)

        # Output
        x = self.conv_out(x)

        if self.output_activation == "Sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation == "Tanh":
            return torch.tanh(x)
        else:
            return x


# ============================================================
# 3. Factory Function for easy import in main.py
# ============================================================
def build_unet(**kwargs):
    """
    Build a UNet model instance using configuration parameters.
    Example:
        model = build_unet(
            in_channels=1, out_channels=1, init_features=64,
            depth=4, activation='ReLU', dropout_rate=0.1,
            kernel_size=3, padding_mode='zeros', output_activation='Sigmoid'
        )
    """
    model = UNet(**kwargs)
    return model
