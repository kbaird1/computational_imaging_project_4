"""
unet_model.py
================
Defines a U-Net architecture for single-pixel image reconstruction in PyTorch.

This model maps a low-quality input image X (1×64×64) to a reconstructed
image Ŷ (1×64×64). The architecture uses an encoder–decoder structure
with skip connections between corresponding layers.

You can adjust the base number of features, depth, and activation
functions to experiment with model capacity.

References:
    - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
"""

import torch
import torch.nn as nn


# ============================================================
# 1. Helper: Double Convolution Block (Conv → BN → ReLU → Conv → BN → ReLU)
# ============================================================
class DoubleConv(nn.Module):
    """
    A helper module that performs two convolutional operations with
    batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# ============================================================
# 2. UNet Class Definition
# ============================================================
class UNet(nn.Module):
    """
    Standard U-Net architecture with encoder–decoder and skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super().__init__()
        features = init_features

        # -------- Encoder --------
        self.enc1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------- Bottleneck --------
        self.bottleneck = DoubleConv(features * 8, features * 16)

        # -------- Decoder --------
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features * 2, features)

        # -------- Output --------
        self.conv_out = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder path with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.conv_out(d1))  # constrain outputs to [0, 1]


# ============================================================
# 3. Factory Function (for easy import in main.py)
# ============================================================
def build_unet(in_channels=1, out_channels=1, init_features=64):
    """
    Factory function to build a U-Net model instance.
    """
    model = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
    return model


# # ============================================================
# # 4. Sanity Test
# # ============================================================
# if __name__ == "__main__":
#     print("Testing U-Net model architecture...")
#     x = torch.randn((1, 1, 64, 64))  # batch of 1 grayscale image
#     model = build_unet(in_channels=1, out_channels=1)
#     y = model(x)
#     print(f"Input shape:  {x.shape}")
#     print(f"Output shape: {y.shape}")
#     print("Model test passed ")
