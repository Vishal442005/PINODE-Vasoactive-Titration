"""
morphomic_cnn.py: Lightweight Depthwise-Separable 1D-CNN with Squeeze-and-Excitation and Self-Attention.

Takes raw 30-second windows (3750 samples at 125 Hz) of normalized ABP and PPG waveforms,
and outputs a dense 32-dimensional vascular resistance embedding: z_vasc(t) in R^32.
Designed for edge hardware (NVIDIA RTX 2050): < 85,000 parameters, < 3 MB memory footprint.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeAndExcitation1D(nn.Module):
    """Channel-wise Squeeze-and-Excitation (SE) attention block for 1D convolutions."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, L)
        b, c, _ = x.shape
        # Squeeze: Global Average Pooling across length L
        y = x.mean(dim=-1)  # (B, C)
        # Excitation: Bottleneck MLP + Sigmoid gating
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-Separable 1D Convolution reducing parameter count and FLOPs by ~70%."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MorphomicWaveformEncoder(nn.Module):
    """
    Lightweight 1D-CNN encoder that compresses raw high-frequency waveforms (ABP + PPG)
    into a continuous vascular resistance embedding vector z_vasc in R^32.
    """

    def __init__(
        self,
        in_channels: int = 2,
        latent_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Initial projection convolution
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),
        )

        # Stage 1: Depthwise Separable Conv (16 -> 32 channels)
        self.stage1 = DepthwiseSeparableConv1d(16, 32, kernel_size=7, stride=2, padding=3)
        self.se1 = SqueezeAndExcitation1D(32, reduction=8)

        # Stage 2: Depthwise Separable Conv (32 -> 64 channels)
        self.stage2 = DepthwiseSeparableConv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.se2 = SqueezeAndExcitation1D(64, reduction=8)

        # Stage 3: Depthwise Separable Conv (64 -> 64 channels)
        self.stage3 = DepthwiseSeparableConv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.se3 = SqueezeAndExcitation1D(64, reduction=8)

        # Multi-Head Temporal Self-Attention Pooling
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True, dropout=dropout)

        # Latent projection head
        self.dropout = nn.Dropout(dropout)
        self.fc_latent = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (Batch, 2, Length) e.g., (B, 2, 3750) for 30s at 125 Hz.

        Returns:
            Vascular resistance embedding z_vasc of shape (Batch, 32).
        """
        # x: (B, 2, L)
        h = self.init_conv(x)  # (B, 16, L/2)
        h = self.se1(self.stage1(h))  # (B, 32, L/4)
        h = self.se2(self.stage2(h))  # (B, 64, L/8)
        h = self.se3(self.stage3(h))  # (B, 64, L/16)

        # Transpose to (Batch, Sequence, Channels) for MultiheadAttention
        h_trans = h.transpose(1, 2)  # (B, L/16, 64)
        attn_out, _ = self.attn(h_trans, h_trans, h_trans)

        # Global average pool over sequence dimension
        pooled = attn_out.mean(dim=1)  # (B, 64)
        pooled = self.dropout(pooled)

        # Project to latent vascular resistance vector z_vasc
        z_vasc = self.fc_latent(pooled)  # (B, 32)
        return z_vasc
