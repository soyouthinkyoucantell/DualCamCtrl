import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import os
from typing_extensions import Literal


class SimpleAdapterMy(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1):
        super(SimpleAdapterMy, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by a factor of 8
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=8)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * 64, out_dim,
                              kernel_size=kernel_size, stride=stride, padding=(kernel_size-1) // 2)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

        self.patch_embedding = nn.Conv3d(
            out_dim, out_dim, kernel_size=(5, 3, 3), stride=(4, 2, 2), padding=(2, 1, 1)
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        assert h % 16 == 0, "Height must be divisible by 16."
        assert w % 16 == 0, "Width must be divisible by 16."
        assert (f+3) % 4 == 0, "The number of frames + 3 must be divisible by 4."
        zero_padding = torch.zeros(
            (bs, c, 3, h, w), dtype=x.dtype, device=x.device)
        x = torch.cat((x, zero_padding), dim=2)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * (f+3), c, h, w)
        x = x.to(dtype=torch.bfloat16)  # Convert to bfloat16 for efficiency
        # print(f"Input shape after reshaping: {x.shape}")
        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # print(f"Output shape after residual blocks: {out.shape}")
        out = rearrange(out, "(b f) c h w -> b c f h w", b=bs, f=(f+3))

        # print(f"Output shape after rearranging: {out.shape}")
        x = self.patch_embedding(out)
        # print(f"Output shape after patch embedding: {x.shape}")

        # x = rearrange(x, "b c f h w -> b (f h w) c", b=bs, c=out_dim)
        # Reshape to restore original bf dimension
        out = rearrange(x, 'b c f h w -> b (f h w) c', b=bs, f=(f+3)//4)
        # print(f"Output shape after final rearranging: {out.shape}")
        return out


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


if __name__ == "__main__":
    # Example usage
    pose_encoder = SimpleAdapterMy(
        in_dim=6, out_dim=1536, kernel_size=3, stride=1, num_residual_blocks=4)
    print(
        f"Num of parameters: {sum(p.numel() for p in pose_encoder.parameters())}")
    dummy_input = torch.randn(1, 6, 13, 64, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        pose_encoder.to(device)
        dummy_input = dummy_input.to(device)
        features = pose_encoder(dummy_input)
    print(
        f"Num of parameters: {sum(p.numel() for p in pose_encoder.parameters())}")
    # print(f"Output features shape: {[f.shape for f in features]}")

# python -m diffsynth.models.camera_pose_encoder.pose_adaptor
