import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import upsample
from .motion_module import TemporalTransformerBlock
import torch.nn.functional as F


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class PoseAdaptor(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        self.pose_encoder = pose_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, pose_embedding):
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]  # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)  # bf c h w
        pose_embedding_features = [
            rearrange(x, "(b f) c h w -> b c f h w", b=bs)
            for x in pose_embedding_features
        ]
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            pose_embedding_features=pose_embedding_features,
        ).sample
        return noise_pred


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True, compress_ratio=1):
        super().__init__()
        ps = ksize // 2

        inter_channel = out_c // compress_ratio
        # if in_c != out_c or sk == False:
        # print(
        #     f"in c, out c, ksize, ps dtype{type(in_c)}, {type(out_c)}, {type(ksize)}, {type(ps)}")
        self.in_conv = nn.Conv2d(in_c, inter_channel, ksize, 1, ps)
        # else:
        #     self.in_conv = None

        self.block1 = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, ksize, 1, ps),
            nn.ReLU(),
            nn.Conv2d(inter_channel, inter_channel, ksize, 1, ps),
        )
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(inter_channel, out_c, ksize, 1, ps)

        if sk == True:
            # TODO
            self.skep = nn.Conv2d(inter_channel, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        # print(
        #     f"x after downsample: {x.shape if self.down else 'not downsampled'}")
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)
        # print(
        #     f"x after in_conv: {x.shape if self.in_conv is not None else 'not applied'}")

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        # print(f"x after block2: {h.shape}")
        if self.skep is not None:
            # print(f"skep {self.skep.in_channels}, {self.skep.out_channels}")
            return h + self.skep(x)
        else:
            return h + x


class DownsampleBlock(nn.Module):

    def __init__(
        self,
        in_c,
        out_c,
    ):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.residual_block = nn.Sequential(
            ResnetBlock(in_c=in_c*4, out_c=out_c, down=False, ksize=3,
                        sk=True, use_conv=True, compress_ratio=2),
            ResnetBlock(
                in_c=out_c, out_c=out_c, down=False, ksize=1, sk=True, use_conv=True, compress_ratio=1
            )
        )

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        x = self.residual_block(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2, ...] = torch.sin(position * div_term)
        pe[0, :, 1::2, ...] = torch.cos(position * div_term)
        pe.unsqueeze_(-1).unsqueeze_(-1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), ...]
        return self.dropout(x)


class CameraPoseEncoder(nn.Module):

    def __init__(
        self,
        downscale_factor=8,
        channels=1536,
        model_embed=1536,
        fuse_blocks=3,
        nums_rb=3,
        cin=1536,
        ksize=1,
        sk=True,
        use_conv=False,
        temporal_attention_nhead=8,
        attention_block_types=("Temporal_Self",),
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=16,
        rescale_output_factor=1.0,
        camera_compress_ratio=2,

    ):
        super(CameraPoseEncoder, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        # Uncomment this
        # -----------------------------------------------------------------
        channels = [
            channels//camera_compress_ratio for _ in range(fuse_blocks)]
        self.channels = channels
        # self.channels.extend([model_embed for _ in range(fuse_blocks)])
        self.fuse_blocks = fuse_blocks
        # -----------------------------------------------------------------
        self.nums_rb = nums_rb
        # self.register_buffer(
        #     "fixed_zero",
        #     # 初始 dtype 和 device 不重要，会在 forward 中自动匹配
        #     torch.zeros(1, 1, 3, 1, 1),
        # )

        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i])
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i])
                elif j == nums_rb - 1:
                    in_dim = int(channels[i])
                    out_dim = channels[i]
                else:
                    in_dim = int(channels[i])
                    out_dim = int(channels[i])

                conv_layer = ResnetBlock(
                    in_dim,
                    out_dim,
                    down=False,
                    ksize=3,
                    sk=sk,
                    use_conv=use_conv,
                )
                temporal_attention_layer = TemporalTransformerBlock(
                    dim=out_dim,
                    num_attention_heads=temporal_attention_nhead,
                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                    attention_block_types=attention_block_types,
                    dropout=0.0,
                    cross_attention_dim=None,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )

                # print(
                #     f"Adding conv layer {j} for block {i} with in_dim: {in_dim}, out_dim: {out_dim}")
                # print(
                #     f"Adding attention layer {j} for block {i} with dim: {out_dim}")
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)

            upsample_block = DownsampleBlock(
                in_c=channels[i], out_c=model_embed
            )
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(
                temporal_attention_layers)
            self.upsample_blocks.append(upsample_block)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        # unshuffle
        # print(f"Input shape: {x.shape}")
        b, c, f, h, w = x.shape
        assert (f+3) % 4 == 0, "The number of (frames plus 3) must be divisible by 4."
        bs = b
        # print(f"x dtype: {x.dtype}, shape: {x.shape}")
        # pad zero in the c channel
        padding = torch.zeros(b, c, 3, h, w, device=x.device, dtype=x.dtype)
        # zero = self.fixed_zero
        # zero = zero.expand(b, c, 3, h, w)  # reshape 展开
        x = torch.cat([x, padding], dim=2)  # b c f+3 h w
        # TODO
        x = x.to(dtype=next(self.parameters()).dtype)  # match dtype

        x = x.view(b, c*4, (f+3)//4, h, w)
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.unshuffle(x)
        # print(f"Unshuffled input shape: {x.shape}")

        # extract features
        features = []
        # print(f"x shape before conv_in: {x.shape}, dtype: {x.dtype}")
        x = self.encoder_conv_in(x)
        # print(f"After conv_in shape: {x.shape}")
        for blockid, (res_block, attention_block, upsample_block) in enumerate(zip(
            self.encoder_down_conv_blocks, self.encoder_down_attention_blocks, self.upsample_blocks
        )):
            # print(
            #     f"Handling block {blockid} with {len(res_block)} resnet layers and {len(attention_block)} attention layers")
            for _inner_layer_idx, (res_layer, attention_layer) in enumerate(zip(res_block, attention_block)):
                # print(
                #     f"Resnet layer: {res_layer}, Attention layer: {attention_layer}")
                # print(f"Handling layer {blockid}.{_inner_layer_idx}")
                x = res_layer(x)
                # print(f"After resnet shape: {x.shape}")
                h, w = x.shape[-2:]
                x = rearrange(x, "(b f) c h w -> (b h w) f c", b=bs)
                x = attention_layer(x)
                x = rearrange(x, "(b h w) f c -> (b f) c h w", h=h, w=w)
                # print(f"After attention shape: {x.shape}")
            # print(f"umsample block {upsample_block}")
            # print(f"Before upsample block {blockid} shape: {x.shape}")
            upsampled_x = upsample_block(x)
            # print(f"After upsample block {blockid} shape: {upsampled_x.shape}")

            out = rearrange(upsampled_x, "(b f) c h w -> b (f h w) c", b=bs)
            features.append(out)
            # print(f"Feature shape {out.shape}")
        # print(f"Features shape: {[f.shape for f in features]}")
        return features[-self.fuse_blocks:]


if __name__ == "__main__":
    # Example usage
    pose_encoder = CameraPoseEncoder()
    # Batch size of 2, 3 channels, 16 frames, 64x64 resolution
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