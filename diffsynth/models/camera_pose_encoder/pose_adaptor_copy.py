
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor

from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers, UNet2DConditionLoadersMixin

from diffusers import UNet3DConditionModel

import math
import torch
import torch.nn as nn
from einops import rearrange
from .motion_module import TemporalTransformerBlock


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v)
                      for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


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
        bs = pose_embedding.shape[0]            # b c f h w
        pose_embedding_features = self.pose_encoder(
            pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]
        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               pose_embedding_features=pose_embedding_features).sample
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
            self.op = conv_nd(dims, self.channels, self.out_channels,
                              3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.,
            max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2, ...] = torch.sin(position * div_term)
        pe[0, :, 1::2, ...] = torch.cos(position * div_term)
        pe.unsqueeze_(-1).unsqueeze_(-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), ...]
        return self.dropout(x)


class CameraPoseEncoder(nn.Module):

    def __init__(self,
                 downscale_factor,
                 channels=[320, 640, 1280, 1280],
                 nums_rb=3,
                 cin=384,
                 ksize=1,
                 sk=True,
                 use_conv=False,
                 compression_factor=1,
                 temporal_attention_nhead=8,
                 attention_block_types=("Temporal_Self", ),
                 temporal_position_encoding=False,
                 temporal_position_encoding_max_len=16,
                 rescale_output_factor=1.0):
        super(CameraPoseEncoder, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(
                        in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(
                        in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(
                        in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(
                        in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                temporal_attention_layer = TemporalTransformerBlock(dim=out_dim,
                                                                    num_attention_heads=temporal_attention_nhead,
                                                                    attention_head_dim=int(
                                                                        out_dim / temporal_attention_nhead),
                                                                    attention_block_types=attention_block_types,
                                                                    dropout=0.0,
                                                                    cross_attention_dim=None,
                                                                    temporal_position_encoding=temporal_position_encoding,
                                                                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                                                                    rescale_output_factor=rescale_output_factor)
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(
                temporal_attention_layers)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        # unshuffle
        bs = x.shape[0]
        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.encoder_conv_in(x)
        for res_block, attention_block in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks):
            for res_layer, attention_layer in zip(res_block, attention_block):
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, '(b f) c h w -> (b h w) f c', b=bs)
                x = attention_layer(x)
                x = rearrange(x, '(b h w) f c -> (b f) c h w', h=h, w=w)
            features.append(x)
        return features



class UNet3DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            center_input_sample: bool = False,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",
            ),
            mid_block_type: str = "UNetMidBlock3DCrossAttn",
            up_block_types: Tuple[str] = (
                    "UpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D",
            ),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: int = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            attention_head_dim: Union[int, Tuple[int]] = 8,
            dual_cross_attention: bool = False,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            addition_embed_type: Optional[str] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",

            # Additional
            use_motion_module=False,
            motion_module_resolutions=(1, 2, 4, 8),
            motion_module_mid_block=False,
            motion_module_type=None,
            motion_module_kwargs={},

            # whether fuse first frame's feature
            fuse_first_frame: bool = False,
    ):
        super().__init__()
        self.logger = logging.get_logger(__name__)

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        self.down_fusers = nn.ModuleList([])
        self.mid_fuser = None
        self.down_fusers.append(
            FusionBlock2D(
                in_channels=block_out_channels[0],
                out_channels=block_out_channels[0],
                temb_channels=time_embed_dim,
                eps=norm_eps,
                groups=norm_num_groups,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=act_fn,
            ) if fuse_first_frame else None
        )

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2 ** i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )

            down_fuser = nn.ModuleList(
                [
                    FusionBlock2D(
                        in_channels=output_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        eps=norm_eps,
                        groups=norm_num_groups,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=act_fn,
                    ) if fuse_first_frame else None for _ in
                    range(layers_per_block if is_final_block else layers_per_block + 1)
                ]
            )

            self.down_blocks.append(down_block)
            self.down_fusers.append(down_fuser)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,

                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        self.mid_fuser = FusionBlock2D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            eps=norm_eps,
            groups=norm_num_groups,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=act_fn,
        ) if fuse_first_frame else None

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,

                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def set_image_layer_lora(self, image_layer_lora_rank: int = 128):
        lora_attn_procs = {}
        for name in self.attn_processors.keys():
            self.logger.info(f"(add lora) {name}")
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=image_layer_lora_rank if image_layer_lora_rank > 16 else hidden_size // image_layer_lora_rank,
            )
        self.set_attn_processor(lora_attn_procs)

        lora_layers = AttnProcsLayers(self.attn_processors)
        self.logger.info(f"(lora parameters): {sum(p.numel() for p in lora_layers.parameters()) / 1e6:.3f} M")
        del lora_layers

    def set_image_layer_lora_scale(self, lora_scale: float = 1.0):
        for block in self.down_blocks: setattr(block, "lora_scale", lora_scale)
        for block in self.up_blocks:   setattr(block, "lora_scale", lora_scale)
        setattr(self.mid_block, "lora_scale", lora_scale)

    def set_motion_module_lora_scale(self, lora_scale: float = 1.0):
        for block in self.down_blocks: setattr(block, "motion_lora_scale", lora_scale)
        for block in self.up_blocks: setattr(block, "motion_lora_scale", lora_scale)
        setattr(self.mid_block, "motion_lora_scale", lora_scale)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # filter out processors in motion module
            if hasattr(module, "set_processor"):
                if not "motion_modules." in name:
                    processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not "motion_modules." in name:
                    if not isinstance(processor, dict):
                        module.set_processor(processor)
                    else:
                        module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_motion_module_lora_layers(self, motion_module_lora_rank: int = 32):
        lora_attn_procs = {}
        for name in self.mm_attn_processors.keys():
            self.logger.info(f"(add lora) {name}")
            cross_attention_dim = None
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=motion_module_lora_rank if motion_module_lora_rank > 16 else hidden_size // motion_module_lora_rank,
            )
        self.set_mm_attn_processor(lora_attn_procs)

        lora_layers = AttnProcsLayers(self.mm_attn_processors)
        return lora_layers

    @property
    def mm_attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module,
                                        processors: Dict[str, AttentionProcessor]):
            # filter out processors in motion module
            if hasattr(module, "set_processor"):
                if "motion_modules." in name:
                    processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_mm_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.mm_attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if "motion_modules." in name:
                    if not isinstance(processor, dict):
                        module.set_processor(processor)
                    else:
                        module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
            class_labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,

            # support controlnet
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,

            # other features
            motion_module_alphas: Union[tuple, float] = 1.0,
            debug: bool = False,
    ) -> Union[UNet3DConditionOutput, Tuple]:

        activations = {}

        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            self.logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary1
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # extend encoder_hidden_states
        video_length = sample.shape[2]
        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b f) n c", f=video_length)

        # emb_single = emb
        # emb = repeat(emb, "b c -> (b f) c", f=video_length)

        # pre-process
        sample = self.conv_in(sample)
        activations["conv_in_out"] = sample

        # to be fused
        if self.down_fusers[0] != None:
            # scale, shift   = self.down_fusers[0](sample[:,:,0].contiguous(), emb_single).unsqueeze(2).chunk(2, dim=1)
            # sample[:,:,1:] = (1 + scale) * sample[:,:,1:].contiguous() + shift
            fused_sample = self.down_fusers[0](
                init_hidden_state=sample[:, :, :1].contiguous(),
                post_hidden_states=sample[:, :, 1:].contiguous(),
                temb=emb_single,
            )
            sample = torch.cat([sample[:, :, :1], fused_sample], dim=2)

        activations["conv_in_fuse_out"] = sample

        # down
        down_block_res_samples = (sample,)

        # motion module alpha
        if isinstance(motion_module_alphas, float):
            motion_module_alphas = (motion_module_alphas,) * 5

        for downsample_block, down_fuser, motion_module_alpha in zip(self.down_blocks, self.down_fusers[1:],
                                                                     motion_module_alphas[:-1]):
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                motion_module_alpha=motion_module_alpha,
                cross_attention_kwargs=cross_attention_kwargs
            )
            # to be fused
            for sample_idx, fuser in enumerate(down_fuser):
                if fuser != None:
                    fused_sample = fuser(
                        init_hidden_state=res_samples[sample_idx][:, :, :1].contiguous(),
                        post_hidden_states=res_samples[sample_idx][:, :, 1:].contiguous(),
                        temb=emb_single,
                    )
                    res_samples = list(res_samples)
                    res_samples[sample_idx] = torch.cat([res_samples[sample_idx][:, :, :1], fused_sample], dim=2)
                    res_samples = tuple(res_samples)

            down_block_res_samples += res_samples

        # support controlnet
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                if len(down_block_additional_residual.shape) == 4:
                    # b c h w
                    # if input single condition, apply it to all frames
                    down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
                    # boardcast will solve the problem
                    # down_block_additional_residual = repeat(down_block_additional_residual, "b c f h w -> b c (f n) h w", n=video_length)

                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask,
            motion_module_alpha=motion_module_alphas[-1], cross_attention_kwargs=cross_attention_kwargs
        )

        # mid block fuser
        if self.mid_fuser != None:
            fused_sample = self.mid_fuser(
                init_hidden_state=sample[:, :, :1],
                post_hidden_states=sample[:, :, 1:],
                temb=emb_single,
            )
            sample = torch.cat([sample[:, :, :1], fused_sample], dim=2)

        # support controlnet
        if mid_block_additional_residual is not None:
            if len(mid_block_additional_residual.shape) == 4:
                mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
                # boardcast will solve this problemq
                # mid_block_additional_residual = repeat(mid_block_additional_residual, "b c f h w -> b c (f n) h w", n=video_length)

            sample = sample + mid_block_additional_residual

        # up
        for i, (upsample_block, motion_module_alpha) in enumerate(zip(self.up_blocks, motion_module_alphas[:-1][::-1])):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    motion_module_alpha=motion_module_alpha,
                    cross_attention_kwargs=cross_attention_kwargs
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size,
                    encoder_hidden_states=encoder_hidden_states, motion_module_alpha=motion_module_alpha,
                    cross_attention_kwargs=cross_attention_kwargs
                )
        activations["upblocks_out"] = sample

        # post-process
        # frame-wise normalization
        sample = rearrange(sample, "b c f h w -> (b f) c h w")
        sample = self.conv_norm_out(sample)
        sample = rearrange(sample, "(b f) c h w -> b c f h w", f=video_length)

        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if (not return_dict):
            return (sample,)
        elif debug:
            return UNet3DConditionOutput(sample=sample), activations
        else:
            return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None, logger=None):
        if logger is not None:
            logger.info(f"Loading unet's pretrained weights from {pretrained_model_path} ...")

        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")

        with open(config_file, "r") as f:
            config = json.load(f)

        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        from diffusers.utils import SAFETENSORS_WEIGHTS_NAME

        model, unused_kwargs = cls.from_config(config, return_unused_kwargs=True, **unet_additional_kwargs)
        if logger is not None:
            logger.info(f"please check unused kwargs in 'unet_additional_kwargs' config:")
        for k, v in unused_kwargs.items():
            if logger is not None:
                logger.info(f"{k:50s}: {repr(v)}")

        model_file = os.path.join(pretrained_model_path, SAFETENSORS_WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")

        state_dict = safetensors.torch.load_file(model_file, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if logger is not None:
            logger.info(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)};")
        assert len(unexpected) == 0

        params = [p.numel() if "motion_modules." in n else 0 for n, p in model.named_parameters()]
        if logger is not None:
            logger.info(f"Motion module parameters: {sum(params) / 1e6} M")

        return model





if __name__ == "__main__":
    # Example usage
    pose_encoder = CameraPoseEncoder(downscale_factor=8)
    print(f"Parameter {sum(p.numel() for p in pose_encoder.parameters())}")
    from diffusers import StableVideoDiffusionPipeline


    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path="runwayml/stable-video-diffusion-2-1",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": True,
            "motion_module_type": "pose_adaptor_copy",
            "motion_module_kwargs": {
                "downscale_factor": 8,
            }
        }
    )