
from accelerate.utils import set_seed
from examples.dataset.colmap_alter import ScenesDataset_Alter
import imageio
import os
import torch
import warnings
import torchvision
import argparse
import json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator, accelerator, DeepSpeedPlugin
from diffsynth import save_video
from safetensors.torch import load_file
import torch
import os
import json
from diffsynth.pipelines.wan_video_new_altered import WanVideoPipeline, ModelConfig
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

scene_dataset = ScenesDataset_Alter(
    split="test", ratio=0.001, patch_size=[720, 480])
print(f"Scenes dataset length: {len(scene_dataset)}")

# epoch = 1
# step = 1199
# weight_path = rf'/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/models/train/Wan2.1-Fun-V1.1-1.3B-InP_full/epoch-{epoch}-step-{step}.safetensors'

# state_dict = load_file(weight_path, device="cpu")
# print(f"state_dict keys: {list(state_dict.keys())}")


def get_data(data):
    if data is None:
        return None
    else:
        input_data = {
            'images': data["images"],
            "height": data["images"][0].shape[-2],
            "width": data["images"][0].shape[-1],
            'input_video': data.get("images", None),
            "num_frames": data["images"].shape[0],
            'input_image': data.get("images")[:1],
            'extra_images': data.get("extra_images", None),
            'extra_image_frame_index': data.get("extra_image_frame_index", None),
            'control_video': data.get("control", None),
            'prompt': data.get("prompt", ''),
            'plucker_embedding': data.get("camera_infos", None),
        }
        return input_data


config_path = '/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/model_config/controlnet_config.yaml'
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="Wan2.1_VAE.pth",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-V1.1-1.3B-InP",
            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            # models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
            offload_device="cpu",
        ),

    ],
    config_path=config_path,
)
res_dir = '/hpc2hdd/home/hongfeizhang/experiments/Wan/2025-07-29_14-24_1e-5'
pipe.dit = load_state_dict_from_zero_checkpoint(
    pipe.dit, res_dir)

device = "cuda"
pipe.to(device)
save_root = os.path.join(res_dir, "test_videos")
sample_step = 1
frame_length = 21
validate_step = 1000

for idx in range(len(scene_dataset)):
    data = scene_dataset.__getitem__(
        idx=idx,
        sample_step=sample_step,
        frame_length=frame_length,
        validate_step=validate_step,
    )

    input_data = get_data(data)
    for key, value in input_data.items():
        if isinstance(value, torch.Tensor):
            input_data[key] = value.to(device)
        elif isinstance(value, list):
            input_data[key] = [v.to(device) if isinstance(
                v, torch.Tensor) else v for v in value]
    # save_path = f"epoch_{epoch}_step_{step}"
    save_path = f"test_{idx}_step_{sample_step}_frame_length_{frame_length}_validate_{validate_step}"
    save_path = os.path.join(save_root, save_path)
    os.makedirs(save_path, exist_ok=True)
    # 生成输入可视化视频：input_image + extra_images
    gt_video = input_data['images'].unsqueeze(0).to(device)
    gt_video = gt_video.permute(0, 1, 3, 4, 2)  # B T H W C
    print(f"GT shape: {gt_video.shape}")
    input_frames = [None] * input_data['num_frames']
    # 插入第0帧
    input_frames[0] = input_data['input_image']
    # 插入extra frames到指定位置
    for img, frameidx in zip(input_data['extra_images'], input_data['extra_image_frame_index']):
        if 0 <= frameidx < len(input_frames):
            input_frames[frameidx] = img.unsqueeze(0)
    print(f"extra frame number: {(input_data['extra_image_frame_index'])}")
    # 将空帧填充成最后一个有效帧（或黑图）
    last_valid = torch.zeros_like(input_frames[0])
    for i in range(len(input_frames)):
        if input_frames[i] is None:
            input_frames[i] = last_valid
    # print(f"first_input_frame shape: {input_frames[0].shape}")
    input_frames = torch.stack(input_frames, dim=0)
    # print(f"input_frames shape: {input_frames.shape}")
    input_frames = input_frames.permute(1, 0, 3, 4, 2)  # B T H W C
    input_frames = pipe.vae_output_to_video(
        input_frames, pattern='B T H W C', min_value=0, max_value=1)
    gt_video = pipe.vae_output_to_video(
        gt_video, pattern='B T H W C', min_value=0, max_value=1)
    video = pipe(
        prompt="",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        # input_video = input_data['input_video'],
        input_image=input_data['input_image'],
        extra_images=input_data['extra_images'],
        extra_image_frame_index=input_data['extra_image_frame_index'],
        control_video=input_data['control_video'],
        seed=0,
        height=input_data['height'],
        width=input_data['width'],
        tiled=True,
        num_inference_steps=50,
        num_frames=input_data['num_frames'],
    )
    save_video(
        gt_video,
        os.path.join(
            save_path,
            f"gt_video_{idx}_step_{sample_step}_frame_length_{frame_length}_validate_{validate_step}.mp4"
        ),
        fps=10,
        quality=5,
    )
    save_video(
        input_frames,
        os.path.join(
            save_path,
            f"video_input_visualization_{idx}_{sample_step}_step_{frame_length}_frames_validate_{validate_step}.mp4"
        ),
        fps=10,
        quality=5,
    )

    save_video(
        video,
        os.path.join(
            save_path,
            f"video_{idx}_step_{sample_step}_frame_length_{frame_length}_validate_{validate_step}.mp4"
        ),
        fps=10,
        quality=5,
    )
