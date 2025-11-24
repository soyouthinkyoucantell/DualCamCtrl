from accelerate.utils import set_seed
import imageio
import warnings
import torchvision
import argparse
import json
from peft import LoraConfig, inject_adapter_in_model
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator, accelerator, DeepSpeedPlugin
from diffsynth import save_video, save_frames
from safetensors.torch import load_file
import torch
import os
import json
from deepspeed.utils.zero_to_fp32 import (
    load_state_dict_from_zero_checkpoint,
    get_fp32_state_dict_from_zero_checkpoint,
)
from examples.dataset.realestate10k import Camera
from diffsynth.pipelines.wan_video_new_camera import WanVideoCameraPipeline, ModelConfig
from omegaconf import OmegaConf
from examples.wanvideo.model_training.train_with_accelerate import WanTrainingModule
from examples.wanvideo.model_training.args import wan_parser
from PIL import Image
import numpy as np 
import cv2
from packaging import version as pver

def get_relative_pose( cam_params):
    # Always zero_init the first camera pose
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array(
        [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [
        target_cam_c2w,
    ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )

    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    # B, V, HW, 3
    # print(f"rays_o shape: {rays_o.shape}, rays_d shape: {rays_d.shape}")
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    # print(f"rays_dxo shape: {rays_dxo.shape}")
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


# Load the model
state_dir = "/data/user/hongfeizhang/experiments/Wan/10-08-train-fuse-I2V-480-mask-depth-70-3e6-5-10"
checkpoint = "1600"
weight_dir = os.path.join(state_dir, f"checkpoint-step-{checkpoint}")
config_path = 'model_config/controlnet_gate_asym_5_10.yaml'
args_path = 'train_config/normal_config/i2v_train_fuse_5_10_70_3e6.yaml'
parser = wan_parser()
import yaml

yaml_args = OmegaConf.load(args_path)
print(f"yaml args: {yaml_args}")
args = yaml_args

# Prepare Accelerator
accelerator = Accelerator()


files = {
    "astronaut.png": "An astronaut walks forward on the Moon, the Earth looming far behind him. In smooth, alternating steps, his right leg lifts and moves forward while the left leg firmly supports his weight, then the left leg follows in perfect rhythm. Each step flows naturally into the next, with no sudden jumps or flickering between legs. His body shifts weight gracefully with each stride, the soft, gentle light casting subtle shadows on his space suit. Every motion is continuous and coordinated, capturing the serene, low-gravity walk. Real scene in space.",
    "route66.jpg": "A straight two-lane highway stretches into the distance, narrowing to a point on the horizon. The asphalt is slightly worn, with faint cracks and darker patches where tires have passed for years. Yellow double lines run down the center, flanked by white edge lines, and near the foreground a large white highway emblem is painted directly on the road surface, its curves and numbers crisp against the dark pavement.On both sides of the road lies a dry, open landscape—low scrub bushes, sandy soil, and scattered tufts of grass, all in muted browns and grays. There are a few fence posts and tiny hints of structures far away, but mostly the land feels empty and wide.",
    "seaside.png": "High aerial view over a British seaside town on a sunny afternoon. Turquoise sea with gentle waves, a long wooden pier stretching into the water, sandy beach and a bustling promenade with parked cars. Foreground: curved coastal road and pastel, terraced houses with orange roofs. Midground: beach, pools and small buildings. Background: white cliffs and rolling hills fading into haze. Few fluffy clouds, bright natural light, crisp visibility. Slow tilt-down and rightward pan, subtle zoom for parallax. Natural color grade, 4K, 24fps, steady gimbal/drone feel, light wind ambience and distant seagulls.",
}
nprom = 'Vibrant colors, overexposed, static, blurry details, subtitles, poorly drawn style or artwork, still image, overall grayish, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands or face, deformed, disfigured, malformed limbs, fused fingers, static image, messy background, three legs, many people in the background, walking backward.'
device = 'cuda'

# Getting the model ready (here for simplicity we just use the training class)
# Note that we don't skip the downloading process of origin DiT here though we won't use it. So if you don't need it at all, you may modify the from_pretrained methods.
model = WanTrainingModule(
    model_id_with_origin_paths=args.model_id_with_origin_paths,
    trainable_models=args.trainable_models,
    use_gradient_checkpointing=args.use_gradient_checkpointing,
    config_path=config_path,
    ckpt_lora_base_model=args.ckpt_lora_base_model,
    ckpt_lora_adapter_name=args.ckpt_lora_adapter_name,
    ckpt_lora_rank=args.ckpt_lora_rank,
    ckpt_lora_module=args.ckpt_lora_module,
    lora_rank=args.lora_rank,
    args=args,
)

state_path = os.path.join(weight_dir, "pytorch_model")
# state_dict = get_fp32_state_dict_from_zero_checkpoint(weight_dir)
# load_state = model.load_state_dict(state_dict, strict=True)
# assert (
#     len(load_state.unexpected_keys) == 0
# ), f"unexpected_keys keys: {load_state.unexpected_keys}"
model = accelerator.prepare(model)

# demo path
file_root = "demo_pic"
output_path = os.path.join("output", "test_demo")
os.makedirs(output_path, exist_ok=True)

frame_len = 61
_height, _width = 320, 480
num_inference_step = 50
ori_h, ori_w = 360,640 # This is predefined for our demo images

def decode_image(image_tensor):
    byte_data = image_tensor.numpy().tobytes()
    np_data = np.frombuffer(byte_data, dtype=np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img


for file_name, prompt in files.items():
    # RGB,D,Prompt
    img_path = os.path.join(file_root, file_name)
    depth_path = img_path.replace(".jpg", "_depth.jpg").replace(".png", "_depth.png")
    prompt = prompt

    # Traj
    prefix = os.path.splitext(file_name)[0]
    data_file = os.path.join(file_root, f"{prefix}.torch")
    data = torch.load(data_file)
    print(f"Data keys{data.keys()}")
    new_data = {"cameras": data["cameras"]}

    # 覆盖写回
    torch.save(new_data, data_file)

    print("After:", torch.load(data_file).keys())
    frame_indices = range(
        0, frame_len
    )  # You may re-sample the plucker embedding to adjust the length here
    cameras_info = data["cameras"]
    cameras_info = torch.concat(
        [torch.zeros(cameras_info.shape[0], 1), cameras_info], dim=1
    )
    cam_params = [Camera(cameras_info[indice]) for indice in frame_indices]

    # Rescale X/Y here
    sample_wh_ratio = _width / _height

    print(f"Origin height width {ori_h,ori_w}")
    ori_wh_ratio = ori_w / ori_h
    if ori_wh_ratio > sample_wh_ratio:  # rescale fx
        resized_ori_w = _width * ori_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / _height
    else:  # rescale fy
        resized_ori_h = _height / ori_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / _width

    intrinsics = np.asarray(
        [
            [
                cam_param.fx * _width,
                cam_param.fy * _height,
                cam_param.cx * _width,
                cam_param.cy * _height,
            ]
            for cam_param in cam_params
        ],
        dtype=np.float32,
    )
    intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]
    c2w_poses = get_relative_pose(cam_params)
    # [1, n_frame, 4, 4]
    c2w = torch.as_tensor(c2w_poses)[None]
    plucker_embedding = (
        ray_condition(
            intrinsics,
            c2w,
            _height,
            _width,
            device="cpu",
            flip_flag=None,
        )[0]
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    plucker_embedding = plucker_embedding.permute(1, 0, 2, 3).unsqueeze(0)
    # print(f"Plucker embedding shape {plucker_embedding.shape}")
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    with torch.no_grad():
        
        input_data={}
        
        img = Image.open(img_path).convert("RGB").resize((_width, _height))
        depth_img = Image.open(depth_path).convert("L").resize((_width, _height))
        
        
        # T C H W 
        img_tensor = (
            torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)
        )
        depth_tensor = (
            torchvision.transforms.ToTensor()(depth_img)
            .unsqueeze(0)
            .to(device)
            .repeat(1, 3, 1, 1)
        )

        
        # Get data ready 
        input_data["input_image"] = img_tensor
        input_data["input_control"] = depth_tensor
        input_data["plucker_embedding"] = plucker_embedding
        input_data['prompt'] = [prompt]
        input_data['negative_prompt'] =  [nprom]
        input_data['num_frames'] = frame_len
        # Prompt
        videos = model.pipe(
            prompt=input_data['prompt'],
            negative_prompt=input_data['negative_prompt'],
            batch_size=1,
            input_image=input_data["input_image"],
            input_control=input_data["input_control"],
            extra_images=None,
            extra_image_frame_index=None,
            plucker_embedding=input_data["plucker_embedding"],
            seed=0,
            t2v=False,
            height=_height,
            width=_width,
            tiled=True,
            return_control_latents=True,
            num_inference_steps=num_inference_step,
            num_frames=frame_len,
        )

        save_path = os.path.join(
            output_path,
            f"validation_results_{num_inference_step}_inference_steps",
        )
        os.makedirs(save_path, exist_ok=True)
        for k, v in videos.items():
            if "control" in k:
                continue
            if v is None:
                continue
            print(
                f"Handling v with {len(v)} videos, first with frames len {len(v[0])}, keys: {k}"
            )
            # Input video (Actually [input_image, (T-1) zero padding])
            input_frames = [None] * input_data["num_frames"]
            input_frames[0] = (
                input_data["input_image"]
                if k == "images"
                else input_data["input_control"]
            )
            # zero-pad here 
            last_valid = torch.zeros_like(input_frames[0])
            for i in range(len(input_frames)):
                if input_frames[i] is None:
                    input_frames[i] = last_valid
            input_frames = torch.stack(input_frames, dim=0)  # F B C H W
            input_frames = input_frames.permute(1, 2, 0, 3, 4)  # B C F H W

            # Convert to video
            input_frames = model.pipe.vae_output_to_video(
                input_frames, pattern="B T H W C", min_value=0, max_value=1
            )

            # Save video
            for _input_frames,  _predict_video in zip(
                input_frames,  v
            ):
                _save_root = os.path.join(
                    save_path,
                    f"{prefix}",
                )
                _video_save_dir = os.path.join(_save_root, f"{k}_vid")

                input_save_path = os.path.join(
                    _video_save_dir,
                    f"video_input.mp4",
                )

                predict_save_path = os.path.join(
                    _video_save_dir,
                    f"video_pred.mp4",
                )
                os.makedirs(_video_save_dir, exist_ok=True)

                print(f"Rank {rank} saving videos to {predict_save_path}")
                save_video(
                    _input_frames,
                    input_save_path,
                    fps=10,
                    quality=5,
                )
                save_video(
                    _predict_video,
                    predict_save_path,
                    fps=10,
                    quality=5,
                )