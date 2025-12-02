import torch
from accelerate import Accelerator
from examples.wanvideo.model_training.train_with_accelerate import WanTrainingModule
from omegaconf import OmegaConf
import os
import gradio as gr
import torchvision
from pathlib import Path
import numpy as np
from packaging import version as pver


class Camera(object):
    def __init__(
        self,
        entry=None,
        fx=1,
        fy=1,
        cx=0.5,
        cy=0.5,
        c2w=None,
        pos=None,
        direction=None,
        up=None,
    ):

        if entry is not None:
            fx, fy, cx, cy = entry[1:5]
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy

            w2c_mat = np.array(entry[7:]).reshape(3, 4)
            w2c_mat_4x4 = np.eye(4)
            w2c_mat_4x4[:3, :] = w2c_mat
            self.w2c_mat = w2c_mat_4x4
            self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
            return

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        if c2w is not None:
            self.c2w_mat = c2w
            self.w2c_mat = np.linalg.inv(c2w)
            return

        # 用 pos + direction 生成 c2w
        if pos is None:
            pos = np.array([0, 0, 0], dtype=float)
        if direction is None:
            direction = np.array([0, 0, -1], dtype=float)
        if up is None:
            up = np.array([0, 1, 0], dtype=float)

        self.c2w_mat = Camera.look_at(pos, pos + direction, up)
        self.w2c_mat = np.linalg.inv(self.c2w_mat)

    @staticmethod
    def look_at(eye, target, up):
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        M = np.eye(4)
        M[0, :3] = right
        M[1, :3] = up
        M[2, :3] = -forward
        M[:3, 3] = eye
        return M

    @staticmethod
    def spiral_path(T=60, radius=2.0, height_change=1.0, target=np.array([0, 0, 0])):
        cams = []
        for i in range(T):
            theta = 2 * np.pi * i / T
            r = radius * (1 - 0.3 * i / T)  # 半径逐渐缩小

            x = r * np.cos(theta)
            y = height_change * (i / T)
            z = r * np.sin(theta)

            pos = np.array([x, y, z])
            direction = target - pos
            cams.append(Camera(pos=pos, direction=direction))
        return cams

    @staticmethod
    def push_in_path(T=60, start=[0, 0, 3], end=[0, 0, 1], target=np.array([0, 0, 0])):
        cams = []
        start = np.array(start)
        end = np.array(end)
        for i in range(T):
            pos = start * (1 - i / T) + end * (i / T)
            direction = target - pos
            cams.append(Camera(pos=pos, direction=direction))
        return cams

    @staticmethod
    def circle_in_plane_path(T=60, radius=2.0, height=0.0, target=np.array([0, 0, 0])):
        cams = []
        for i in range(T):
            theta = 2 * np.pi * i / T

            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            pos = np.array([x, height, z])

            direction = target - pos
            cams.append(Camera(pos=pos, direction=direction))
        return cams

    @staticmethod
    def left_right_pan_path(
        T=60, range_x=2.0, height=0.0, dist=2.0, target=np.array([0, 0, 0])
    ):
        cams = []
        for i in range(T):
            x = range_x * (i / T - 0.5)  # 从 -range/2 → +range/2
            pos = np.array([x, height, dist])
            direction = target - pos
            cams.append(Camera(pos=pos, direction=direction))
        return cams


def get_relative_pose(cam_params):
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



args_path = "train_config/normal_config/i2v_train_fuse_5_10_70_3e6.yaml"
config_path = "model_config/controlnet_gate_asym_5_10.yaml"

accelerator = Accelerator()
device = accelerator.device  # 自动选择多卡设备
args = OmegaConf.load(args_path)

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


weight_path = "checkpoints/dualcamctrl_diffusion_transformer.pt"
if os.path.exists(weight_path):
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

model.eval()
model = accelerator.prepare(model)
# --------------------------------------------------------


nprom = "Vibrant colors, overexposed, static, blurry details, subtitles, poorly drawn style or artwork, still image, overall grayish, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands or face, deformed, disfigured, malformed limbs, fused fingers, static image, messy background, three legs, many people in the background, walking backward."


def generate_video(
    input_image,
    depth_image,
    prompt,
    negative_prompt=nprom,
    num_frames=61,
    camera_type="Spiral",
):
    # Resize input
    _width, _height = 480, 320
    img = input_image.convert("RGB").resize((_width, _height))
    depth_img = depth_image.convert("L").resize((_width, _height))

    # To tensor
    img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)
    depth_tensor = (
        torchvision.transforms.ToTensor()(depth_img)
        .unsqueeze(0)
        .to(device)
        .repeat(1, 3, 1, 1)
    )

    if camera_type == "Spiral":
        cam_params = Camera.spiral_path(num_frames)
    elif camera_type == "Push-in":
        cam_params = Camera.push_in_path(num_frames)
    elif camera_type == "Circle":
        cam_params = Camera.circle_in_plane_path(num_frames)
    elif camera_type == "Pan":
        cam_params = Camera.left_right_pan_path(num_frames)
    else:
        cam_params = Camera.spiral_path(num_frames)

    # For demo: fake camera embedding (replace with real camera if needed)
    # Here we create a dummy camera with identity matrices
    intrinsics = torch.tensor(
        [[cam.fx, cam.fy, cam.cx, cam.cy] for cam in cam_params], dtype=torch.float32
    )[
        None
    ]  # -> (1, T, 4)

    c2w_poses = get_relative_pose(cam_params)
    c2w = torch.tensor(c2w_poses)[None]
    plucker_embedding = (
        ray_condition(intrinsics, c2w, _height, _width, device="cpu")[0]
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    plucker_embedding = plucker_embedding.permute(1, 0, 2, 3).unsqueeze(0)

    # Prepare input
    input_data = {
        "input_image": img_tensor,
        "input_control": depth_tensor,
        "plucker_embedding": plucker_embedding,
        "prompt": [prompt],
        "negative_prompt": [negative_prompt],
        "num_frames": num_frames,
    }

    with torch.no_grad():
        videos = model.pipe(
            prompt=input_data["prompt"],
            negative_prompt=input_data["negative_prompt"],
            batch_size=1,
            input_image=input_data["input_image"],
            input_control=input_data["input_control"],
            extra_images=None,
            extra_image_frame_index=None,
            plucker_embedding=input_data["plucker_embedding"],
            seed=42,
            t2v=False,
            cfg_scale=5,
            height=_height,
            width=_width,
            tiled=True,
            return_control_latents=True,
            num_inference_steps=50,
            num_frames=num_frames,
        )

    # Save video to temp folder
    output_dir = Path("gradio_output")
    output_dir.mkdir(exist_ok=True)
    result_paths = []

    for k, v in videos.items():
        if "control" in k or v is None:
            continue

        # Input video
        input_frames = [img_tensor if k == "images" else depth_tensor] + [
            torch.zeros_like(img_tensor) for _ in range(num_frames - 1)
        ]
        input_frames = torch.stack(input_frames, dim=0).permute(1, 2, 0, 3, 4)
        input_frames = model.pipe.vae_output_to_video(
            input_frames, pattern="B T H W C", min_value=0, max_value=1
        )

        for _input_frames, _predict_video in zip(input_frames, v):
            video_path = output_dir / f"{k}_pred.mp4"
            save_video(_predict_video, str(video_path), fps=10, quality=5)
            result_paths.append(str(video_path))

    return result_paths[0] if result_paths else None


# Gradio interface

iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Image(type="pil", label="Depth Image"),
        gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt here"),
        gr.Textbox(
            label="Negative Prompt", lines=3, placeholder="Optional", value=nprom
        ),
        gr.Slider(minimum=17, maximum=61, step=4, value=29, label="Number of Frames"),
        gr.Radio(
            ["Spiral", "Push-in", "Circle", "Pan"],
            value="Spiral",
            label="Camera Movement Type",
        ),
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Image-to-Video Generation",
    description="Upload an image and a depth map, enter a prompt, and generate a video.",
)

# iface.launch(
#     server_name="0.0.0.0",
#     server_port=7860,
#     share=False,
# )


def launch_gradio_with_retry(interface, max_retries=5, wait_seconds=2, **kwargs):
    """
    Attempt to launch Gradio multiple times. If it fails, wait and retry.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} to launch Gradio...")
            interface.launch(**kwargs)
            print("Gradio launched successfully")
            break  # Exit loop if launch succeeds
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"Waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)
            else:
                print("Exceeded maximum retry attempts. Launch failed.")
                raise


launch_gradio_with_retry(
    iface,
    max_retries=999,
    wait_seconds=3,
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
)
