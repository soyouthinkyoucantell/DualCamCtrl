import torch
# from diffsynth import save_video
from diffsynth import save_video, VideoData

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

origin = False  # Set to True if you want to use the original WanVideoPipeline
# from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# origin = True

import cv2
from PIL import Image
from typing import List, Optional
import numpy as np


def read_video_as_pil_frames(video_path: str) -> Optional[List[Image.Image]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    frames: List[Image.Image] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads in BGR format, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)

    cap.release()
    return frames


video_path = "/hpc2hdd/home/hongfeizhang/dataset/ttr/abandonedfactory/Easy/P000/video_right/P000_right.mp4"
input_video = read_video_as_pil_frames(video_path)
# input_image = Image.open("/hpc2hdd/home/hongfeizhang/dataset/ttr/westerndesert/Easy/P007/image_left/000001_left.png")
# end_image = Image.open("/hpc2hdd/home/hongfeizhang/dataset/ttr/westerndesert/Easy/P007/image_left/000090_left.png")
start_frame = 0
end_frame = 80
input_image = input_video[start_frame]
end_image = input_video[end_frame]
# input_image = input_video[0]
# end_image = input_video[200]
extract_image_frame_index = [i for i in range(5, 9)]
extra_images = [input_video[i] for i in extract_image_frame_index]

print(f"Shape of input image: {input_image.size}, end image: {end_image.size}")
print(
    f"Shape of input video: {len(input_video)} frames, first frame size: {input_video[0].size}"
)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-1.3B-Control",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-1.3B-Control",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-1.3B-Control",
            origin_file_pattern="Wan2.1_VAE.pth",
            offload_device="cpu",
        ),
        ModelConfig(
            model_id="PAI/Wan2.1-Fun-1.3B-Control",
            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            offload_device="cpu",
        ),
    ],
)
pipe.enable_vram_management()
num_inference_steps = 50

from modelscope import dataset_snapshot_download

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=f"data/examples/wan/control_video.mp4"
)
# print(f"Loaded {len(input_video)} frames from video: {video_path}")
# input_video = input_video[:81]
control_video = VideoData("data/examples/wan/control_video.mp4", height=832, width=576)

video = pipe(
    prompt="一个动画场景，镜头慢慢往左下移动，从门看向地板，视频清晰可见，地板和墙上有明显的反光。",
    # negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=input_image,
    end_image=end_image,
    # extra_images=extra_images,
    # extract_image_frame_index=extract_image_frame_index,
    control_video=input_video[start_frame : end_frame + 1],
    # input_video=input_video,
    # camera_control_direction='Left',
    seed=0,
    height=480,
    width=640,
    tiled=True,
    num_inference_steps=num_inference_steps,
    # num_frames=end_frame - start_frame + 1,
)
save_video(
    video,
    f"video_{start_frame}_{end_frame}_{'origin' if origin else 'fake'}_{num_inference_steps}_steps_{pipe.__class__.__name__}.mp4",
    fps=15,
    quality=5,
)
