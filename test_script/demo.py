import numpy as np
from typing import List, Optional
from PIL import Image
import cv2
import torch
from diffsynth import save_video

from diffsynth.pipelines.wan_video_new_altered import WanVideoPipeline, ModelConfig

origin = False  # Set to True if you want to use the original WanVideoPipeline
# from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# origin = True


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


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a PyTorch tensor."""
    tensor = torch.tensor(np.array(pil_image))
    print(f"Converted PIL image to tensor with shape: {tensor.shape}")
    if tensor.ndim == 3:  # H W C
        tensor = tensor.permute(2, 0, 1).float() / 255.0
    elif tensor.ndim == 4:  # F H W C
        tensor = tensor.permute(0, 3, 1, 2).float() / 255.0

    return tensor.unsqueeze(0)  # Add batch dimension


video_path = "/hpc2hdd/home/hongfeizhang/dataset/ttr/abandonedfactory/Easy/P000/video_right/P000_right.mp4"
input_video = read_video_as_pil_frames(video_path)
# input_image = Image.open("/hpc2hdd/home/hongfeizhang/dataset/ttr/westerndesert/Easy/P007/image_left/000001_left.png")
# end_image = Image.open("/hpc2hdd/home/hongfeizhang/dataset/ttr/westerndesert/Easy/P007/image_left/000090_left.png")
start_frame = 0
end_frame = 40
input_image = input_video[start_frame]

end_image = input_video[end_frame]
# input_image = input_video[0]
# end_image = input_video[200]

extra_start = 5
extra_end = 29
extra_frame_idx = list(range(5, 29))  # Indices of extra frames to be used
extract_image_frame_index = extra_frame_idx.copy()
extra_images = [input_video[i] for i in extract_image_frame_index]

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Shape of input image: {input_image.size}, end image: {end_image.size}")
print(
    f"Shape of input video: {len(input_video)} frames, first frame size: {input_video[0].size}"
)

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
    config_path='/hpc2hdd/home/hongfeizhang/hongfei_workspace/DiffSynth-Studio/model_config/controlnet_pre_pose.yaml'
)
# pipe.enable_vram_management()
num_inference_steps = 50
pipe.to(device)

# print(f"Loaded {len(input_video)} frames from video: {video_path}")
# input_video = input_video[:81]
input_image = pil_to_tensor(input_image).to(device).repeat(1, 1, 1, 1)
num_frame = end_frame - start_frame + 1
batch_size = input_image.shape[0]

frame_mask = torch.ones(batch_size, num_frame).to(
    device)  # Assuming all frames are valid
frame_mask[:, 1:] = 0  # Set all frames except the first one to 0
frame_mask[:, extra_start:extra_end + 1] = 1  # Set the range of frames to 1


extra_images = pil_to_tensor(
    input_video[extra_start:extra_end + 1]).to(device).repeat(1, 1, 1, 1, 1)
print(f"Extra images shape before processing: {extra_images.shape}")
b, _, c, h, w = extra_images.shape
extra_images_total = torch.zeros(
    (batch_size, num_frame, c, h, w), device=device, dtype=extra_images.dtype
)
extra_images_total += 0.5
extra_images_total[:, extra_start:extra_end + 1] = extra_images


simulate_batch_size = 2

# Repeat input image for batch size
input_image = input_image.repeat(simulate_batch_size, 1, 1, 1)
extra_images_total = extra_images_total.repeat(
    simulate_batch_size, 1, 1, 1, 1
)  # Repeat extra images for batch size
extra_images_total[1, extra_start+8:extra_end + 1] = 0.5


frame_mask = frame_mask.repeat(simulate_batch_size, 1)
frame_mask[1, extra_start+8:extra_end + 1] = 0

print(f"frame mask {frame_mask}")
print(f"Extra images mean on c h w: {extra_images_total.mean(dim=(2, 3,4))}")


# Repeat frame mask for batch size
control_video = pil_to_tensor(
    input_video[start_frame: end_frame + 1]).repeat(
        simulate_batch_size, 1, 1, 1, 1).to(device)
videos = pipe(
    prompt=["一个动画场景，镜头慢慢往左下移动，从门看向地板，视频清晰可见，地板和墙上有明显的反光。",
            '一个动画场，屋顶上有个海绵宝宝。'],
    negative_prompt=[
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"]*simulate_batch_size,

    input_image=input_image,
    batch_size=input_image.shape[0],
    # end_image=end_image,
    extra_images=extra_images_total,
    extra_image_frame_index=frame_mask,
    control_video=control_video,
    # input_video=input_video,
    # camera_control_direction='Left',
    frame_mask=frame_mask,
    seed=0,
    height=480,
    width=640,
    tiled=True,
    num_inference_steps=num_inference_steps,
    num_frames=num_frame,
)
for idx, video in enumerate(videos):
    save_video(
        video,
        f"video_idx_{idx}_{start_frame}_{end_frame}_extra_{extract_image_frame_index[0]}-{extract_image_frame_index[-1]}_{'origin' if origin else 'fake'}_{num_inference_steps}_steps.mp4",
        fps=15,
        quality=5,
    )
