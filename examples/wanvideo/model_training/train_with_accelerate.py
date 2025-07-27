
from accelerate.utils import set_seed
from .args import wan_parser
from ...dataset.colmap_debug import ScenesDataset
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

import torch
import os
import json
from diffsynth.pipelines.wan_video_new_altered import WanVideoPipeline, ModelConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_data(data):
    input_data = {
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


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        trainable_modules = filter(
            lambda p: p.requires_grad, self.parameters())
        return trainable_modules

    def trainable_param_names(self):
        trainable_param_names = list(filter(
            lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0]
                                    for named_param in trainable_param_names])
        return trainable_param_names

    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name,
                      param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(
                ":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)

        print(f"dit data type : {self.pipe.dit.parameters().__next__().dtype}")

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        self.pipe.freeze_except(
            [] if trainable_models is None else trainable_models.split(","))

        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        print(f"Training Module initialized with the following configurations:")
        print(
            f"Using gradient checkpointing: {self.use_gradient_checkpointing}")
        print(
            f"Using gradient checkpointing offload: {self.use_gradient_checkpointing_offload}")
        self.extra_inputs = extra_inputs.split(
            ",") if extra_inputs is not None else []

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data.get("prompt", ""), }
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "height": data.get("height", None),
            "width": data.get("width", None),
            "input_video": data.get("input_video", None),
            "num_frames": data.get("num_frames", 81),
            'input_image': data.get("input_image", None),
            'extra_images': data.get("extra_images", None),
            'extra_image_frame_index': data.get("extra_image_frame_index", None),
            'control_video': data.get("control_video", None),
            'plucker_embedding': data.get("plucker_embedding"),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            # input("Press Enter to continue...")  # For debugging purposes
            # print(f"Handling unit: {unit.__class__.__name__}")
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
            # print(f"Device of text_encoder,dit,vae,image_encoder: "
            #     f"{next(self.pipe.text_encoder.parameters()).device}, "
            #     f"{next(self.pipe.dit.parameters()).device}, "
            #     f"{next(self.pipe.vae.parameters()).device}, "
            #     f"{next(self.pipe.image_encoder.parameters()).device}")
            # print(
            #     f"Noise shape: {inputs_shared['noise'].shape if 'noise' in inputs_shared else 'N/A'}")
            # print(
            #     f"Input latent shape: {inputs_shared['input_latents'].shape if 'input_latents' in inputs_shared else 'N/A'}")
            # print(
            #     f"Latent shape: {inputs_shared['latents'].shape if 'latents' in inputs_shared else 'N/A'}")
            # print(
            #     f"y shape: {inputs_shared['y'].shape if 'y' in inputs_shared else 'N/A'}")
            # print(
            #     f"control_latent shape: {inputs_shared['control_latent'].shape if 'control_latent' in inputs_shared else 'N/A'}")
            # print(
            #     f"plucker_embedding shape: {inputs_shared['plucker_embedding'].shape if 'plucker_embedding' in inputs_shared else 'N/A'}")
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):

        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name)
                  for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

    def validate(self, accelerator, global_step, test_dataloader=None):
        # This method is a placeholder for validation logic.
        # You can implement your validation logic here if needed.
        import numpy as np
        from typing import List, Optional
        from PIL import Image
        import cv2
        import torch
        from diffsynth import save_video
        assert test_dataloader is not None, "Test dataloader must be provided for validation."

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

        for idx, batch in enumerate(test_dataloader):
            if idx >= 4:
                break
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                input_data = get_data(batch)
                # mask the video every 10 frames

                video = self.pipe(
                    prompt="清楚的，清晰的，明亮的，色调自然，色彩丰富，细节清晰，动态画面，动态视频，动态场景",
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
                save_path = "validation_videos"
                os.makedirs(save_path, exist_ok=True)

                save_video(
                    video,
                    os.path.join(
                        save_path,
                        f"video_{idx}_step_{global_step}.mp4"
                    ),
                    fps=10,
                    quality=5,
                )


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def on_step_end(self, loss):
        print(
            f"Step loss: {loss.mean().item() if isinstance(loss, torch.Tensor) else loss}")

    def save_model_state(self, accelerator, model, epoch_id, global_step):
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            # print(f"state_dict keys: {list(state_dict.keys())}")
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(
                self.output_path, f"epoch-{epoch_id}-step-{global_step}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
        accelerator.wait_for_everyone()

    def save_training_state(self, accelerator, epoch_id, global_step):
        if accelerator.is_main_process:
            accelerator.print("Saving training state...")
            os.makedirs(self.output_path, exist_ok=True)
            # save all internal states properly
            accelerator.save_state(self.output_path)
            # manually save extra info
            torch.save(
                {
                    "epoch": epoch_id,
                    "global_step": global_step
                },
                os.path.join(self.output_path, "meta_state.pt")
            )
        accelerator.wait_for_everyone()

    def load_training_state(self, accelerator):
        state_path = os.path.join(self.output_path, "meta_state.pt")
        if os.path.exists(state_path):
            accelerator.print(f"Loading training state from {state_path}")
            accelerator.load_state(self.output_path)
            meta = torch.load(state_path, map_location="cpu")
            return meta["epoch"], meta["global_step"]
        else:
            accelerator.print(
                f"No training state found at {state_path}. Starting from scratch.")
            return 0, 0

    def load_latest_model_weights(self,  model):
        from safetensors.torch import load_file
        if not os.path.exists(self.output_path):
            print(
                f"Output path {self.output_path} does not exist. Skipping loading weights.")
            return
        ckpts = sorted([
            f for f in os.listdir(self.output_path)
            if f.endswith(".safetensors")
        ], reverse=True)
        if not ckpts:
            print(
                f"No checkpoints found in {self.output_path}. Skipping loading weights.")
            return
        latest_ckpt = os.path.join(self.output_path, ckpts[0])
        print(f"Loading model weights from {latest_ckpt}")
        state_dict = load_file(latest_ckpt)
        model.pipe.dit.load_state_dict(
            state_dict, strict=True)


def launch_training_task(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    validate_step: int = 500,
    gradient_accumulation_steps: int = 1,
):
    train_dataloader = torch.utils.data.DataLoader(
        # Assuming dataset returns a dict
        train_dataset, shuffle=True, batch_size=1, num_workers=2, collate_fn=lambda x: x[0], prefetch_factor=1, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=2, collate_fn=lambda x: x[0], prefetch_factor=1, pin_memory=True
    )

    # deepspeed_plugin = DeepSpeedPlugin(
    #     hf_ds_config='deepspeed.json'
    # )
    accelerator = Accelerator()

    set_seed(42)

    accelerator.print(
        f"Initial accelerator with gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator.print(
        f"Using {accelerator.num_processes} processes for training.")
    accelerator.print(
        f"Preparing model, optimizer, dataloader, and scheduler with accelerator.")
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    accelerator.print(
        f"Model param dtype : {next(model.pipe.dit.parameters()).dtype}")

    start_epoch, global_step = model_logger.load_training_state(
        accelerator)
    if global_step > 0:
        accelerator.print(
            f"Resuming training from epoch {start_epoch}, global step {global_step}")
    accelerator.print(f"Validate every {validate_step} steps.")
    # global_step = 0
    for epoch_id in range(num_epochs):
        train_dataloader.set_epoch(epoch_id + start_epoch)
        # model.validate(accelerator=accelerator, global_step=global_step)
        for idx, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch_id + 1}/{num_epochs}")):
            with accelerator.accumulate(model):
                input_data = get_data(data)
                optimizer.zero_grad()
                # accelerator.print(
                #     f"Forward no. {idx + 1}/{len(train_dataloader)}")

                # with accelerator.autocast():
                loss = model(input_data)
                # accelerator.print(
                #     f"Backward no. {idx + 1}/{len(train_dataloader)}")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.trainable_modules(), max_norm=1.0)
                    global_step += 1
                    if (global_step+validate_step) % validate_step == 1:
                        model_logger.save_model_state(
                            accelerator, model, epoch_id, global_step)
                        model_logger.save_training_state(
                            accelerator, epoch_id, global_step)
                        accelerator.print(
                            f"Checkpoint saved at step {global_step}")
                        if accelerator.is_main_process:
                            model.pipe.dit.eval()
                            model.validate(
                                accelerator=accelerator,
                                global_step=global_step,
                                test_dataloader=test_dataloader
                            )
                            model.pipe.dit.train()
                        accelerator.wait_for_everyone()
                optimizer.step()
                scheduler.step()


def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key]
                      for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(
                output_path, "data_cache", f"{data_id}.pth"))


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    # dataset = VideoDataset(args=args)

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
    )
    print(f"Model data type : {model.pipe.dit.parameters().__next__().dtype}")
    # print the module needs grads
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} requires grad")

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,

    )
    # Default set the ratio that train test has no overlap.
    train_ratio = 0.99
    test_ratio = 1 - train_ratio

    train_dataset = ScenesDataset(
        relative_pose=True,
        split='train',
        ratio=train_ratio,
        patch_size=[832, 480],  # W H
    )
    test_dataset = ScenesDataset(
        relative_pose=True,
        split='test',
        ratio=test_ratio,
        patch_size=[832, 480],  # W H
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    model_logger.load_latest_model_weights(model)
    # model.pipe.enable_vram_management()

    launch_training_task(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        model_logger=model_logger,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        validate_step=args.validate_step,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
