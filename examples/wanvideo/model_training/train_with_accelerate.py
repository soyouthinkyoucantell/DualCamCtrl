from omegaconf import OmegaConf
from accelerate.utils import set_seed
from .args import wan_parser
from ...dataset.colmap_debug import ScenesDataset
from ...dataset.realestate10k import RealEstate10KPose
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
from diffsynth.pipelines.wan_video_new_camera import WanVideoCameraPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]

        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], str):
            collated[key] = values
        elif values[0] is None:
            collated[key] = None
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(values[0])}")
    return collated


def get_data(data, args):
    if data is None:
        return None
    else:
        input_data = {
            "images": data["images"],
            "height": data["images"].shape[-2],
            "width": data["images"].shape[-1],
            "input_video": data.get("images", None),
            "num_frames": data["images"].shape[1],
            "input_image": data.get("images")[:, 0],
            "input_control": data.get("control")[:, 0],
            "extra_images": data.get("extra_images", None),
            "extra_image_frame_index": data.get("extra_image_frame_index", None),
            "t2v": args.t2v,
            'drop_loss_rate':args.drop_loss_rate,
            "control_video": data.get("control", None),
            "prompt": data.get("prompt", ""),
            "return_control_latents": args.return_control_latents,
            "negative_prompt": data.get(
                "negative_prompt", [""] * data["images"].shape[0]
            ),
            "batch_size": data["images"].shape[0],
            "plucker_embedding": data.get("camera_infos", None),
        }
        input_data["negative_prompt"] = [
            "Vibrant colors, overexposed, static, blurry details, subtitles, poorly drawn style or artwork, still image, overall grayish, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands or face, deformed, disfigured, malformed limbs, fused fingers, static image, messy background, three legs, many people in the background, walking backward."
            for _ in range(input_data["batch_size"])
        ]

        if args.t2v:
            input_data["input_image"] = torch.zeros_like(input_data["input_image"])
            input_data["input_control"] = torch.zeros_like(input_data["input_control"])
        else:
            input_data['prompt'] = ['']*input_data['batch_size']
        return input_data


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules

    def trainable_param_names(self):
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.named_parameters(),
            )
        )
        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        return trainable_param_names

    def add_lora_to_model(
        self, model, target_modules, lora_rank, lora_alpha=None, adapter_name="default"
    ):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules
        )
        model = inject_adapter_in_model(lora_config, model, adapter_name=adapter_name)
        return model

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if name in trainable_param_names
        }
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix) :]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        args,
        model_paths=None,
        model_id_with_origin_paths=None,
        trainable_models=None,
        ckpt_lora_base_model=None,
        ckpt_lora_module="q,k,v,o,ffn.0,ffn.2",
        ckpt_lora_rank=32,
        lora_rank=32,
        ckpt_lora_adapter_name="default",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        copy_control_weights=True,
        extra_inputs=None,
        config_path=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [
                ModelConfig(
                    model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]
                )
                for i in model_id_with_origin_paths
            ]
        if args.pipe_name == "wan_video_camera":
            self.pipe = WanVideoCameraPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
                model_configs=model_configs,
                copy_control_weights=copy_control_weights,
                config_path=config_path,
            )
        elif args.pipe_name == "wan_video_inp":
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cpu",
                model_configs=model_configs,
                config_path=config_path,
            )

        from omegaconf import OmegaConf

        if config_path is not None:
            self.model_config = OmegaConf.load(config_path)

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        self.pipe.freeze_except(
            [] if trainable_models is None else trainable_models.split(",")
        )

        # Add LoRA to the base models
        if ckpt_lora_base_model is not None and args.ckpt_has_lora:

            model = self.add_lora_to_model(
                getattr(self.pipe, ckpt_lora_base_model),
                target_modules=(
                    ckpt_lora_module.split(",") if ckpt_lora_module else None
                ),
                lora_rank=lora_rank,
                adapter_name=ckpt_lora_adapter_name,
            )
            setattr(self.pipe, ckpt_lora_base_model, model)

        else:
            print(
                f"No New LoRA added, training parameters: {sum(p.numel() for p in self.trainable_modules())}"
            )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        print(f"Training Module initialized with the following configurations:")
        print(f"Using gradient checkpointing: {self.use_gradient_checkpointing}")
        print(
            f"Using gradient checkpointing offload: {self.use_gradient_checkpointing_offload}"
        )
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []

    def forward_preprocess(self, data):
        # CFG-sensitive parameters

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "height": data.get("height", None),
            "width": data.get("width", None),
            "input_video": data.get("input_video", None),
            "num_frames": data.get("num_frames", None),
            "input_image": data.get("input_image", None),
            "input_control": data.get("input_control", None),
            "extra_images": data.get("extra_images", None),
            "extra_image_frame_index": data.get("extra_image_frame_index", None),
            "control_video": data.get("control_video", None),
            "plucker_embedding": data.get("plucker_embedding"),
            "batch_size": data.get("batch_size"),
            "return_control_latents": data.get("return_control_latents"),
            "t2v": data.get("t2v"),
            'drop_loss_rate':data.get("drop_loss_rate",30),
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
        batch_size = inputs_shared["batch_size"]

        inputs_posi = {
            "prompt": data.get("prompt", [""] *data.get('batch_size')),
        }
        # print(f"positive_prompt :{inputs_posi}")

        # if not inputs_shared['t2v']:
        #     inputs_posi={
        #         'prompt': ['']*data.get('batch_size')
        #     }
        inputs_nega = {
            "negative_prompt": data.get("negative_prompt", [""] * data.get('batch_size')),
        }

        # print(f"Negative prompt: {inputs_nega}")
        for unit in self.pipe.units:
            # print(f"Processing unit: {unit.__class__.__name__}")
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        # if inputs is None:
        inputs = self.forward_preprocess(data)
        models = {
            name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models
        }
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

    def validate(
        self,
        accelerator,
        global_step,
        args,
        test_dataloader=None,
        output_path=None,
        validate_batch=1,
    ):
        # This method is a placeholder for validation logic.
        # You can implement your validation logic here if needed.
        import numpy as np
        from typing import List, Optional
        from PIL import Image
        import cv2
        import torch
        from diffsynth import save_video

        assert (
            test_dataloader is not None
        ), "Test dataloader must be provided for validation."
        num_inference_steps = [50]
        rank = accelerator.process_index
        world_size = accelerator.num_processes

        for num_inference_step in num_inference_steps:
            # so the init save num should start from (rank)*batchsize e.g. 0->0 1->batch_size 2->batch_size*2
            rgb_video_save_num = rank * args.batch_size
            control_video_save_num = rank * args.batch_size

            for idx, batch in enumerate(test_dataloader):

                # no more than validate_batch
                if idx >= validate_batch:
                    break

                if idx != 0:
                    rgb_video_save_num += (world_size - 1) * args.batch_size
                    control_video_save_num += (world_size - 1) * args.batch_size
                with torch.no_grad():
                    input_data = get_data(batch, args)
                    # print(f"input data input control shape {inp_ctrl.shape}")
                    # Prompt
                    _pp= input_data['prompt']
                    _np= input_data['negative_prompt']
                    # print(f"Using prompt {_pp, _np}")
                    
                    videos = self.pipe(
                        prompt=input_data["prompt"],
                        negative_prompt=input_data["negative_prompt"],
                        batch_size=input_data["batch_size"],
                        input_image=input_data["input_image"],
                        input_control=input_data["input_control"],
                        extra_images=input_data["extra_images"],
                        extra_image_frame_index=input_data["extra_image_frame_index"],
                        plucker_embedding=input_data["plucker_embedding"],
                        seed=0,
                        t2v=args.t2v,
                        height=input_data["height"],
                        width=input_data["width"],
                        tiled=True,
                        return_control_latents=True,
                        num_inference_steps=num_inference_step,
                        num_frames=input_data["num_frames"],
                    )
                    # Create save path
                    save_path = os.path.join(
                        output_path,
                        f"validation_results_{num_inference_step}_inference_steps",
                    )
                    os.makedirs(save_path, exist_ok=True)
                    # really need to be very careful at the multi-gpu case when saving
                    for k, v in videos.items():

                        if v is None:
                            continue
                        # print(
                        #     f"Handling v with {len(v)} videos, first with frames len {len(v[0])}, keys: {k}"
                        # )

                        # Key in ['images','control_video']
                        # GT video
                        gt_video = input_data[k].to(self.pipe.device)
                        gt_video = gt_video.permute(0, 2, 1, 3, 4)  # B C F H W

                        # Input video
                        input_frames = [None] * input_data["num_frames"]
                        input_frames[0] = (
                            input_data["input_image"]
                            if k == "images"
                            else input_data["input_control"]
                        )

                        last_valid = torch.zeros_like(input_frames[0])
                        for i in range(len(input_frames)):
                            if input_frames[i] is None:
                                input_frames[i] = last_valid
                        input_frames = torch.stack(input_frames, dim=0)  # F B C H W
                        input_frames = input_frames.permute(1, 2, 0, 3, 4)  # B C F H W

                        # Convert to video, increment video_save_num
                        input_frames = self.pipe.vae_output_to_video(
                            input_frames, pattern="B T H W C", min_value=0, max_value=1
                        )
                        gt_video = self.pipe.vae_output_to_video(
                            gt_video, pattern="B T H W C", min_value=0, max_value=1
                        )

                        # Save video, self-increase
                        for _input_frames, _gt_video, _predict_video in zip(
                            input_frames, gt_video, v
                        ):
                            input_save_path = os.path.join(
                                save_path,
                                f"video_{rgb_video_save_num if k=='images' else control_video_save_num}_{k}_input.mp4",
                            )
                            gt_save_path = os.path.join(
                                save_path,
                                f"video_{rgb_video_save_num if k=='images' else control_video_save_num}_{k}_gt.mp4",
                            )
                            predict_save_path = os.path.join(
                                save_path,
                                f"video_{rgb_video_save_num if k=='images' else control_video_save_num}_{k}_step_{global_step}.mp4",
                            )
                            print(f"Rank {rank} saving videos to {predict_save_path}")
                            if not os.path.exists(input_save_path):
                                save_video(
                                    _input_frames,
                                    input_save_path,
                                    fps=10,
                                    quality=5,
                                )
                            if not os.path.exists(gt_save_path):
                                save_video(
                                    _gt_video,
                                    gt_save_path,
                                    fps=10,
                                    quality=5,
                                )
                            save_video(
                                _predict_video,
                                predict_save_path,
                                fps=10,
                                quality=5,
                            )
                            if k == "images":
                                rgb_video_save_num += 1
                            else:
                                control_video_save_num += 1
                            # print(
                            #     f"rgb_video_save_num: {rgb_video_save_num}, control_video_save_num: {control_video_save_num}"
                            # )


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x
    ):
        self.output_path = output_path
        import time

        # current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        # self.output_path = os.path.join(
        #     self.output_path, f"logs_{current_time}"
        # )
        os.makedirs(self.output_path, exist_ok=True)

        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def load_training_state(self, accelerator, dir=None):
        if dir is None:
            # dir = self.output_path
            return 0, 0

        meta_path = os.path.join(dir, "meta_state.pt")
        if not os.path.exists(meta_path):
            print(f"No training state found at {meta_path}.")
            return 0, 0

        # 先用 accelerate 载入所有状态，包括模型权重、优化器和调度器等
        accelerator.load_state(dir)

        # 再读取训练元信息
        meta = torch.load(meta_path, map_location="cpu")
        epoch = meta.get("epoch", 0)
        global_step = meta.get("global_step", 0)

        print(
            f"Loaded training state from {meta_path}: epoch={epoch}, global_step={global_step}"
        )
        return epoch, global_step

    def save_training_state(self, accelerator, epoch_id, global_step):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Saving training state...")

        # 这里准备一个 dict 存放自定义训练元信息
        meta_state = {
            "epoch": epoch_id,
            "global_step": global_step,
        }
        step_save_path = os.path.join(
            self.output_path, f"checkpoint-step-{global_step}"
        )
        os.makedirs(step_save_path, exist_ok=True)
        # 先保存 meta_state 到一个临时文件
        meta_path = os.path.join(step_save_path, "meta_state.pt")
        if accelerator.is_main_process:
            torch.save(meta_state, meta_path)

        # 统一用 accelerator.save_state 保存所有状态到 output_path
        # 它会自动保存 model, optimizer, scheduler 等所有由 accelerate 管理的组件
        accelerator.save_state(step_save_path)


def launch_training_task(
    accelerator,
    start_epoch,
    global_step,
    args,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    validate_step: int = 500,
    log_step: int = 10,
):

    accelerator.print(
        f"Initial accelerator with gradient accumulation steps: {accelerator.gradient_accumulation_steps}"
    )
    accelerator.print(f"Using {accelerator.num_processes} processes for training.")
    accelerator.print(
        f"Preparing model, optimizer, dataloader, and scheduler with accelerator."
    )

    accelerator.print(f"Model param dtype : {next(model.pipe.dit.parameters()).dtype}")

    if global_step > 0:
        accelerator.print(
            f"Resuming training from epoch {start_epoch}, global step {global_step}"
        )
    else:
        accelerator.print("Starting training from scratch.")

    accelerator.print(f"Validate every {validate_step} steps.")

    if args.init_validate:
        accelerator.print(
            f"Starting validation with model at epoch {start_epoch}, global step {global_step}"
        )
        model.pipe.dit.eval()
        model.validate(
            accelerator=accelerator,
            global_step=global_step,
            args=args,
            test_dataloader=test_dataloader,
            output_path=model_logger.output_path,
            validate_batch=args.validate_batch,
        )
        model.pipe.scheduler.set_timesteps(1000, training=True)
        model.pipe.dit.train()
    accelerator.wait_for_everyone()

    print(f"accelerator.state.deepspeed_plugin: {accelerator.state.deepspeed_plugin}")
    optimizer.zero_grad()

    # old_step = optimizer.step  # 备份原始方法

    # def patched_step(*args, **kwargs):
    #     print(
    #         f"[Patched] optimizer.step() called, lr: {optimizer.param_groups[0]['lr']:.6f}")
    #     return old_step(*args, **kwargs)

    # optimizer.step = patched_step

    accumulate_main_loss = 0.0
    accumulate_control_loss = 0.0
    acm_cnt = 0
    for epoch_id in range(num_epochs):
        train_dataloader.set_epoch(epoch_id + start_epoch)
        for idx, data in enumerate(
            tqdm(
                train_dataloader,
                desc=f"Epoch {epoch_id + 1}/{num_epochs}",
                disable=not accelerator.is_main_process,
            )
        ):
            # Forward and backward pass
            with accelerator.accumulate(model):
                input_data = get_data(data, args=args)
                loss = model(input_data)
                accelerator.backward(loss["total_loss"])
                accumulate_main_loss += loss["main_loss"]
                accumulate_control_loss += loss["control_loss"]
                acm_cnt += 1
                # Update optimizer and scheduler
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.trainable_modules(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    accelerator.print(
                        f"Learning rate : {scheduler.get_last_lr()[0]:.8f}"
                    )
                    # Calculate the average loss across all processes
                    if global_step % log_step == 0:
                        accumulate_main_loss /= acm_cnt
                        accumulate_control_loss /= acm_cnt

                        local_avg_main_loss_tensor = (
                            accumulate_main_loss.detach().clone()
                        )

                        all_main_loss = accelerator.gather_for_metrics(
                            local_avg_main_loss_tensor
                        )
                        accelerator.print(
                            f"local avg main loss tensor: {all_main_loss}"
                        )

                        accumulate_main_loss = all_main_loss.mean().item()

                        if accumulate_control_loss != 0:
                            local_avg_control_loss_tensor = (
                                accumulate_control_loss.detach().clone()
                            )
                            all_control_loss = accelerator.gather_for_metrics(
                                local_avg_control_loss_tensor
                            )
                            accelerator.print(
                                f"local avg control loss tensor: {all_control_loss}"
                            )
                            accumulate_control_loss = all_control_loss.mean().item()

                        accelerator.print(
                            f"Loss at step {global_step}: main loss = {accumulate_main_loss:.4f}, control loss = {accumulate_control_loss:.4f}"
                        )
                        accumulate_main_loss, accumulate_control_loss = 0.0, 0.0
                        acm_cnt = 0

                    global_step += 1
                    if (global_step) % validate_step == 0:
                        model_logger.save_training_state(
                            accelerator=accelerator,
                            epoch_id=epoch_id + start_epoch,
                            global_step=global_step,
                        )
                        accelerator.wait_for_everyone()
                        accelerator.print(f"Checkpoint saved at step {global_step}")
                        model.pipe.dit.eval()
                        model.validate(
                            accelerator=accelerator,
                            global_step=global_step,
                            args=args,
                            test_dataloader=test_dataloader,
                            output_path=model_logger.output_path,
                            validate_batch=args.validate_batch,
                        )
                        model.pipe.scheduler.set_timesteps(1000, training=True)
                        model.pipe.dit.train()
                        accelerator.wait_for_everyone()


# def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
#     dataloader = torch.utils.data.DataLoader(
#         dataset, shuffle=False, collate_fn=lambda x: x[0])
#     accelerator = Accelerator()
#     model, dataloader = accelerator.prepare(model, dataloader)
#     os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
#     for data_id, data in enumerate(tqdm(dataloader)):
#         with torch.no_grad():
#             inputs = model.forward_preprocess(data)
#             inputs = {key: inputs[key]
#                       for key in model.model_input_keys if key in inputs}
#             torch.save(inputs, os.path.join(
#                 output_path, "data_cache", f"{data_id}.pth"))


if __name__ == "__main__":

    import builtins
    import os
    import sys

    accelerator = Accelerator()
    set_seed(42, device_specific=True)
    from omegaconf import OmegaConf
    import argparse

    def get_config():
        # 支持命令行传递 config.yaml 路径
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, default=None, help="Path to config yaml"
        )
        args = parser.parse_args()

        # 读取 YAML 配置
        if not os.path.exists(args.config):
            parser = wan_parser()
            args = parser.parse_args()
        else:
            cfg = OmegaConf.load(args.config)
        return cfg

    cfg = get_config()
    print(OmegaConf.to_yaml(cfg))  # 打印出来检查
    args = cfg

    if not accelerator.is_main_process:
        import logging
        import deepspeed

        deepspeed.utils.logging.logger.disabled = True
    # dataset = VideoDataset(args=args)
    # export DEEPSPEED_LOG_LEVEL=debug
    # export DEEPSPEED_ZERO_LOG_LEVEL=debug

    # Save model config and args
    model_config_save_path = os.path.join(args.output_path, "model_config.yaml")
    os.makedirs(args.output_path, exist_ok=True)
    if accelerator.is_main_process and not os.path.exists(model_config_save_path):
        accelerator.print(f"Saving model configuration to {model_config_save_path}")

        model_config = OmegaConf.load(args.model_config_path)

        OmegaConf.save(config=model_config, f=model_config_save_path)
    args_save_path = os.path.join(args.output_path, "args.yaml")
    import yaml

    if accelerator.is_main_process and not os.path.exists(args_save_path):
        accelerator.print(f"Saving args to {args_save_path}")
        OmegaConf.save(config=args, f=args_save_path)
    accelerator.wait_for_everyone()

    # Load model
    model = WanTrainingModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        config_path=args.model_config_path,
        ckpt_lora_base_model=args.ckpt_lora_base_model,
        ckpt_lora_adapter_name=args.ckpt_lora_adapter_name,
        ckpt_lora_rank=args.ckpt_lora_rank,
        ckpt_lora_module=args.ckpt_lora_module,
        lora_rank=args.lora_rank,
        copy_control_weights=args.copy_control_weights,
        args=args,
    )
    # assert the
    lora_list = []
    for name, param in model.pipe.dit.named_parameters():
        if "lora" in name:
            lora_list.append(name)
    accelerator.print(f"After initializing model, lora parameters: {len(lora_list)}")

    model_state_dir = os.path.join(f"{args.training_state_dir}", "pytorch_model")
    if os.path.exists(model_state_dir):
        from deepspeed.utils.zero_to_fp32 import (
            load_state_dict_from_zero_checkpoint,
            get_fp32_state_dict_from_zero_checkpoint,
        )

        accelerator.print(f"Loading model state from {model_state_dir}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(
            args.training_state_dir
        )  # already on cpu
        lora_list = []
        for name, param in state_dict.items():
            if "lora" in name:
                lora_list.append(name)
        accelerator.print(
            f"The model to be loaded has lora parameters: {len(lora_list)}"
        )
        load_state = model.load_state_dict(state_dict, strict=False)
        # assert all keys are loaded

        print(
            f"Head and tail of unexpected keys: {load_state.unexpected_keys[:5]} ... {load_state.unexpected_keys[-5:]}"
        )
        print(
            f"Head of tail of missing keys: {load_state.missing_keys[:5]} ... {load_state.missing_keys[-5:]}"
        )
        lora_list = []
        for name, param in model.pipe.dit.named_parameters():
            if "lora" in name:
                lora_list.append(name)
        accelerator.print(f"After loading model, lora parameters: {len(lora_list)},")
    else:
        accelerator.print(
            f"No model state found at {model_state_dir}, starting from scratch."
        )

    if args.add_lora:
        lora_model = model.add_lora_to_model(
            getattr(model.pipe, args.lora_base_model),
            target_modules=(
                args.lora_target_modules.split(",")
                if args.lora_target_modules
                else None
            ),
            adapter_name=args.lora_adapter_name,
            lora_rank=args.lora_rank,
        )
        setattr(model.pipe, args.lora_base_model, lora_model)

        lora_list = []
        lora_freeze = []
        lora_unfreeze = []
        trainable_param = []
        for name, param in model.pipe.dit.named_parameters():
            if "lora" in name:
                lora_list.append(name)
                if param.requires_grad:
                    lora_unfreeze.append(name)
                else:
                    lora_freeze.append(name)
            if param.requires_grad:
                trainable_param.append(name)
        accelerator.print(
            f"After adding lora to the model, lora parameters: {len(lora_list)}, trainable parameters: {len(trainable_param)}, lora freeze: {len(lora_freeze)}, lora unfreeze: {len(lora_unfreeze)}"
        )
    else:
        accelerator.print(
            f"No LoRA added to the model, training parameters: {sum(p.numel() for p in model.pipe.dit.parameters() if p.requires_grad)}"
        )
    # Worth mentioning that the peft add lora methods lead to all parameters but lora are freezed. So we manually set all param to be freezed then unfreeze it from the config we specify.

    # Step 1 freeze all param
    for name, param in model.pipe.dit.named_parameters():
        param.requires_grad = False

    def debug_print(model):
        trainable_list = []
        for name, param in model.pipe.dit.named_parameters():
            if param.requires_grad:
                trainable_list.append(name)
        accelerator.print(f"Trainable parameters: {len(trainable_list)}")

    # Step 2 unfreeze we want to train
    # Main branch
    if args.freeze_main_except == "all":
        for name, param in model.pipe.dit.named_parameters():
            if "blocks." in name and "control" not in name:
                param.requires_grad = True
    elif args.freeze_main_except == "lora":
        for name, param in model.pipe.dit.named_parameters():
            if "lora" in name and "blocks." in name and "control" not in name:
                param.requires_grad = True

    # Control branch
    if args.has_control_model:
        # if args.copy_control_weights:
        #     model.pipe.dit.copy_weights_from_main_branch()
        # model.pipe.dit.alter_camera_blocks()
        accelerator.print(f"Model has control model...")
        if args.freeze_control_except == "all":
            for name, param in model.pipe.dit.named_parameters():
                if "control_blocks." in name:
                    param.requires_grad = True
        elif args.freeze_control_except == "lora":
            for name, param in model.pipe.dit.named_parameters():
                if "lora" in name and "control_blocks." in name:
                    param.requires_grad = True
        # Also don't forget to unfreeze the zero init parameters
        if not args.freeze_zero_linear:
            for name, param in model.pipe.dit.named_parameters():
                if "zero_inits" in name:
                    param.requires_grad = True

    # Always check if there's any mistake
    lora_list = []
    lora_freeze = []
    lora_unfreeze = []
    trainable_param = []
    for name, param in model.pipe.dit.named_parameters():
        if "lora" in name:
            lora_list.append(name)
            if param.requires_grad:
                lora_unfreeze.append(name)
            else:
                lora_freeze.append(name)
        if param.requires_grad:
            trainable_param.append(name)
    accelerator.print(
        f"After adjust main and control branch, lora parameters: {len(lora_list)}, trainable parameters: {len(trainable_param)}, lora freeze: {len(lora_freeze)}, lora unfreeze: {len(lora_unfreeze)}"
    )
    accelerator.print(f"Trainable param:{trainable_param}")
    accelerator.print(
        f"Trainable parameters: {sum(p.numel() for p in model.pipe.dit.parameters() if p.requires_grad)}"
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    # Default set the ratio that train test has no overlap.
    dataset_name = args.dataset_name
    if dataset_name.lower() == "dl3dv":
        h, w = args.height, args.width
        resize_h, resize_w = args.train_height, args.train_width
        assert args.train_min_frame % 4 == 1, "train_min_frame should be 4n+1"
        assert args.test_min_frame % 4 == 1, "test_min_frame should be 4n+1"

        train_ratio = 0.98
        test_ratio = 1 - train_ratio
        train_dataset = ScenesDataset(
            no_extra_frame=args.no_extra_frame,
            max_frame=args.train_max_frame,
            min_frame=args.train_min_frame,
            min_sample_step=args.train_min_sample_stride,
            max_sample_step=args.train_max_sample_stride,
            ratio=train_ratio,
            relative_pose=True,
            split="train",
            patch_size=[w, h],  # W H
            resize_size=[resize_w, resize_h],  # W H
        )

        test_dataset = ScenesDataset(
            no_extra_frame=args.no_extra_frame,
            # max_frame=args.test_max_frame, # test only use the min frame as the max frame
            min_frame=args.test_min_frame,
            min_sample_step=args.test_min_sample_stride,
            max_sample_step=args.test_max_sample_stride,
            relative_pose=True,
            split="test",
            ratio=test_ratio,
            patch_size=[w, h],  # W H
            resize_size=[resize_w, resize_h],  # W H
        )
    elif dataset_name.lower() == "re10k":
        train_dataset = RealEstate10KPose(
            split="train",
            sample_stride=8,
            sample_n_frames=61,
            relative_pose=True,
            sample_size=[args.train_height, args.train_width],
            use_image_depth=args.use_image_depth,
            rescale_fxy=True,
            use_flip=False,
            no_extra_frame=True,
        )
        test_dataset = RealEstate10KPose(
            split="test",
            sample_stride=8,
            sample_n_frames=61,
            relative_pose=True,
            use_image_depth=args.use_image_depth,
            sample_size=[args.train_height, args.train_width],
            rescale_fxy=True,
            use_flip=False,
            no_extra_frame=True,
        )

    train_dataloader = torch.utils.data.DataLoader(
        # Assuming dataset returns a dict
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=6,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=4,
    )
    test_dataloader = torch.utils.data.DataLoader(
        # Assuming dataset returns a dict
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        prefetch_factor=2,
    )

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)

    world_size = accelerator.num_processes

    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=args.warmup_steps * world_size
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    model, optimizer, scheduler, train_dataloader, test_dataloader = (
        accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, test_dataloader
        )
    )

    start_epoch, global_step = 0, 0
    if args.load_optimizer:
        start_epoch, global_step = model_logger.load_training_state(
            accelerator=accelerator,
            dir=args.training_state_dir,
        )

    accelerator.wait_for_everyone()

    launch_training_task(
        accelerator=accelerator,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        start_epoch=start_epoch,
        global_step=global_step,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        model_logger=model_logger,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        validate_step=args.validate_step,
        log_step=args.log_step,
        args=args,
    )
