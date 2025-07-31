
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
            model_configs += [ModelConfig(model_id=i.split(
                ":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs,
            config_path=config_path,
        )
        from omegaconf import OmegaConf
        if config_path is not None:
            self.model_config = OmegaConf.load(config_path)

        # print(f"dit data type : {self.pipe.dit.parameters().__next__().dtype}")

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
            "num_frames": data.get("num_frames", None),
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
        # print(f"Num frames: {inputs_shared['num_frames']}")
        for unit in self.pipe.units:
            # print(f"Processing unit: {unit.__class__.__name__}")
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        # if inputs is None:
        inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name)
                  for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

    def validate(self, accelerator, global_step, test_dataloader=None, output_path=None):
        # This method is a placeholder for validation logic.
        # You can implement your validation logic here if needed.
        import numpy as np
        from typing import List, Optional
        from PIL import Image
        import cv2
        import torch
        from diffsynth import save_video
        assert test_dataloader is not None, "Test dataloader must be provided for validation."

        for idx, batch in enumerate(test_dataloader):
            if idx >= 4:
                break
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                input_data = get_data(batch)
                # mask the video every 10 frames

                print(
                    f"extra_image_frame_index: {input_data['extra_image_frame_index']}")
                if input_data is None:
                    continue
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
                save_path = os.path.join(output_path, "validation_results")
                os.makedirs(save_path, exist_ok=True)

                gt_video = input_data['images'].unsqueeze(
                    0).to(self.pipe.device)
                gt_video = gt_video.permute(0, 1, 3, 4, 2)  # B T H W C

                # 生成输入可视化视频：input_image + extra_images
                input_frames = [None] * input_data['num_frames']

                # 插入第0帧
                input_frames[0] = input_data['input_image']

                # 插入extra frames到指定位置
                if input_data['extra_images'] is not None and input_data['extra_image_frame_index'] is not None:
                    for img, frameidx in zip(input_data['extra_images'], input_data['extra_image_frame_index']):
                        if 0 <= frameidx < len(input_frames):
                            input_frames[frameidx] = img.unsqueeze(0)

                # 将空帧填充成最后一个有效帧（或黑图）
                last_valid = torch.zeros_like(input_frames[0])
                for i in range(len(input_frames)):
                    if input_frames[i] is None:
                        input_frames[i] = last_valid

                input_frames = torch.stack(input_frames, dim=0)

                input_frames = input_frames.permute(1, 0, 3, 4, 2)  # B T H W C
                input_frames = self.pipe.vae_output_to_video(
                    input_frames, pattern='B T H W C', min_value=0, max_value=1)
                gt_video = self.pipe.vae_output_to_video(
                    gt_video, pattern='B T H W C', min_value=0, max_value=1)

                input_save_path = os.path.join(
                    save_path,
                    f"video_input_{idx}.mp4"
                )
                gt_save_path = os.path.join(
                    save_path,
                    f"gt_video_{idx}.mp4"
                )
                if not os.path.exists(input_save_path):
                    save_video(
                        input_frames,
                        input_save_path,
                        fps=10,
                        quality=5,
                    )
                if not os.path.exists(gt_save_path):
                    save_video(
                        gt_video,
                        gt_save_path,
                        fps=10,
                        quality=5,
                    )
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
        import time
        # current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        # self.output_path = os.path.join(
        #     self.output_path, f"logs_{current_time}"
        # )
        os.makedirs(self.output_path, exist_ok=True)

        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def on_step_end(self, loss):
        print(
            f"Step loss: {loss.mean().item() if isinstance(loss, torch.Tensor) else loss}")

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
            f"Loaded training state from {meta_path}: epoch={epoch}, global_step={global_step}")
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
        # 先保存 meta_state 到一个临时文件
        meta_path = os.path.join(self.output_path, "meta_state.pt")
        if accelerator.is_main_process:
            torch.save(meta_state, meta_path)

        # 统一用 accelerator.save_state 保存所有状态到 output_path
        # 它会自动保存 model, optimizer, scheduler 等所有由 accelerate 管理的组件
        accelerator.save_state(self.output_path)

    # def load_training_state(self,accelerator, optimizer, scheduler, dir=None):
    #     state_path = os.path.join(dir, "meta_state.pt")
    #     optimizer_path = os.path.join(dir, "optimizer.pt")
    #     scheduler_path = os.path.join(dir, "scheduler.pt")

    #     if os.path.exists(state_path):
    #         print(f"Loading training state from {state_path}")
    #         meta = torch.load(state_path, map_location="cpu")
    #         epoch_id = meta["epoch"]
    #         global_step = meta["global_step"]

    #         if os.path.exists(optimizer_path):
    #             optimizer.load_state_dict(torch.load(
    #                 optimizer_path, map_location="cpu",weights_only=False))
    #         if os.path.exists(scheduler_path):
    #             scheduler.load_state_dict(torch.load(
    #                 scheduler_path, map_location="cpu",weights_only=False))

    #         return epoch_id, global_step
    #     else:
    #         print(f"No training state found at {state_path}.")
    #         return 0, 0

    # def load_latest_model_weights(self, model, ckpt_path):
    #     from safetensors.torch import load_file

    #     if not os.path.exists(ckpt_path):
    #         print(
    #             f"Checkpoint path {ckpt_path} does not exist. Skipping loading weights.")
    #         return

    #     state_dict = load_file(ckpt_path)
    #     model.pipe.dit.load_state_dict(state_dict, strict=True)
    #     print(f"Loaded model weights from {ckpt_path}")

    # def save_model_state(self, accelerator, model, epoch_id, global_step):
    #     accelerator.wait_for_everyone()
    #     model_config_save_path = os.path.join(
    #         self.output_path, "model_config.yaml")
    #     if accelerator.is_main_process and not os.path.exists(model_config_save_path):
    #         accelerator.print(
    #             f"Saving model configuration to {model_config_save_path}")
    #         from omegaconf import OmegaConf
    #         OmegaConf.save(config=model.model_config,
    #                        f=model_config_save_path)
    #     if accelerator.is_main_process:
    #         state_dict = accelerator.get_state_dict(model)
    #         state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
    #             state_dict, remove_prefix=self.remove_prefix_in_ckpt)
    #         # print(f"state_dict keys: {list(state_dict.keys())}")
    #         state_dict = self.state_dict_converter(state_dict)
    #         os.makedirs(self.output_path, exist_ok=True)
    #         path = os.path.join(
    #             self.output_path, f"epoch-{epoch_id}-step-{global_step}.safetensors")
    #         accelerator.save(state_dict, path, safe_serialization=True)

    # def save_training_state(self, accelerator, optimizer, scheduler, epoch_id, global_step):
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         accelerator.print("Saving training state...")
    #         os.makedirs(self.output_path, exist_ok=True)
    #         optimizer_save_path = os.path.join(
    #             self.output_path, "optimizer.pt")
    #         scheduler_save_path = os.path.join(
    #             self.output_path, "scheduler.pt")
    #         meta_state = {
    #             "epoch": epoch_id,
    #             "global_step": global_step,
    #         }
    #         meta_state_save_path = os.path.join(
    #             self.output_path, "meta_state.pt")
    #         accelerator.save(optimizer.state_dict(), optimizer_save_path)
    #         accelerator.save(scheduler.state_dict(), scheduler_save_path)
    #         accelerator.save(meta_state, meta_state_save_path)


def launch_training_task(
    accelerator,
    start_epoch,
    global_step,
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    validate_step: int = 500,

):
    train_dataloader = torch.utils.data.DataLoader(
        # Assuming dataset returns a dict
        train_dataset, shuffle=True, batch_size=1, num_workers=0, collate_fn=lambda x: x[0] if x else None, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=0, collate_fn=lambda x: x[0] if x else None,  pin_memory=True
    )

    # deepspeed_plugin = DeepSpeedPlugin(
    #     hf_ds_config='deepspeed.json'
    # )

    accelerator.print(
        f"Initial accelerator with gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    accelerator.print(
        f"Using {accelerator.num_processes} processes for training.")
    accelerator.print(
        f"Preparing model, optimizer, dataloader, and scheduler with accelerator.")
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    accelerator.print(
        f"Model param dtype : {next(model.pipe.dit.parameters()).dtype}")

    if global_step > 0:
        accelerator.print(
            f"Resuming training from epoch {start_epoch}, global step {global_step}")
    else:
        accelerator.print("Starting training from scratch.")

    accelerator.print(f"Validate every {validate_step} steps.")

    if accelerator.is_main_process:
        accelerator.print(
            f"Starting validation with model at epoch {start_epoch}, global step {global_step}")
        model.pipe.dit.eval()
        model.validate(
            accelerator=accelerator,
            global_step=global_step,
            test_dataloader=test_dataloader,
            output_path=model_logger.output_path,
        )
        model.pipe.scheduler.set_timesteps(1000, training=True)
        model.pipe.dit.train()
    accelerator.wait_for_everyone()
    for epoch_id in range(num_epochs):
        train_dataloader.set_epoch(epoch_id + start_epoch)
        for idx, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch_id + 1}/{num_epochs}", disable=not accelerator.is_main_process)):
            with accelerator.accumulate(model):
                input_data = get_data(data)
                optimizer.zero_grad()

                loss = model(input_data)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.print(
                        f"Loss at step {global_step}: {loss.mean().item()}")
                    accelerator.clip_grad_norm_(
                        model.trainable_modules(), max_norm=1.0)
                    global_step += 1
                    if (global_step) % validate_step == 0:
                        model_logger.save_training_state(
                            accelerator=accelerator,
                            epoch_id=epoch_id + start_epoch,
                            global_step=global_step,
                        )
                        accelerator.wait_for_everyone()
                        accelerator.print(
                            f"Checkpoint saved at step {global_step}")
                        if accelerator.is_main_process:
                            model.pipe.dit.eval()
                            model.validate(
                                accelerator=accelerator,
                                global_step=global_step,
                                test_dataloader=test_dataloader,
                                output_path=model_logger.output_path,
                            )
                            model.pipe.scheduler.set_timesteps(
                                1000, training=True)
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

    import builtins
    import os
    accelerator = Accelerator()
    set_seed(42, device_specific=True)
    parser = wan_parser()
    args = parser.parse_args()
    # dataset = VideoDataset(args=args)
    # export DEEPSPEED_LOG_LEVEL=debug
    # export DEEPSPEED_ZERO_LOG_LEVEL=debug
    model = WanTrainingModule(
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        config_path=args.model_config_path,
    )
    print(f"Model data type : {model.pipe.dit.parameters().__next__().dtype}")
    if args.train_only_camera:
        print("Training only the camera model.")
        for param in model.pipe.dit.parameters():
            param.requires_grad = False
        for param in model.pipe.dit.pose_encoder.parameters():
            param.requires_grad = True
        for name, param in model.pipe.dit.named_parameters():
            if '_zl_' in name:
                param.requires_grad = True
    # train_param = []
    # for name, param in model.pipe.named_parameters():
    #     if param.requires_grad:
    #         train_param.append(name)
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.pipe.dit.parameters() if p.requires_grad)}")
    # print the module needs grads
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} requires grad")

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,

    )
    # Default set the ratio that train test has no overlap.
    train_ratio = 0.98
    test_ratio = 1 - train_ratio
    h, w = args.height, args.width
    assert args.train_min_frame % 4 == 1, "train_min_frame should be 4n+1"
    assert args.test_min_frame % 4 == 1, "test_min_frame should be 4n+1"
    train_dataset = ScenesDataset(
        no_extra_frame=args.no_extra_frame,
        max_frame=args.train_max_frame,
        min_frame=args.train_min_frame,
        ratio=train_ratio,
        relative_pose=True,
        split='train',
        patch_size=[w, h],  # W H
    )

    test_dataset = ScenesDataset(
        no_extra_frame=args.no_extra_frame,
        # max_frame=args.test_max_frame, # test only use the min frame as the max frame
        min_frame=args.test_min_frame,
        relative_pose=True,
        split='test',
        ratio=test_ratio,
        patch_size=[w, h],  # W H
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    start_epoch, global_step = 0, 0
    start_epoch, global_step = model_logger.load_training_state(
        accelerator=accelerator,
        dir=args.training_state_dir,
    )
    accelerator.wait_for_everyone()

    launch_training_task(
        accelerator=accelerator,
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
    )
