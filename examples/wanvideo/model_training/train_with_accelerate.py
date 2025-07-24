# from .....dataset.colmap_debug import Dataset
# from ...dataset.colmap_debug import Dataset

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
from accelerate import Accelerator

import torch
import os
import json
from diffsynth.pipelines.wan_video_new_altered import WanVideoPipeline, ModelConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        self.extra_inputs = extra_inputs.split(
            ",") if extra_inputs is not None else []

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            # "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            'input_image': data.get("input_image", None),
            'extra_images': data.get("extra_images", None),
            'extra_image_frame_index': data.get("extra_image_frame_index", None),
            'control_video': data.get("control_video", None),
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

        # # Extra inputs
        # for extra_input in self.extra_inputs:
        #     if extra_input == "input_image":
        #         inputs_shared["input_image"] = data["video"][0]
        #     elif extra_input == "end_image":
        #         inputs_shared["end_image"] = data["video"][-1]
        #     else:
        #         inputs_shared[extra_input] = data[extra_input]

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):

        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name)
                  for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def on_step_end(self, loss):
        pass

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(
                self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
):
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0])
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler)

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(loss)
                scheduler.step()
        model_logger.on_epoch_end(accelerator, model, epoch_id)


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


def wan_parser():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="",
                        required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str,
                        default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720,
                        help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None,
                        help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None,
                        help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video",
                        help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1,
                        help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int,
                        default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str,
                        default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str,
                        default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None,
                        help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str,
                        default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int,
                        default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None,
                        help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False,
                        action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1, help="Gradient accumulation steps.")
    return parser


def flux_parser():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="",
                        required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str,
                        default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024,
                        help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None,
                        help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None,
                        help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image",
                        help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1,
                        help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int,
                        default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str,
                        default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str,
                        default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None,
                        help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str,
                        default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int,
                        default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None,
                        help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true",
                        help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False,
                        action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False,
                        action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1, help="Gradient accumulation steps.")
    return parser


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
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    dataset = ScenesDataset(
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
