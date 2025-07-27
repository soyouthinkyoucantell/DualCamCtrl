import argparse

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
    parser.add_argument("--trainable_models", type=str, default='dit,',
                        help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str,
                        default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int,
                        default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None,
                        help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=True,
                        action="store_true", help="Whether to use gradient checkpointing to save memory.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False,
                        action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--validate_step", type=int, default=1000,
                        help="Save model every N steps.")

    return parser