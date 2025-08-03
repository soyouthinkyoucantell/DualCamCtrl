import argparse


def wan_parser():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--model_config_path", type=str,
                        required=True, help="Path to the model configuration file.")
    parser.add_argument('--model_weight_path', type=str, default=None,
                        help="Path to the model weight file. If not provided, the model will be initialized from the config.")
    parser.add_argument('--training_state_dir', type=str, default=None,
                        help="Path to the training state directory. If not provided, a new directory will be created.")
    parser.add_argument("--height", type=int, default=None,
                        help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None,
                        help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--output_path", type=str,
                        default="./models", help="Output save path.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int,
                        default=1, help="Number of epochs.")

    parser.add_argument("--remove_prefix_in_ckpt", type=str,
                        default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default='dit,',
                        help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--validate_step", type=int, default=1000,
                        help="Validate and save model every N steps.")
    parser.add_argument("--use_gradient_checkpointing", default=True,
                        action="store_true", help="Whether to use gradient checkpointing to save memory.")
    parser.add_argument("--no_extra_frame", default=False, action="store_true",
                        help="Whether to only use the first frame of the video for training.")
    parser.add_argument("--train_only_camera", default=False, action="store_true",
                        help="Whether to train only the camera model.")
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help="Number of warmup steps for the learning rate scheduler.") 
    # Dataset
    parser.add_argument("--train_max_frame", type=int, default=49,
                        help="Maximum number of frames to use for training. If set to 0, all frames will be used.")
    parser.add_argument("--train_min_frame", type=int, default=5,
                        help="Minimum number of frames to use for training. If set to 0, all frames will be used.")
    parser.add_argument("--test_min_frame", type=int, default=5,
                        help="Minimum number of frames to use for testing. If set to 0, all frames will be used.")
    
    return parser
