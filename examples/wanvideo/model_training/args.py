import argparse


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # Model
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--model_weight_path",
        type=str,
        default=None,
        help="Path to the model weight file. If not provided, the model will be initialized from the config.",
    )
    parser.add_argument(
        "--model_id_with_origin_paths",
        type=str,
        default=None,
        help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.",
    )
    parser.add_argument(
        "--pipe_name", type=str, choices=["wan_video_camera", "wan_video_inp"]
    )
    # Train
    parser.add_argument(
        "--training_state_dir",
        type=str,
        default=None,
        help="Path to the training state directory. If not provided, a new directory will be created.",
    )

    parser.add_argument(
        "--output_path", type=str, default="./models", help="Output save path."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10000, help="Number of epochs."
    )
    parser.add_argument(
        "--remove_prefix_in_ckpt",
        type=str,
        default="pipe.dit.",
        help="Remove prefix in ckpt.",
    )

    # Model to train
    parser.add_argument(
        "--trainable_models",
        type=str,
        default="dit,",
        help="Models to train, e.g., dit, vae, text_encoder.",
    )
    parser.add_argument(
        "--return_control_latents",
        action="store_true",
        default=False,
        help="Whether to return control latent in the model. If set, the model will return control latent.",
    )
    # Ckpt lora
    parser.add_argument(
        "--ckpt_has_lora",
        action="store_true",
        default=False,
        help="Whether the checkpoint has LoRA weights.",
    )
    parser.add_argument(
        "--ckpt_lora_base_model",
        type=str,
        default="dit",
        help="Which model LoRA is added to.",
    )
    parser.add_argument(
        "--ckpt_lora_module",
        default="q,k,v,o,ffn.0,ffn.2",
        help="Which layers LoRA is added to in the checkpoint.",
        type=str,
    )
    parser.add_argument(
        "--ckpt_lora_rank",
        type=int,
        default=32,
        help="Rank of LoRA in the checkpoint.",
    )
    parser.add_argument(
        "--ckpt_lora_adapter_name",
        type=str,
        default="default",
        help="Adapter name for LoRA in the checkpoint.",
    )

    # If add LoRA
    parser.add_argument(
        "--add_lora",
        default=False,
        action="store_true",
        help="Whether to add LoRA to the model. If set, LoRA will be added to the specified base model.",
    )
    parser.add_argument(
        "--lora_base_model",
        type=str,
        default=None,
        help="Which model LoRA is added to.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Which layers LoRA is added to.",
    )
    parser.add_argument(
        "--lora_adapter_name",
        type=str,
        default="control",
        help="Adapter name for LoRA. If not specified, the default adapter name will be used.",
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument(
        "--load_optimizer",
        action="store_true",
        default=False,
        help="Whether to load optimizer state from checkpoint.",
    )

    # For the main branch
    parser.add_argument(
        "--freeze_main_except",
        choices=["none", "lora", "all"],
        default="none",
        help="Whether to freeze the main model weights except for LoRA or all weights.",
    )

    # For the control branch
    parser.add_argument(
        "--has_control_model",
        action="store_true",
        default=False,
        help="Whether the model has a control branch.",
    )

    parser.add_argument(
        "--freeze_control_except",
        choices=["none", "lora", "all"],
        default="none",
        help="Whether to freeze the control model weights except for LoRA or all weights.",
    )
    parser.add_argument(
        "--copy_control_weights",
        action="store_true",
        default=False,
        help="Whether to copy weights from the main model to the control model at the beginning of training.",
    )  # Enable this when the control model is not initialized with any weights.
    parser.add_argument(
        "--freeze_zero_linear",
        action="store_true",
        default=False,
        help="Whether to freeze the zero linear layers in the control model.",
    )

    # Training
    parser.add_argument(
        "--validate_step",
        type=int,
        default=1000,
        help="Validate and save model every N steps.",
    )
    parser.add_argument(
        "--log_step",
        type=int,
        default=10,
        help="Log training information every N steps.",
    )

    parser.add_argument(
        "--validate_batch",
        type=int,
        default=1,
        help="How many batch to use for validation.",
    )

    parser.add_argument(
        "--use_gradient_checkpointing",
        default=True,
        action="store_true",
        help="Whether to use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--no_extra_frame",
        default=False,
        action="store_true",
        help="Whether to only use the first frame of the video for training.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for the learning rate scheduler.",
    )

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["re10k", "dl3dv"],
        default="dl3dv",
        help="Dataset name to use for training. Choose from 're10k' or 'dl3dv'.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--train_height",
        type=int,
        default=None,
        help="Height of images or videos for training. Leave `train_height` and `train_width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--train_width",
        type=int,
        default=None,
        help="Width of images or videos for training. Leave `train_height` and `train_width` empty to enable dynamic resolution.",
    )

    parser.add_argument(
        "--use_image_depth",
        action="store_true",
        default=False,
        help="Whether to use image depth as input",
    )

    # dl3dv dataset
    parser.add_argument(
        "--train_max_frame",
        type=int,
        default=49,
        help="Maximum number of frames to use for training. If set to 0, all frames will be used.",
    )
    parser.add_argument(
        "--train_min_frame",
        type=int,
        default=5,
        help="Minimum number of frames to use for training. If set to 0, all frames will be used.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--test_min_frame",
        type=int,
        default=5,
        help="Minimum number of frames to use for testing. If set to 0, all frames will be used.",
    )
    parser.add_argument(
        "--init_validate",
        default=False,
        action="store_true",
        help="Whether to validate the model before training.",
    )
    parser.add_argument(
        "--train_min_sample_stride",
        default=4,
        type=int,
        help="Minimum sample stride for training. If set to 0, all samples will be used.",
    )
    parser.add_argument(
        "--train_max_sample_stride",
        default=8,
        type=int,
        help="Maximum sample stride for training. If set to 0, all samples will be used.",
    )
    parser.add_argument(
        "--test_min_sample_stride",
        default=6,
        type=int,
        help="Minimum sample stride for testing. If set to 0, all samples will be used.",
    )
    parser.add_argument(
        "--test_max_sample_stride",
        default=6,
        type=int,
        help="Maximum sample stride for testing. If set to 0, all samples will be used.",
    )

    # re10k dataset

    return parser