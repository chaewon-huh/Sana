#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SanaPipeline, SanaTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Gemma2Model


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


class ImageConditioningProjector(nn.Module):
    def __init__(self, vae_latent_channels: int, transformer_pooled_projection_dim: int, hidden_dim_ratio: int = 4):
        super().__init__()
        # Simplified projector: average pool VAE latents and project.
        # A more complex projector might use Conv2D layers.
        # Assuming VAE latents are (B, C, H, W)
        # pooled_projection_dim is typically text_encoder_hidden_size for Sana/SD3
        self.proj_in = vae_latent_channels
        self.proj_out = transformer_pooled_projection_dim
        
        # Using a few conv layers to reduce spatial dimensions and extract features
        # This example assumes VAE output might be 64x64 or 128x128 in latents
        # For example, if VAE latent channels = 4 (like SD VAE)
        # And transformer_pooled_projection_dim = 4096 (like Gemma2 Large)
        
        # Using a simple MLP on spatially averaged features for now
        # More sophisticated designs (e.g. ResNet blocks, attention) can be used here.
        intermediate_dim = vae_latent_channels * hidden_dim_ratio 

        self.fc1 = nn.Linear(self.proj_in, intermediate_dim)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(intermediate_dim, self.proj_out)


    def forward(self, x):
        # x shape: (batch_size, latent_channels, height, width)
        # Global average pooling
        x = torch.mean(x, dim=[2, 3]) # (batch_size, latent_channels)
        x = self.act1(self.fc1(x))
        x = self.fc2(x) # (batch_size, proj_out_dim)
        return x


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    text_prompt=None, # Changed from instance_prompt
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Sana Img2Img LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} Img2Img LoRA weights for {base_model}.
The LoRA weights are for the transformer, and an additional image conditioning projector is also trained.

The weights were trained using the [Sana diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sana.md) (modified for img2img).

## Trigger words

You should use a text prompt like `{text_prompt if text_prompt else "a photo of a cat"}` along with an input image.

## Download model

[Download the *.safetensors LoRA and image_projector.safetensors]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
# TODO: Add usage example for img2img with the projector
from diffusers import SanaPipeline, AutoencoderKL
from PIL import Image
import torch

# Load base model
pipeline = SanaPipeline.from_pretrained("your-base-sana-model", torch_dtype=torch.bfloat16)
pipeline.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16) # Example VAE
pipeline.to("cuda")

# Load LoRA weights for transformer and the image projector
pipeline.load_lora_weights("{repo_id}", weight_name="pytorch_lora_weights.safetensors") # Or the specific safetensors file for transformer LoRA

# Load image projector weights (assuming it's saved separately)
image_projector_weights_path = "{repo_id}/image_projector.safetensors" # Adjust path as needed
# This part needs a custom ImageConditioningProjector class definition identical to the training one.
# For simplicity, this example assumes it's directly loaded into a compatible attribute of the pipeline or handled manually.
# image_projector = ImageConditioningProjector(...) 
# image_projector.load_state_dict(torch.load(image_projector_weights_path))
# image_projector.to("cuda", dtype=torch.bfloat16)


# Prepare input image and text prompt
input_image = Image.open("your_input_image.png").convert("RGB")
prompt = "{text_prompt if text_prompt else "A fantasy landscape"}"

# Preprocess input image and get conditional embeddings (simplified)
# This part needs to replicate the conditioning logic from training:
# 1. VAE encode input_image -> condition_latents
# 2. image_projector(condition_latents) -> image_pooled_embeds
# 3. text_pipeline.encode_prompt(prompt) -> text_prompt_embeds (sequence), text_pooled_embeds
# 4. combined_pooled_embeds = text_pooled_embeds + image_pooled_embeds (or other combination strategy)

# Example of how you might pass pooled embeddings if pipeline supports it:
# (This is a hypothetical modification to pipeline usage, actual implementation depends on pipeline design)
# For Sana, the pooled_prompt_embeds are passed to the transformer.

# Placeholder for actual generation logic using the projector:
# with torch.no_grad():
#    condition_latents = pipeline.vae.encode(pipeline.image_processor.preprocess(input_image).to(pipeline.device, dtype=torch.bfloat16)).latent_dist.sample()
#    condition_latents = condition_latents * pipeline.vae.config.scaling_factor
#    image_pooled_embeds = image_projector(condition_latents)
#
#    prompt_embeds, _, text_pooled_embeds, _ = pipeline.encode_prompt(prompt, device=pipeline.device, num_images_per_prompt=1, do_classifier_free_guidance=False) # Assuming CFG is handled or not used
#    
#    combined_pooled_embeds = text_pooled_embeds + image_pooled_embeds # Example combination

#    # The pipeline's __call__ might need to accept precomputed pooled_prompt_embeds
#    # or you might need to modify the pipeline's internal _prepare_text_encoder_inputs or similar
#    image = pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds=combined_pooled_embeds, generator=torch.manual_seed(0)).images[0]


# For now, a simple text-to-image generation with LoRA (doesn't show img2img part):
image = pipeline(prompt, generator=torch.manual_seed(0)).images[0]
image.save("generated_image_with_lora.png")

```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

TODO
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other", # TODO: consider license
        base_model=base_model,
        prompt=text_prompt, # Using generic text_prompt
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image", # Should be img-to-img once properly set up
        "image-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "sana",
        "sana-diffusers",
        "template:sd-lora", # Potentially a new template for img2img lora
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline, # This is the base SanaPipeline
    vae,
    image_conditioning_projector,
    text_encoding_pipeline, # Separate text encoding pipeline for flexibility
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt} and input image: {args.validation_input_image_path}."
    )
    
    # Ensure models are on the correct device and in eval mode
    image_conditioning_projector.to(accelerator.device, dtype=torch.bfloat16) # Assuming bf16 for projector
    image_conditioning_projector.eval()
    
    # vae is already prepared and on device by main training loop before validation
    # pipeline.transformer is already unwrapped and has LoRA weights loaded

    pipeline = pipeline.to(accelerator.device) # Base pipeline
    pipeline.set_progress_bar_config(disable=True)

    if args.validation_input_image_path is None:
        logger.warning("`validation_input_image_path` not provided. Skipping img2img validation.")
        return []

    try:
        validation_input_image = Image.open(args.validation_input_image_path).convert("RGB").resize((args.resolution, args.resolution))
    except FileNotFoundError:
        logger.error(f"Validation input image {args.validation_input_image_path} not found. Skipping validation.")
        return []

    # Preprocess input image for VAE
    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    input_image_tensor = image_transforms(validation_input_image).unsqueeze(0).to(device=accelerator.device, dtype=vae.dtype)


    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    images = []

    with torch.no_grad():
        # 1. Encode input image with VAE
        condition_latents = vae.encode(input_image_tensor).latent_dist.sample() * vae.config.scaling_factor
        
        # 2. Project condition latents
        # Ensure projector is in eval mode and on correct device (done above)
        image_pooled_embeds = image_conditioning_projector(condition_latents.to(image_conditioning_projector.fc1.weight.dtype)) # Match projector's dtype

        # 3. Encode text prompt
        # text_encoding_pipeline should be on device and correct dtype
        text_encoding_pipeline.to(accelerator.device) # Ensure it's on device
        prompt_embeds_seq, prompt_attention_mask, text_pooled_embeds, _ = text_encoding_pipeline.encode_prompt(
            args.validation_prompt,
            device=accelerator.device,
            num_images_per_prompt=1, # num_validation_images handled by loop
            do_classifier_free_guidance=False, # Assuming no CFG for simplicity in validation, or needs negative prompts
            max_sequence_length=args.max_sequence_length,
            complex_human_instruction=args.complex_human_instruction,
        )
        
        # 4. Combine pooled embeddings
        # Ensure dtypes match before addition
        combined_pooled_embeds = text_pooled_embeds.to(image_pooled_embeds.dtype) + image_pooled_embeds
        
        pipeline_args = {
            "prompt_embeds": prompt_embeds_seq,
            "pooled_prompt_embeds": combined_pooled_embeds,
            "generator": generator,
            # Add other necessary args for SanaPipeline like height, width, num_inference_steps
            "height": args.resolution,
            "width": args.resolution,
            "num_inference_steps": 25, # Example, make it an arg
        }

        for _ in range(args.num_validation_images):
            # The transformer used by the pipeline already has LoRA weights
            img = pipeline(**pipeline_args).images[0]
            images.append(img)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(f"{phase_name}_img2img", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    f"{phase_name}_img2img": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt} (img2img)") for i, image in enumerate(images)
                    ]
                }
            )
    
    # Move projector back to CPU if offloading strategy is used, or keep on device
    # For simplicity, assume it stays on device or handled by accelerator

    # Don't del pipeline here as it's passed in and might be used again if not final validation.
    # Let the caller manage its lifecycle or manage it based on is_final_validation.
    # if is_final_validation:
    # del pipeline # This pipeline is a copy with LoRA, safe to delete if locally created.
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data (image pairs and prompts)."
            " It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_pairs_dir", # New argument for local image pairs
        type=str,
        default=None,
        help=("A folder containing the training data as image pairs (e.g., input_0.png, target_0.png, prompt_0.txt). "
              "Ignored if --dataset_name is provided."),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    # Columns for HF datasets
    parser.add_argument(
        "--input_image_column", type=str, default="input_image", help="Column name for input images in HF Dataset."
    )
    parser.add_argument(
        "--target_image_column", type=str, default="target_image", help="Column name for target images in HF Dataset."
    )
    parser.add_argument(
        "--caption_column", type=str, default="prompt", help="Column name for text prompts in HF Dataset."
    )
    # End Columns for HF datasets
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=300, # Keep as in base
        help="Maximum sequence length to use with with the Gemma model for text prompts.",
    )
    parser.add_argument(
        "--complex_human_instruction", # Keep as in base
        type=str,
        default=None,
        help="Instructions for complex human attention for text prompts: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A text prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_input_image_path", # New argument for validation
        type=str,
        default=None,
        help="Path to an input image to use for img2img validation.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=10, # Reduced from 50 for faster feedback
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices for the transformer."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sana-img2img-lora", # Changed default output dir
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512, # Keep as in base, or 1024 for newer models
        help="The resolution for input and target images.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop images to the resolution. If not set, images are resized and randomly cropped.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    # sample_batch_size not directly applicable as we don't sample class images
    parser.add_argument("--num_train_epochs", type=int, default=100) # Example epochs
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=('Whether training should be resumed from a previous checkpoint. Use a path or "latest".'),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing for the transformer.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4, # Common LoRA LR
        help="Initial learning rate for LoRA weights and image projector.",
    )
    parser.add_argument(
        "--learning_rate_projector", # Optional: separate LR for projector
        type=float,
        default=None, # If None, uses main learning_rate
        help="Learning rate for the image conditioning projector. Defaults to `learning_rate`.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=('The scheduler type. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles", type=int, default=1, help="Number of hard resets for cosine_with_restarts."
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor for polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--weighting_scheme", type=str, default="none", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Weighting scheme for timesteps (Sana/SD3 specific)."
    )
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit_normal weighting.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Std for logit_normal weighting.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale for mode weighting.")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Prodigy"])
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit AdamW.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--prodigy_beta3", type=float, default=None)
    parser.add_argument("--prodigy_decouple", type=bool, default=True)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay for UNet/Transformer LoRA and projector.")
    # adam_weight_decay_text_encoder is not relevant if text_encoder is frozen
    parser.add_argument(
        "--lora_layers", type=str, default=None, help='Transformer modules for LoRA, e.g., "to_k,to_q,to_v".'
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True)
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Hub token.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository name on Hub.")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs.")
    parser.add_argument("--cache_latents", action="store_true", default=False, help="Cache VAE latents for target images.")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "comet_ml", "all"])
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument(
        "--upcast_before_saving", action="store_true", default=False, help="Upcast transformer LoRA to fp32 before saving."
    )
    parser.add_argument(
        "--offload", action="store_true", help="Offload VAE and text encoder to CPU when not used (during training step)."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.image_pairs_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--image_pairs_dir`")

    if args.dataset_name is not None and args.image_pairs_dir is not None:
        warnings.warn("Both `--dataset_name` and `--image_pairs_dir` were specified. Ignoring `--image_pairs_dir` and using `--dataset_name`.")
        args.image_pairs_dir = None # Prioritize dataset_name

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
        
    if args.learning_rate_projector is None:
        args.learning_rate_projector = args.learning_rate

    return args


class Img2ImgDataset(Dataset):
    def __init__(
        self,
        dataset_name=None,
        dataset_config_name=None,
        image_pairs_dir=None,
        caption_column="prompt",
        input_image_column="input_image",
        target_image_column="target_image",
        tokenizer=None, # Not used here, prompts are handled in main loop
        size=512,
        center_crop=False,
        random_flip=False,
        cache_dir=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.tokenizer = tokenizer # Unused currently in __getitem__

        self.input_images = []
        self.target_images = []
        self.prompts = []

        if dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Please install the datasets library: `pip install datasets`.")
            
            dataset = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir)["train"] # Assuming 'train' split
            column_names = dataset.column_names

            if input_image_column not in column_names:
                raise ValueError(f"--input_image_column '{input_image_column}' not found in dataset: {column_names}")
            if target_image_column not in column_names:
                raise ValueError(f"--target_image_column '{target_image_column}' not found in dataset: {column_names}")
            if caption_column not in column_names:
                raise ValueError(f"--caption_column '{caption_column}' not found in dataset: {column_names}")

            self.input_images = dataset[input_image_column]
            self.target_images = dataset[target_image_column]
            self.prompts = dataset[caption_column]

        elif image_pairs_dir is not None:
            image_pairs_dir = Path(image_pairs_dir)
            if not image_pairs_dir.exists():
                raise ValueError(f"image_pairs_dir '{image_pairs_dir}' does not exist.")

            # Discover pairs: input_XXX.png, target_XXX.png, prompt_XXX.txt
            # A more robust way would be to list all inputs, and find corresponding targets/prompts.
            # This simple glob assumes a consistent naming pattern.
            input_files = sorted(list(image_pairs_dir.glob("*_input.*"))) # Supports .png, .jpg etc.
            for input_file_path in input_files:
                base_name = input_file_path.name.replace("_input.", ".") #  image_001_input.png -> image_001.png
                
                # Try to find target and prompt based on common extensions
                found_target = None
                for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    target_file_path = image_pairs_dir / base_name.replace(Path(base_name).suffix, f"_target{ext}")
                    if target_file_path.exists():
                        found_target = target_file_path
                        break
                
                found_prompt_text = None
                prompt_file_path = image_pairs_dir / base_name.replace(Path(base_name).suffix, "_prompt.txt")
                if prompt_file_path.exists():
                    with open(prompt_file_path, "r", encoding="utf-8") as pf:
                        found_prompt_text = pf.read().strip()
                
                if found_target and found_prompt_text is not None: # Ensure prompt is found, even if empty
                    self.input_images.append(Image.open(input_file_path))
                    self.target_images.append(Image.open(found_target))
                    self.prompts.append(found_prompt_text)
                else:
                    logger.warning(f"Skipping {input_file_path}, missing corresponding target image or prompt file.")
            
            if not self.input_images:
                raise ValueError(f"No valid image pairs found in {image_pairs_dir}. "
                                 "Expected files like 'name_input.png', 'name_target.png', 'name_prompt.txt'.")
        else:
            raise ValueError("Either dataset_name or image_pairs_dir must be provided.")

        self.num_images = len(self.input_images)
        self._length = self.num_images

        # Image transformations
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.5) if random_flip else nn.Identity(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # Special transform for input images if random crop should be consistent (not implemented here, use same seed or fixed crop)
        # For now, using the same random transform for both.
        self.input_image_transforms = self.image_transforms 
        self.target_image_transforms = self.image_transforms


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        input_img = self.input_images[index % self.num_images]
        target_img = self.target_images[index % self.num_images]
        prompt_text = self.prompts[index % self.num_images]

        input_img = exif_transpose(input_img)
        target_img = exif_transpose(target_img)

        if not input_img.mode == "RGB":
            input_img = input_img.convert("RGB")
        if not target_img.mode == "RGB":
            target_img = target_img.convert("RGB")
        
        # Apply transforms
        # If center_crop=False, RandomCrop might select different crops for input and target if not careful.
        # For simplicity here, they are independent. For paired random crops, apply together or seed.
        # Or, resize first, then apply the *same* crop parameters.
        # For simplicity, we apply resize then potentially different random crops.
        # If a fixed relationship is needed (e.g. same crop), this needs adjustment.
        
        # Option 1: Independent random crops (current simple implementation)
        example["input_pixel_values"] = self.input_image_transforms(input_img)
        example["target_pixel_values"] = self.target_image_transforms(target_img)
        
        # Option 2: Consistent random crops if center_crop is False
        # if not self.center_crop:
        #     # Resize first
        #     resize_transform = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        #     input_img_resized = resize_transform(input_img)
        #     target_img_resized = resize_transform(target_img)
            
        #     # Get crop parameters once (e.g., for input_img) and apply to both
        #     # This assumes input_img and target_img are of same size after resize, or adapt params
        #     # For simplicity, this example doesn't implement shared random crop params.
        #     # The current `self.image_transforms` will apply independent random crops if center_crop=False
        #     # which might be okay for some img2img tasks, or undesirable for others.
        #     # To ensure same crop:
        #     # i, j, h, w = transforms.RandomCrop.get_params(input_img_resized, output_size=(self.size, self.size))
        #     # input_img_cropped = crop(input_img_resized, i, j, h, w)
        #     # target_img_cropped = crop(target_img_resized, i, j, h, w) 
        #     # # Then ToTensor and Normalize
        #     # pixel_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        #     # example["input_pixel_values"] = pixel_transform(input_img_cropped)
        #     # example["target_pixel_values"] = pixel_transform(target_img_cropped)
        # else: # center_crop is True
        #     example["input_pixel_values"] = self.input_image_transforms(input_img)
        #     example["target_pixel_values"] = self.target_image_transforms(target_img)

        example["prompt"] = prompt_text
        return example


def collate_fn(examples):
    input_pixel_values = torch.stack([example["input_pixel_values"] for example in examples])
    target_pixel_values = torch.stack([example["target_pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]

    input_pixel_values = input_pixel_values.to(memory_format=torch.contiguous_format).float()
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_pixel_values": input_pixel_values,
        "target_pixel_values": target_pixel_values,
        "prompts": prompts,
    }
    return batch


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Cannot use both --report_to=wandb and --hub_token.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("bfloat16 mixed precision is not supported on MPS.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # May need to set False if projector causes issues
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb to use it for logging.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub and accelerator.is_main_process:
        repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, private=True).repo_id
    
    # Load tokenizer, text_encoder, vae, transformer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, cache_dir=args.cache_dir,
    )
    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)


    # Initialize ImageConditioningProjector
    # VAE latent channels: vae.config.latent_channels (e.g., 4 for SD VAE, 16 for Sana DC VAE)
    # Transformer pooled_projection_dim: transformer.config.pooled_projection_dim (e.g. 4096 for Gemma2 Large based Sana)
    if not hasattr(vae.config, 'latent_channels'):
        logger.warning("VAE config does not have latent_channels, trying to infer from a sample encoding.")
        try:
            sample_image = torch.randn(1, 3, args.resolution, args.resolution).to(text_encoder.dtype) # temp device & dtype
            sample_latents = vae.encode(sample_image.to(vae.device, dtype=vae.dtype)).latent_dist.sample()
            vae.config.latent_channels = sample_latents.shape[1]
            logger.info(f"Inferred VAE latent channels: {vae.config.latent_channels}")
        except Exception as e:
            raise ValueError(f"Could not determine VAE latent_channels: {e}. Please ensure VAE is compatible or set it manually.")

    if not hasattr(transformer.config, 'pooled_projection_dim') or transformer.config.pooled_projection_dim is None:
         # Infer from text_encoder's hidden_size if text_projection exists and maps to it.
         # SanaPipeline uses text_encoder output[0] (last_hidden_state) and then a text_projection
         # The output of text_projection is transformer.config.pooled_projection_dim
         # Let's assume text_encoder.config.hidden_size if text_projection is a simple linear
         # This needs to be robust. For Sana default, it's likely text_encoder.config.hidden_size (e.g. 4096 for Gemma 7B)
        if hasattr(text_encoder.config, 'hidden_size'):
             transformer.config.pooled_projection_dim = text_encoder.config.hidden_size 
             logger.warning(f"transformer.config.pooled_projection_dim not set. Assuming text_encoder.config.hidden_size: {transformer.config.pooled_projection_dim}")
        else:
            raise ValueError("Cannot determine transformer.config.pooled_projection_dim for ImageConditioningProjector.")


    image_conditioning_projector = ImageConditioningProjector(
        vae_latent_channels=vae.config.latent_channels,
        transformer_pooled_projection_dim=transformer.config.pooled_projection_dim
    )

    # Freeze VAE, text_encoder, and transformer main weights
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False) # LoRA will unfreeze parts of it

    # Text encoding pipeline (kept on CPU if offloading)
    text_encoding_pipeline = SanaPipeline.from_pretrained( # Lightweight pipeline for text encoding
        args.pretrained_model_name_or_path,
        vae=None, # VAE not needed for text encoding part
        transformer=None, # Transformer not needed
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16, # Gemma2 likes bfloat16
        cache_dir=args.cache_dir,
    )
    # text_projection layer from this pipeline is what determines pooled_prompt_embeds dimension.
    # Ensure projector's output matches: transformer.config.pooled_projection_dim
    # which is typically text_encoding_pipeline.text_projection.out_features


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("bfloat16 is not supported on MPS.")

    # VAE usually fp32 or fp16 for SDXL VAE. Sana's DC VAE might also be fp32.
    # vae.to(dtype=torch.float32) # Keep VAE in specific precision
    transformer.to(accelerator.device, dtype=weight_dtype) # Base transformer
    text_encoder.to(dtype=torch.bfloat16) # Gemma is bf16
    image_conditioning_projector.to(accelerator.device, dtype=weight_dtype) # Projector matches training precision

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        # image_conditioning_projector.enable_gradient_checkpointing() # If model is complex

    # Add LoRA adapter to transformer
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else: # Default target modules for Sana might differ, use common ones for transformers
        target_modules = ["to_q", "to_v", "to_k", "to_out.0", "ff.1", "ff.2"] # Example, check SanaTransformer modules
        logger.info(f"No lora_layers specified, using default: {target_modules}")


    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    logger.info(f"Added LoRA adapter to transformer with rank {args.rank} for modules: {target_modules}")


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Custom saving & loading hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            image_projector_state_dict_to_save = None

            for model_idx, model in enumerate(models):
                # Important: unwrap model before checking type
                unwrapped_model_for_type_check = unwrap_model(model) 
                if isinstance(unwrapped_model_for_type_check, SanaTransformer2DModel):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model) # Pass wrapped model to get_peft
                elif isinstance(unwrapped_model_for_type_check, ImageConditioningProjector):
                    image_projector_state_dict_to_save = unwrap_model(model).state_dict() # Get state_dict from unwrapped
                else:
                    logger.warning(f"Unexpected model type in save_model_hook: {type(unwrapped_model_for_type_check)}")
                
                # Pop weights to prevent double saving by accelerator
                # Only pop if we are handling the saving explicitly
                if isinstance(unwrapped_model_for_type_check, SanaTransformer2DModel) or \
                   isinstance(unwrapped_model_for_type_check, ImageConditioningProjector):
                    weights.pop(model_idx)


            if transformer_lora_layers_to_save is not None:
                SanaPipeline.save_lora_weights( # This saves only transformer LoRA
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )
                logger.info(f"Saved transformer LoRA weights to {output_dir}")
            
            if image_projector_state_dict_to_save is not None:
                projector_save_path = os.path.join(output_dir, "image_projector.safetensors")
                # Use safetensors if available, otherwise torch.save
                try:
                    from safetensors.torch import save_file
                    save_file(image_projector_state_dict_to_save, projector_save_path)
                except ImportError:
                    torch.save(image_projector_state_dict_to_save, projector_save_path.replace(".safetensors", ".bin"))
                logger.info(f"Saved image conditioning projector weights to {projector_save_path}")


    def load_model_hook(models, input_dir):
        transformer_ = None
        image_projector_ = None

        # Identify models from the list (accelerator might reorder them)
        temp_models = list(models) # Use a copy for iteration while popping
        
        # Clear original models list provided by accelerator. We will re-populate it if needed or let accelerator handle what's left.
        # This part is tricky with accelerator. It expects models to be loaded in place.
        # models.clear() # This might be too aggressive.

        loaded_transformer = False
        loaded_projector = False

        for model_idx in range(len(temp_models) -1, -1, -1): # Iterate backwards for safe pop
            model = temp_models[model_idx]
            unwrapped_model_for_type_check = unwrap_model(model)

            if isinstance(unwrapped_model_for_type_check, SanaTransformer2DModel) and not loaded_transformer:
                transformer_ = model # This is the model instance from accelerator.prepare()
                try:
                    lora_state_dict = SanaPipeline.lora_state_dict(input_dir) # Expects transformer LoRA weights
                    # The lora_state_dict might have "transformer." prefix, adjust if needed by set_peft_model_state_dict
                    # Or, ensure save_lora_weights and lora_state_dict are compatible
                    
                    # Convert to PEFT format if necessary (original script does this)
                    # transformer_state_dict = {f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
                    # peft_transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict) # convert_unet may not be right for Sana
                    
                    # Simpler: Assume lora_state_dict is directly loadable by PEFT
                    # The `lora_state_dict` from `SanaPipeline.lora_state_dict` should be the raw LoRA weights.
                    # `set_peft_model_state_dict` expects weights without "base_model.model." prefix.
                    # SanaPipeline.lora_state_dict likely returns it correctly.
                    
                    incompatible_keys = set_peft_model_state_dict(transformer_, lora_state_dict, adapter_name="default")
                    if incompatible_keys and incompatible_keys.unexpected_keys:
                        logger.warning(f"Loading transformer LoRA led to unexpected keys: {incompatible_keys.unexpected_keys}")
                    logger.info(f"Loaded transformer LoRA weights from {input_dir} into model instance.")
                    loaded_transformer = True
                    # models.pop(model_idx) # If accelerator expects us to handle it fully.
                except Exception as e:
                    logger.error(f"Could not load transformer LoRA weights from {input_dir}: {e}")


            elif isinstance(unwrapped_model_for_type_check, ImageConditioningProjector) and not loaded_projector:
                image_projector_ = model # Instance from accelerator.prepare()
                projector_path_st = os.path.join(input_dir, "image_projector.safetensors")
                projector_path_bin = os.path.join(input_dir, "image_projector.bin")
                loaded_path = None

                if os.path.exists(projector_path_st):
                    loaded_path = projector_path_st
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(projector_path_st, device="cpu")
                    except ImportError:
                        logger.warning("safetensors not found, trying torch.load for .safetensors file (might fail).")
                        state_dict = torch.load(projector_path_st, map_location="cpu")
                elif os.path.exists(projector_path_bin):
                    loaded_path = projector_path_bin
                    state_dict = torch.load(projector_path_bin, map_location="cpu")
                
                if loaded_path:
                    image_projector_.load_state_dict(state_dict)
                    logger.info(f"Loaded image conditioning projector weights from {loaded_path} into model instance.")
                    loaded_projector = True
                    # models.pop(model_idx)
                else:
                    logger.warning(f"Image projector weights not found at {projector_path_st} or {projector_path_bin}")
        
        # After loading, cast trainable parameters if needed (e.g., for fp16 mixed precision)
        if args.mixed_precision == "fp16":
            # Cast LoRA params of transformer and all params of projector to fp32 for training
            # The `cast_training_params` function can be used.
            # It expects a list of models whose *trainable* parameters should be cast.
            models_to_cast = []
            if transformer_ is not None: models_to_cast.append(transformer_)
            if image_projector_ is not None: models_to_cast.append(image_projector_)
            if models_to_cast:
                 cast_training_params(models_to_cast, dtype=torch.float32) # cast *trainable* params

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_projector = ( # Scale projector LR too if scaled
             args.learning_rate_projector * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )


    # Ensure LoRA parameters in transformer and all parameters of projector are in float32 if mixed precision fp16
    if args.mixed_precision == "fp16":
        # `cast_training_params` upcasts trainable params (LoRA in transformer, all in projector) to fp32
        cast_training_params([transformer, image_conditioning_projector], dtype=torch.float32)


    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    projector_parameters = list(filter(lambda p: p.requires_grad, image_conditioning_projector.parameters()))

    params_to_optimize = [
        {"params": transformer_lora_parameters, "lr": args.learning_rate},
        {"params": projector_parameters, "lr": args.learning_rate_projector}, # Use specific LR for projector
    ]

    if not projector_parameters:
        logger.warning("Image conditioning projector has no trainable parameters!")


    # Optimizer creation (AdamW or Prodigy)
    if args.optimizer.lower() == "adamw":
        optimizer_class = torch.optim.AdamW
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                raise ImportError("Install bitsandbytes: `pip install bitsandbytes` for 8-bit Adam.")
        
        optimizer = optimizer_class(
            params_to_optimize, # This now contains dicts for different param groups
            # lr will be taken from param groups, default AdamW lr is not used if all params in groups
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay, # Can also be per-group
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
            optimizer_class = prodigyopt.Prodigy
        except ImportError:
            raise ImportError("Install prodigyopt: `pip install prodigyopt`.")
        
        if args.learning_rate <= 0.1 or args.learning_rate_projector <= 0.1 : # Prodigy likes LR around 1.0
            logger.warning("Learning rate for Prodigy is low. Consider LR around 1.0.")

        optimizer = optimizer_class(
            params_to_optimize,
            # lr=args.learning_rate, # lr from param groups
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay, # Can be per-group
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    else: # Should not happen due to choices in argparser
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")


    train_dataset = Img2ImgDataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        image_pairs_dir=args.image_pairs_dir,
        caption_column=args.caption_column,
        input_image_column=args.input_image_column,
        target_image_column=args.target_image_column,
        size=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        cache_dir=args.cache_dir,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn, # Custom collate_fn
        num_workers=args.dataloader_num_workers,
    )

    # Text encoding function
    def compute_text_conditioning(prompts, text_pipeline, device):
        text_pipeline.to(device) # Ensure pipeline is on correct device
        with torch.no_grad():
            prompt_embeds_seq, prompt_attention_mask, pooled_prompt_embeds, _ = text_pipeline.encode_prompt(
                prompts,
                device=device,
                num_images_per_prompt=1, # Handled by batch size
                do_classifier_free_guidance=False, # Not doing CFG in this training script
                max_sequence_length=args.max_sequence_length,
                complex_human_instruction=args.complex_human_instruction,
            )
        if args.offload: # Offload text pipeline if specified
             text_pipeline.to("cpu")
        return prompt_embeds_seq, prompt_attention_mask, pooled_prompt_embeds
    
    # VAE scaling factor
    vae_scale_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 0.18215 # Common SD value, check Sana VAE
    if not hasattr(vae.config, 'scaling_factor'):
        logger.warning(f"VAE does not have scaling_factor in config. Using default {vae_scale_factor}")


    # Cache latents for target images if enabled
    target_latents_cache = []
    if args.cache_latents:
        vae.to(accelerator.device, dtype=vae.dtype) # VAE to device for caching
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Caching target latents", total=len(train_dataloader)):
            with torch.no_grad():
                target_pixels = batch["target_pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                cached_latents = vae.encode(target_pixels).latent_dist.sample() * vae_scale_factor
                target_latents_cache.append(cached_latents.cpu()) # Store on CPU
        if args.offload or args.validation_prompt is None: # Offload VAE if fully cached and not needed for validation soon
            vae.to("cpu")
            if torch.cuda.is_available(): torch.cuda.empty_cache()


    # Scheduler and training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare models, optimizer, dataloader, scheduler with accelerator
    transformer, image_conditioning_projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, image_conditioning_projector, optimizer, train_dataloader, lr_scheduler
    )
    # VAE and text_encoder are not prepared with accelerator if parts are offloaded or manually managed.
    # If they were always on GPU, they could be prepared.
    # For now, VAE and text_encoding_pipeline are manually moved.

    if overrode_max_train_steps: # Recalculate after accelerator prepare
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("img2img-sana-lora", config=vars(args))

    # Training loop
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, dist & accum) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting new training.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path)) # This loads optimizer, scheduler, and registered models (transformer, projector) via hooks
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32, noise_scheduler_ref=noise_scheduler_copy, device=accelerator.device):
        sigmas_ = noise_scheduler_ref.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps_ = noise_scheduler_ref.timesteps.to(device=device)
        timesteps_ = timesteps.to(device=device) # Ensure timesteps on correct device
        
        # Handle cases where timesteps might not be exactly in schedule_timesteps (e.g. due to float conversion or different scheduler)
        # Find nearest schedule timestep if exact match fails.
        # For FlowMatchEulerDiscreteScheduler, timesteps should align.
        try:
            step_indices = [(schedule_timesteps_ == t).nonzero().item() for t in timesteps_]
        except ValueError: # If a timestep is not found
            # Alternative: find closest index
            # For now, assume timesteps from scheduler.timesteps[indices] will be found
            logger.error(f"Timestep not found in schedule: {timesteps_} vs {schedule_timesteps_}")
            # Fallback or error, for now, let it raise to debug.
            # A robust way for arbitrary timesteps: interpolate sigmas.
            # But here, timesteps *are* from the scheduler's discrete steps.
            raise

        sigma = sigmas_[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Ensure VAE is on the correct device before starting epochs if not offloaded
    if not args.cache_latents and not args.offload:
        vae.to(accelerator.device, dtype=vae.dtype)


    for epoch in range(first_epoch, args.num_train_epochs):
        # Set models to train mode
        unwrap_model(transformer).train() # PEFT model's train method
        unwrap_model(image_conditioning_projector).train()

        for step, batch in enumerate(train_dataloader):
            # Models to accumulate gradients for (those with trainable params)
            models_to_accumulate = [transformer, image_conditioning_projector] 
            with accelerator.accumulate(models_to_accumulate):
                # Get text conditioning
                prompts = batch["prompts"]
                text_embeds_seq, text_attention_mask, text_pooled_embeds = compute_text_conditioning(
                    prompts, text_encoding_pipeline, accelerator.device
                )
                text_embeds_seq = text_embeds_seq.to(weight_dtype)
                text_pooled_embeds = text_pooled_embeds.to(weight_dtype)
                if text_attention_mask is not None:
                     text_attention_mask = text_attention_mask.to(accelerator.device)


                # Get target latents (from cache or encode on the fly)
                if args.cache_latents:
                    # Batch comes from dataloader, step is batch index for dataloader
                    # Cache was populated in order of dataloader.
                    target_latents = target_latents_cache[step].to(accelerator.device, dtype=weight_dtype)
                else:
                    if args.offload: vae.to(accelerator.device, dtype=vae.dtype)
                    target_pixels = batch["target_pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                    target_latents = vae.encode(target_pixels).latent_dist.sample() * vae_scale_factor
                    target_latents = target_latents.to(weight_dtype)
                    if args.offload: vae.to("cpu")
                
                # Get input image conditioning
                if args.offload: vae.to(accelerator.device, dtype=vae.dtype)
                input_pixels = batch["input_pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                condition_latents = vae.encode(input_pixels).latent_dist.sample() * vae_scale_factor
                condition_latents = condition_latents.to(weight_dtype)
                if args.offload: vae.to("cpu")

                # Project condition latents
                # Projector is already on accelerator.device and in correct dtype from prepare()
                image_pooled_embeds = unwrap_model(image_conditioning_projector)(condition_latents) # (B, pooled_dim)
                
                # Combine text and image pooled embeddings
                # Ensure dtypes match before addition
                combined_pooled_embeds = text_pooled_embeds.to(image_pooled_embeds.dtype) + image_pooled_embeds


                # Sample noise and timesteps for target latents
                noise = torch.randn_like(target_latents)
                bsz = target_latents.shape[0]

                u = compute_density_for_timestep_sampling( # For non-uniform timestep sampling if scheme != "none"
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale,
                )
                # Indices for scheduler's timesteps array
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=target_latents.device) # Ensure timesteps on correct device

                # Add noise to target latents (Flow Matching: zt = (1 - t)x + t*z1)
                sigmas = get_sigmas(timesteps, n_dim=target_latents.ndim, dtype=target_latents.dtype, device=target_latents.device)
                noisy_target_latents = (1.0 - sigmas) * target_latents + sigmas * noise

                # Predict with transformer
                # Transformer is already on accelerator.device and in correct dtype from prepare()
                model_pred = unwrap_model(transformer)(
                    hidden_states=noisy_target_latents,
                    timestep=timesteps,
                    encoder_hidden_states=text_embeds_seq,
                    encoder_attention_mask=text_attention_mask,
                    pooled_prompt_embeds=combined_pooled_embeds, # Pass combined pooled embeds
                    return_dict=False,
                )[0]

                # Compute loss
                # Weighting for loss based on sigmas (SD3 style)
                loss_weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                
                # Flow matching target: (noise - model_input) where model_input is target_latents
                flow_target = noise - target_latents

                loss = torch.mean(
                    (loss_weighting.float() * (model_pred.float() - flow_target.float()) ** 2).reshape(flow_target.shape[0], -1),
                    1, # Mean over spatial and channel dimensions
                )
                loss = loss.mean() # Mean over batch

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # Clip gradients for LoRA params in transformer and all params in projector
                    params_to_clip = itertools.chain(
                        unwrap_model(transformer).parameters(), 
                        unwrap_model(image_conditioning_projector).parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging and checkpointing
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.detach().item(), "lr_transformer": lr_scheduler.get_last_lr()[0], "lr_projector": lr_scheduler.get_last_lr()[1 if len(lr_scheduler.get_last_lr()) > 1 else 0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)


                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(f"Removing {len(removing_checkpoints)} checkpoints: {', '.join(removing_checkpoints)}")
                            for ckpt_name in removing_checkpoints:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt_name))
                    
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path) # Triggers save_model_hook
                    logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break
        
        # Validation step at end of epoch
        if accelerator.is_main_process:
            if args.validation_prompt is not None and args.validation_input_image_path is not None and epoch % args.validation_epochs == 0:
                logger.info("Running validation at end of epoch...")
                # Create a fresh base pipeline for validation to load LoRA into
                # Or, if pipeline can be reused, ensure LoRA is correctly applied to unwrapped_transformer
                
                # For validation, we need the unwrapped models with current weights
                val_transformer = unwrap_model(transformer)
                val_image_projector = unwrap_model(image_conditioning_projector)
                
                # Ensure VAE is on device for validation
                vae.to(accelerator.device, dtype=vae.dtype) # vae.dtype might be different from weight_dtype

                # Create a base Sana pipeline for inference
                # The text_encoder and tokenizer are part of text_encoding_pipeline
                # which is already configured.
                # We need a pipeline instance that uses our val_transformer.
                # SanaPipeline.from_pretrained(...) and then replace transformer is one way.
                
                # Lightweight text encoding pipeline (already exists as text_encoding_pipeline)
                # Base pipeline for generation structure (transformer will be set)
                # Use a temporary pipeline object for validation
                # Keep vae, text_encoder, tokenizer separate for more control
                
                # We need a `SanaPipeline` instance that can use the currently trained transformer.
                # The pipeline should be configured with the base model's components,
                # and then we set its transformer to our `val_transformer`.
                
                validation_pipeline = SanaPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae, # Use the same VAE instance
                    text_encoder=text_encoding_pipeline.text_encoder, # Use from prepared text pipeline
                    tokenizer=text_encoding_pipeline.tokenizer,
                    transformer=val_transformer, # Crucially, use the TRAINED transformer
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype, # For inference, match training dtype
                    cache_dir=args.cache_dir,
                )
                # The pipeline above has the LoRA layers from val_transformer.
                # No need to call load_lora_weights explicitly if val_transformer is already the LoRA-adapted one.
                
                # Ensure the validation pipeline is on the correct device
                validation_pipeline.to(accelerator.device)


                _ = log_validation( # Images are returned but not used here, logged by function
                    pipeline=validation_pipeline, # Pass the pipeline with the trained transformer
                    vae=vae, # Pass VAE separately for direct use if needed by log_validation
                    image_conditioning_projector=val_image_projector,
                    text_encoding_pipeline=text_encoding_pipeline, # Pass the prepared text pipeline
                    args=args,
                    accelerator=accelerator,
                    epoch=epoch,
                )
                
                free_memory() # Clean up after validation
                del validation_pipeline # Release temp pipeline

                # Move VAE back to CPU if offloading
                if args.offload or args.cache_latents : # If cached, VAE is not needed in loop
                     if not (args.cache_latents and step < len(train_dataloader) -1 ): # if not end of epoch with cache
                        vae.to("cpu")

        if global_step >= args.max_train_steps:
            break

    # End of training
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Unwrap models
        transformer_final = unwrap_model(transformer)
        image_projector_final = unwrap_model(image_conditioning_projector)

        if args.upcast_before_saving: # Upcast LoRA weights of transformer to fp32
            transformer_final.to(torch.float32) 
        # Projector is likely already in correct precision or float32 if cast_training_params was used

        # Get final LoRA layers for transformer
        transformer_lora_layers_final = get_peft_model_state_dict(transformer_final) # Pass wrapped if PEFT expects it
        
        # Save transformer LoRA
        SanaPipeline.save_lora_weights(
            args.output_dir,
            transformer_lora_layers=transformer_lora_layers_final,
        )
        logger.info(f"Saved final transformer LoRA weights to {args.output_dir}")

        # Save image projector weights
        projector_final_state_dict = image_projector_final.state_dict()
        projector_save_path = os.path.join(args.output_dir, "image_projector.safetensors")
        try:
            from safetensors.torch import save_file
            save_file(projector_final_state_dict, projector_save_path)
        except ImportError:
            torch.save(projector_final_state_dict, projector_save_path.replace(".safetensors", ".bin"))
        logger.info(f"Saved final image conditioning projector weights to {projector_save_path}")


        # Final validation run
        final_images = []
        if args.validation_prompt and args.validation_input_image_path and args.num_validation_images > 0:
            logger.info("Running final validation...")
            vae.to(accelerator.device, dtype=vae.dtype) # Ensure VAE on device

            # Create pipeline for final inference (similar to epoch validation)
            final_pipeline = SanaPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                text_encoder=text_encoding_pipeline.text_encoder,
                tokenizer=text_encoding_pipeline.tokenizer,
                transformer=transformer_final, # Use the final unwrapped transformer
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype, 
                cache_dir=args.cache_dir,
            )
            final_pipeline.to(accelerator.device)
            
            final_images = log_validation(
                pipeline=final_pipeline,
                vae=vae,
                image_conditioning_projector=image_projector_final,
                text_encoding_pipeline=text_encoding_pipeline,
                args=args,
                accelerator=accelerator,
                epoch=args.num_train_epochs, # Use num_train_epochs as final epoch number
                is_final_validation=True,
            )
            del final_pipeline
            free_memory()

        if args.push_to_hub:
            # Use a relevant prompt for the model card, perhaps validation_prompt
            model_card_prompt = args.validation_prompt or "An image generated with img2img"
            save_model_card(
                repo_id,
                images=final_images,
                base_model=args.pretrained_model_name_or_path,
                text_prompt=model_card_prompt,
                validation_prompt=args.validation_prompt, # Keep this if different for widget
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of img2img LoRA training",
                ignore_patterns=["step_*", "epoch_*", "*.ckpt"], # Ignore checkpoints
            )
        
        # Clean up VAE from device if it was moved
        vae.to("cpu")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args) 