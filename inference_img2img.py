#!/usr/bin/env python3
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import yaml

# Import necessary modules
from diffusion.model.builder import build_model, get_vae, vae_encode, vae_decode
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.model.nets.sana_blocks import ImageConditionEmbedder
from diffusion import DPMS, FlowEuler
from diffusion.utils.checkpoint import load_checkpoint


class Img2ImgInference:
    def __init__(self, config_path, checkpoint_path, device="cuda"):
        self.device = device
        self.config = self._load_config(config_path)
        self.model = None
        self.vae = None
        self._setup_model(checkpoint_path)
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert dict to SanaConfig object
        config = SanaConfig()
        for key, value in config_dict.items():
            setattr(config, key, value)
            
        return config
        
    def _setup_model(self, checkpoint_path):
        """Setup model and load checkpoint"""
        print("Setting up model...")
        
        # Setup basic parameters
        image_size = self.config.model.image_size
        latent_size = image_size // self.config.vae.vae_downsample_rate
        
        # Build model
        model_kwargs = model_init_config(self.config, latent_size=latent_size)
        
        # Create null embedding path
        null_embed_path = "output/pretrained_models/null_embed_img2img.pth"
        os.makedirs("output/pretrained_models", exist_ok=True)
        
        # Create dummy null embedding if it doesn't exist
        if not os.path.exists(null_embed_path):
            null_embed_data = {'uncond_prompt_embeds': torch.zeros(1, 2304)}
            torch.save(null_embed_data, null_embed_path)
            print(f"Created dummy null embedding at {null_embed_path}")
        
        # Build model
        self.model = build_model(
            self.config.model.model,
            False,  # grad_checkpointing
            getattr(self.config.model, "fp32_attention", False),
            null_embed_path=null_embed_path,
            **model_kwargs,
        ).eval()
        
        # Replace y_embedder with ImageConditionEmbedder
        print("Replacing y_embedder with ImageConditionEmbedder...")
        vae_latent_channels = 32
        
        img_embedder = ImageConditionEmbedder(
            in_channels=vae_latent_channels,
            hidden_size=self.model.hidden_size,
            uncond_prob=0.0,  # No CFG for inference
            token_num=getattr(self.model.y_embedder, 'token_num', 300)
        )
        
        self.model.y_embedder = img_embedder
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove y_embedder keys from checkpoint (they're incompatible)
        keys_to_remove = [key for key in state_dict.keys() if key.startswith("y_embedder")]
        for key in keys_to_remove:
            del state_dict[key]
            print(f"Removed incompatible key: {key}")
            
        # Load state dict
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup VAE
        self.vae = get_vae(
            self.config.vae.vae_type, 
            self.config.vae.vae_pretrained, 
            self.device
        ).to(torch.float32)
        
        print("Model setup complete!")
        
    def preprocess_image(self, image_path, target_size):
        """Preprocess input image"""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
        
        return image_tensor
        
    def generate(self, source_image_path, output_path, num_steps=20, cfg_scale=1.0, sampler="flow_euler"):
        """Generate target image from source image"""
        print(f"Generating image from {source_image_path}")
        
        image_size = self.config.model.image_size
        latent_size = image_size // self.config.vae.vae_downsample_rate
        
        # Preprocess source image
        source_tensor = self.preprocess_image(source_image_path, image_size)
        
        # Encode source image to latent space
        with torch.no_grad():
            source_latent = vae_encode(
                self.config.vae.vae_type, 
                self.vae, 
                source_tensor, 
                False, 
                self.device
            )
            
        # Create initial noise
        noise = torch.randn(1, self.config.vae.vae_latent_dim, latent_size, latent_size, device=self.device)
        
        # Setup model kwargs
        hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=self.device)
        ar = torch.tensor([[1.0]], device=self.device)
        data_info = {"img_hw": hw, "aspect_ratio": ar}
        
        # Create null condition for unconditional generation
        null_condition = torch.zeros_like(source_latent)
        
        model_kwargs = dict(
            data_info=data_info,
            img_condition=source_latent
        )
        
        # Generate image
        with torch.no_grad():
            if sampler == "flow_euler":
                flow_solver = FlowEuler(
                    self.model, 
                    condition=source_latent,
                    uncondition=null_condition,
                    cfg_scale=cfg_scale,
                    model_kwargs=model_kwargs
                )
                generated_latent = flow_solver.sample(noise, steps=num_steps)
                
            elif sampler == "dpm-solver":
                dpm_solver = DPMS(
                    self.model.forward_with_dpmsolver,
                    condition=source_latent,
                    uncondition=null_condition,
                    cfg_scale=cfg_scale,
                    model_kwargs=model_kwargs,
                )
                generated_latent = dpm_solver.sample(
                    noise,
                    steps=num_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            else:
                raise ValueError(f"Unsupported sampler: {sampler}")
        
        # Decode latent to image
        with torch.no_grad():
            generated_image = vae_decode(self.config.vae.vae_type, self.vae, generated_latent)
            
        # Convert to PIL Image
        generated_image = torch.clamp(127.5 * generated_image + 128.0, 0, 255)
        generated_image = generated_image.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
        result_image = Image.fromarray(generated_image)
        
        # Save result
        result_image.save(output_path)
        print(f"Generated image saved to {output_path}")
        
        return result_image


def main():
    parser = argparse.ArgumentParser(description="Sana Image-to-Image Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--source", type=str, required=True, help="Path to source image")
    parser.add_argument("--output", type=str, required=True, help="Path to output image")
    parser.add_argument("--steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--sampler", type=str, default="flow_euler", choices=["flow_euler", "dpm-solver"], help="Sampler type")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize inference pipeline
    pipeline = Img2ImgInference(args.config, args.checkpoint, args.device)
    
    # Generate image
    pipeline.generate(
        source_image_path=args.source,
        output_path=args.output,
        num_steps=args.steps,
        cfg_scale=args.cfg_scale,
        sampler=args.sampler
    )


if __name__ == "__main__":
    main() 