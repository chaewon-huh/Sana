#!/usr/bin/env python
import argparse
import torch
from diffusers import SanaPipeline, AutoencoderKL
from PIL import Image
import os
from pathlib import Path

class ImageConditioningProjector(torch.nn.Module):
    def __init__(self, vae_latent_channels: int, transformer_pooled_projection_dim: int, hidden_dim_ratio: int = 4):
        super().__init__()
        self.proj_in = vae_latent_channels
        self.proj_out = transformer_pooled_projection_dim
        intermediate_dim = vae_latent_channels * hidden_dim_ratio 
        self.fc1 = torch.nn.Linear(self.proj_in, intermediate_dim)
        self.act1 = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(intermediate_dim, self.proj_out)

    def forward(self, x):
        x = torch.mean(x, dim=[2, 3])
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Run Sana img2img inference")
    parser.add_argument
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the trained LoRA weights",
    )
    parser.add_argument(
        "--projector_path",
        type=str,
        required=True,
        help="Path to the trained image projector weights",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision type",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else torch.float32
    
    # Load base pipeline
    pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )
    
    # Load LoRA weights
    pipeline.load_lora_weights(args.lora_path)
    
    # Load image projector
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    image_projector = ImageConditioningProjector(
        vae_latent_channels=vae.config.latent_channels,
        transformer_pooled_projection_dim=text_encoder.config.hidden_size,
    )
    image_projector.load_state_dict(torch.load(args.projector_path))
    image_projector.to(device, dtype=dtype)
    
    # Move pipeline to device
    pipeline = pipeline.to(device)
    
    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None
    
    # Load and preprocess input image
    input_image = Image.open(args.input_image).convert("RGB")
    input_image = pipeline.image_processor.preprocess(input_image).to(device, dtype=dtype)
    
    # Encode input image
    with torch.no_grad():
        input_latents = vae.encode(input_image).latent_dist.sample()
        input_latents = input_latents * vae.config.scaling_factor
        image_embeds = image_projector(input_latents)
    
    # Generate image
    output = pipeline(
        prompt=args.prompt,
        image_embeds=image_embeds,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    
    # Save output image
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_image = output.images[0]
    output_path = output_dir / f"generated_{Path(args.input_image).stem}.png"
    output_image.save(output_path)
    print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    main() 