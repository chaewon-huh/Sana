#!/usr/bin/env python3
"""
Batch testing script for Sana Image-to-Image model
Tests multiple images from the dataset and compares results
"""

import os
import glob
import argparse
from pathlib import Path
import time
from PIL import Image

# Import the inference class
from inference_img2img import Img2ImgInference


def create_comparison_grid(source_path, generated_path, target_path=None, output_path=None):
    """Create a comparison grid with source, generated, and optionally target images"""
    source_img = Image.open(source_path)
    generated_img = Image.open(generated_path)
    
    if target_path and os.path.exists(target_path):
        target_img = Image.open(target_path)
        # Create 3-image grid: source | generated | target
        width, height = source_img.size
        grid = Image.new('RGB', (width * 3, height))
        grid.paste(source_img, (0, 0))
        grid.paste(generated_img, (width, 0))
        grid.paste(target_img, (width * 2, 0))
        
        if output_path:
            grid.save(output_path)
            print(f"Comparison grid saved to: {output_path}")
        return grid
    else:
        # Create 2-image grid: source | generated
        width, height = source_img.size
        grid = Image.new('RGB', (width * 2, height))
        grid.paste(source_img, (0, 0))
        grid.paste(generated_img, (width, 0))
        
        if output_path:
            grid.save(output_path)
            print(f"Comparison grid saved to: {output_path}")
        return grid


def main():
    parser = argparse.ArgumentParser(description="Batch test Sana Image-to-Image model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing source images")
    parser.add_argument("--target_dir", type=str, default=None, help="Directory containing target images (optional)")
    parser.add_argument("--output_dir", type=str, default="output/batch_test_results", help="Output directory")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to test")
    parser.add_argument("--steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--sampler", type=str, default="flow_euler", choices=["flow_euler", "dpm-solver"], help="Sampler type")
    parser.add_argument("--create_grids", action="store_true", help="Create comparison grids")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    generated_dir = os.path.join(args.output_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)
    
    if args.create_grids:
        grid_dir = os.path.join(args.output_dir, "comparison_grids")
        os.makedirs(grid_dir, exist_ok=True)
    
    # Get list of source images
    source_pattern = os.path.join(args.source_dir, "*.png")
    source_images = glob.glob(source_pattern)
    source_images.extend(glob.glob(os.path.join(args.source_dir, "*.jpg")))
    source_images.extend(glob.glob(os.path.join(args.source_dir, "*.jpeg")))
    
    if len(source_images) == 0:
        print(f"No images found in {args.source_dir}")
        return
    
    # Limit number of images to test
    source_images = source_images[:args.num_images]
    
    print(f"Found {len(source_images)} images to test")
    print(f"Testing with {args.steps} steps using {args.sampler} sampler")
    print("="*50)
    
    # Initialize inference pipeline
    print("Initializing inference pipeline...")
    pipeline = Img2ImgInference(args.config, args.checkpoint, args.device)
    
    total_time = 0
    successful_generations = 0
    
    for i, source_path in enumerate(source_images):
        try:
            print(f"\nProcessing {i+1}/{len(source_images)}: {os.path.basename(source_path)}")
            
            # Generate output path
            source_name = Path(source_path).stem
            output_path = os.path.join(generated_dir, f"{source_name}_generated.png")
            
            # Run inference
            start_time = time.time()
            pipeline.generate(
                source_image_path=source_path,
                output_path=output_path,
                num_steps=args.steps,
                cfg_scale=args.cfg_scale,
                sampler=args.sampler
            )
            inference_time = time.time() - start_time
            total_time += inference_time
            successful_generations += 1
            
            print(f"Generation time: {inference_time:.2f}s")
            
            # Create comparison grid if requested
            if args.create_grids:
                grid_path = os.path.join(grid_dir, f"{source_name}_comparison.png")
                
                # Look for corresponding target image
                target_path = None
                if args.target_dir:
                    # Try different naming patterns
                    possible_targets = [
                        os.path.join(args.target_dir, f"{source_name}_refined_256.png"),
                        os.path.join(args.target_dir, f"{source_name}.png"),
                        os.path.join(args.target_dir, f"{source_name}.jpg"),
                    ]
                    for target in possible_targets:
                        if os.path.exists(target):
                            target_path = target
                            break
                
                create_comparison_grid(source_path, output_path, target_path, grid_path)
            
        except Exception as e:
            print(f"Error processing {source_path}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH TEST SUMMARY")
    print("="*50)
    print(f"Images processed: {successful_generations}/{len(source_images)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/successful_generations:.2f}s" if successful_generations > 0 else "N/A")
    print(f"Generated images saved to: {generated_dir}")
    
    if args.create_grids:
        print(f"Comparison grids saved to: {grid_dir}")
    
    print(f"\nResults saved in: {args.output_dir}")


if __name__ == "__main__":
    main() 