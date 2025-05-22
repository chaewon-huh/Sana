#!/bin/bash

# Base model and weights paths
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export LORA_PATH="./sana-img2img-lora-output"  # Path to your trained LoRA weights
export PROJECTOR_PATH="./sana-img2img-lora-output/image_projector.safetensors"  # Path to your trained projector weights

# Input/Output settings
export INPUT_IMAGE="./input_images/your_input_image.png"  # Replace with your input image path
export PROMPT="A beautiful painting of a sunset over mountains, vibrant colors"  # Replace with your prompt
export OUTPUT_DIR="./generated_images"

# Generation settings
export NUM_INFERENCE_STEPS=50
export GUIDANCE_SCALE=7.5
export SEED=42  # Optional: Set to None for random seed
export MIXED_PRECISION="bf16"  # or "fp16" or "no"

# Run inference
python Sana/run.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --lora_path=$LORA_PATH \
    --projector_path=$PROJECTOR_PATH \
    --input_image=$INPUT_IMAGE \
    --prompt="$PROMPT" \
    --output_dir=$OUTPUT_DIR \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --guidance_scale=$GUIDANCE_SCALE \
    ${SEED:+--seed=$SEED} \
    --mixed_precision=$MIXED_PRECISION

echo "Inference completed. Check $OUTPUT_DIR for generated images." 