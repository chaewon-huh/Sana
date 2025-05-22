#!/bin/bash

# Activate your Python environment if needed
# source /path/to/your/venv/bin/activate

export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers" # Or your specific Sana base model
export OUTPUT_DIR="./sana-img2img-lora-output"
export HUB_MODEL_ID="your-username/sana-img2img-lora-test" # Optional: For pushing to Hugging Face Hub

# --- Dataset Configuration ---
# Option 1: Using a local directory with image pairs and prompts
# Create a directory (e.g., ./my_img2img_data) with files like:
#   image_001_input.png
#   image_001_target.png
#   image_001_prompt.txt
#   image_002_input.jpg
#   image_002_target.jpg
#   image_002_prompt.txt
#   ...
export IMAGE_PAIRS_DIR="./my_img2img_data" # Replace with your actual data directory path
export DATASET_NAME="" # Leave empty if using image_pairs_dir

# Option 2: Using a Hugging Face Dataset
# export DATASET_NAME="your-hf-dataset-name" # Replace with your dataset on the Hub
# export IMAGE_PAIRS_DIR="" # Leave empty if using dataset_name
# export INPUT_IMAGE_COLUMN="input_image_field_name" # Column name for input images in your HF dataset
# export TARGET_IMAGE_COLUMN="target_image_field_name" # Column name for target images in your HF dataset
# export CAPTION_COLUMN="prompt_field_name" # Column name for prompts in your HF dataset

# --- Validation Configuration ---
export VALIDATION_PROMPT="A beautiful painting of a sunset over mountains, vibrant colors"
export VALIDATION_INPUT_IMAGE_PATH="./my_validation_samples/sample_input_image.png" # Replace with a path to your validation input image

# --- Training Hyperparameters ---
export RESOLUTION=512
export TRAIN_BATCH_SIZE=2 # Adjust based on your GPU memory
export GRADIENT_ACCUMULATION_STEPS=2 # Effective batch size = TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE=1e-4
export LEARNING_RATE_PROJECTOR=1e-4 # Can be same or different from main LR
export LR_SCHEDULER="constant"
export LR_WARMUP_STEPS=0
export MAX_TRAIN_STEPS=10000 # Or NUM_TRAIN_EPOCHS
export NUM_TRAIN_EPOCHS=50 # Ignored if MAX_TRAIN_STEPS is set
export RANK=8 # LoRA rank
export LORA_LAYERS="to_q,to_v,to_k,to_out.0" # Example, adjust for SanaTransformer2DModel
export OPTIMIZER="AdamW"
export MIXED_PRECISION="bf16" # or "fp16" or "no"
export CACHE_LATENTS="True" # Set to False if GPU memory is very limited or dataset is huge
export CHECKPOINTING_STEPS=1000
export VALIDATION_EPOCHS=5 # Validate more frequently initially

# --- Accelerator Configuration ---
# Use `accelerate config` to create a default config file, then adjust as needed.
# Or, pass relevant args directly to accelerate launch.
# Example for a single GPU setup:
# ACCELERATE_NUM_PROCESSES=1
# ACCELERATE_MIXED_PRECISION=$MIXED_PRECISION
# ACCELERATE_USE_CPU=False

accelerate launch Sana/train_scripts/train_img2img_lora_sana.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name="$DATASET_NAME" \
  --image_pairs_dir=$IMAGE_PAIRS_DIR \
  `# --input_image_column=$INPUT_IMAGE_COLUMN \` # Uncomment if using HF Dataset
  `# --target_image_column=$TARGET_IMAGE_COLUMN \` # Uncomment if using HF Dataset
  `# --caption_column=$CAPTION_COLUMN \` # Uncomment if using HF Dataset
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --learning_rate=$LEARNING_RATE \
  --learning_rate_projector=$LEARNING_RATE_PROJECTOR \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  `# --num_train_epochs=$NUM_TRAIN_EPOCHS \` # Comment out if using max_train_steps
  --rank=$RANK \
  --lora_layers="$LORA_LAYERS" \
  --optimizer=$OPTIMIZER \
  --mixed_precision=$MIXED_PRECISION \
  --validation_prompt="$VALIDATION_PROMPT" \
  --validation_input_image_path="$VALIDATION_INPUT_IMAGE_PATH" \
  --num_validation_images=4 \
  --validation_epochs=$VALIDATION_EPOCHS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=5 \
  --report_to="wandb" `# or "tensorboard" or "all"` \
  --allow_tf32 \
  --gradient_checkpointing \
  `# --push_to_hub --hub_model_id=$HUB_MODEL_ID \` # Uncomment to push to Hub
  `# --hub_token YOUR_HUB_TOKEN \` # Needed if push_to_hub and not logged in via CLI
  --seed=42 \
  --dataloader_num_workers=4 \
  `# --cache_latents=$CACHE_LATENTS \` # cache_latents is an action, presence enables it
  ${CACHE_LATENTS:+--cache_latents} \
  --max_sequence_length=256 `# Adjust as needed, shorter can save memory` 
  `# --offload \` # Uncomment to enable CPU offloading of VAE/Text Encoder

echo "Training finished."

# Example for running validation/inference later:
# (Assuming you have a saved model in $OUTPUT_DIR)
# python your_inference_script.py --base_model $MODEL_NAME --lora_path $OUTPUT_DIR --projector_path $OUTPUT_DIR/image_projector.safetensors ... 