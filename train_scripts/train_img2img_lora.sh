#!/bin/bash

# Activate your Python environment if needed
# source /path/to/your/venv/bin/activate

export MODEL_NAME="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
export OUTPUT_DIR="/workspace/model/img2img_v0"

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
export IMAGE_PAIRS_DIR="/workspace/data/fined_img2img_data" # Replace with your actual data directory path
export DATASET_NAME="" # Leave empty if using image_pairs_dir


# --- Training Hyperparameters ---
export RESOLUTION=512
export TRAIN_BATCH_SIZE=2 # Adjust based on your GPU memory
export GRADIENT_ACCUMULATION_STEPS=2 # Effective batch size = TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE=1
export LEARNING_RATE_PROJECTOR=1e-4 # Can be same or different from main LR
export LR_SCHEDULER="constant"
export LR_WARMUP_STEPS=0
export MAX_TRAIN_STEPS=10000 # Or NUM_TRAIN_EPOCHS
export NUM_TRAIN_EPOCHS=50 # Ignored if MAX_TRAIN_STEPS is set
export RANK=8 # LoRA rank
export LORA_LAYERS="to_q,to_v,to_k,to_out.0" # Example, adjust for SanaTransformer2DModel
export OPTIMIZER="Prodigy"
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
  --resolution=$RESOLUTION \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --learning_rate=$LEARNING_RATE \
  --learning_rate_projector=$LEARNING_RATE_PROJECTOR \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --rank=$RANK \
  --lora_layers="$LORA_LAYERS" \
  --optimizer=$OPTIMIZER \
  --mixed_precision=$MIXED_PRECISION \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=5 \
  --allow_tf32 \
  --gradient_checkpointing \
  --seed=42 \
  --dataloader_num_workers=4 \
  --max_sequence_length=256 

echo "Training finished."

# Example for running validation/inference later:
# (Assuming you have a saved model in $OUTPUT_DIR)
# python your_inference_script.py --base_model $MODEL_NAME --lora_path $OUTPUT_DIR --projector_path $OUTPUT_DIR/image_projector.safetensors ... 