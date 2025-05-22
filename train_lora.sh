export MODEL_NAME="Efficient-Large-Model/Sana_1600M_512px_diffusers"
export INSTANCE_DATA_DIR="/workspace/data/final_prompts_dataset"
export OUTPUT_DIR="/workspace/model/lora/sprint2_v0"

accelerate launch train_scripts/train_dreambooth_lora_sana.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DATA_DIR \
  --image_column="file_name" \
  --caption_column="text" \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --resolution=128 \
  --train_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1.0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --seed="0" \
  --resume_from_checkpoint="latest" \
  --rank=256 \
  --optimizer="prodigy" \
  --push_to_hub \
  # --validation_prompt="image of uppercase English letter 'A' rendered in AvenirNext-Medium font." \
  # --validation_epochs=1 \
  