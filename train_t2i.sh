bash train_scripts/train.sh \
  configs/sana_config/512ms/Sana_600M_img512.yaml \
  --data.data_dir="[asset/example_data]" \
  --data.type=SanaImgDataset \
  --model.multi_scale=false \
  --train.train_batch_size=32