bash train_scripts/train_img2img.sh \
  configs/sana_config/512ms/Sana_600M_img512.yaml \
  --data.data_dir="[asset/example_img2img_data]" \
  --data.type=SanaImg2ImgDataset \
  --model.multi_scale=false \
  --train.train_batch_size=16 