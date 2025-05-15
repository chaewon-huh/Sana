# python Sana/tools/convert_sana_to_diffusers.py \
#       --orig_ckpt_path model/eng_inference.pth  \
#       --model_type SanaMS_600M_P1_D28 \
#       --dtype fp16 \
#       --dump_path /workspace/model/eng_tuned_fp16_diffusers \
#       --save_full_pipeline

python Sana/tools/convert_sana_to_diffusers.py \
      --orig_ckpt_path model/eng_inference.pth \
      --model_type SanaMS_600M_P1_D28 \
      --dump_path model/sana_eng_default \
      --dtype fp16