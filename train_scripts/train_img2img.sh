#!/bin/bash
set -e

work_dir=output/img2img_debug
np=1


if [[ $1 == *.yaml ]]; then
    config=$1
    shift
else
    config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml"
    # config="configs/sana1-5_config/1024ms/Sana_1600M_1024px_AdamW_fsdp.yaml"      FSDP config file
    echo "Only support .yaml files, but get $1. Set to --config_path=$config"
fi

TRITON_PRINT_AUTOTUNING=1 \
    torchrun --nproc_per_node=$np --master_port=15433 \
        train_scripts/train_img2img.py \
        --config_path=$config \
        --work_dir=$work_dir \
        --name=img2img_tmp \
        --debug=true \
        --report_to=tensorboard \
        --train.visualize=false \
        --train.save_model_steps=5 \
        --train.num_epochs=1 \
        "$@" 