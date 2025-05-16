#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit=12.4 -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# init a raw torch to avoid installation errors.
# pip install torch

# update pip to latest version for pyproject.toml setup.
pip install -U pip

# for fast attn
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install prodigyopt

# install sana
pip install -e .

# install torchprofile
# pip install git+https://github.com/zhijian-liu/torchprofile


# 4. Hugging Face CLI 로그인
huggingface-cli login --token "$(grep '^hf_token=' .env | cut -d '=' -f2)" 

# 5. Git 사용자 정보 설정
git config --global user.name "chaewon.huh"
git config --global user.email "cw.huh@postech.ac.kr" 

# 6. 추가 패키지 설치 (unzip)
apt-get update
apt-get install -y unzip 