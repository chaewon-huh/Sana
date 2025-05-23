# Sana Image-to-Image Dataset Format

이 디렉토리는 Sana 모델의 이미지 투 이미지 학습을 위한 데이터셋 형식 예시입니다.

## 데이터셋 구조

### 방법 1: source/target 디렉토리 구조
```
example_img2img_data/
├── source/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── target/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

- `source/` 디렉토리: 입력 이미지들 (조건부 이미지)
- `target/` 디렉토리: 목표 이미지들 (학습할 타겟 이미지)
- 파일명이 동일한 이미지들이 자동으로 페어링됩니다

### 방법 2: metadata.jsonl 형식
```
example_img2img_data/
├── images/
│   ├── source1.jpg
│   ├── target1.jpg
│   ├── source2.jpg
│   ├── target2.jpg
│   └── ...
└── metadata.jsonl
```

`metadata.jsonl` 파일 형식:
```json
{"source": "images/source1.jpg", "target": "images/target1.jpg"}
{"source": "images/source2.jpg", "target": "images/target2.jpg"}
```

## 지원 이미지 형식
- `.jpg`, `.jpeg`, `.png`, `.webp`

## 사용법

### 학습 실행
```bash
bash train_i2i.sh
```

또는 직접 실행:
```bash
bash train_scripts/train_img2img.sh \
  configs/sana_config/512ms/Sana_600M_img512.yaml \
  --data.data_dir="[asset/example_img2img_data]" \
  --data.type=SanaImg2ImgDataset \
  --model.multi_scale=false \
  --train.train_batch_size=16
```

### 설정 변경

`train_i2i.sh`에서 다음 항목들을 수정할 수 있습니다:

- `--data.data_dir`: 데이터셋 경로
- `--train.train_batch_size`: 배치 크기
- `--model.image_size`: 이미지 해상도 (512, 1024 등)

## 주의사항

1. **텍스트 프롬프트 불필요**: 이미지 투 이미지 학습에서는 텍스트 캡션이나 프롬프트가 필요하지 않습니다.
2. **이미지 페어링**: source와 target 이미지가 올바르게 페어링되어 있는지 확인하세요.
3. **해상도**: 모든 이미지는 학습 중에 설정된 해상도로 자동 리사이즈됩니다.
4. **데이터 양**: 토이 데이터셋의 경우 자동으로 반복됩니다.

## 예시 데이터 생성

실제 학습을 위해서는 이 디렉토리에 source와 target 이미지들을 추가하거나, 
metadata.jsonl 파일을 생성하여 사용하세요.

```python
# 예시: 간단한 이미지 페어 생성 스크립트
import os
from PIL import Image

# source 이미지들을 target 디렉토리로 복사 (예시)
source_dir = "asset/example_img2img_data/source"
target_dir = "asset/example_img2img_data/target"

# 실제 사용 시에는 의미 있는 이미지 변환을 수행하세요
# 예: 스타일 전환, 노이즈 제거, 해상도 향상 등 