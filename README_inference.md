# Sana Image-to-Image Inference Guide

이 문서는 Sana 이미지 투 이미지 모델의 inference 사용법을 설명합니다.

## 파일 구조

```
inference_img2img.py       # 메인 inference 스크립트
run_inference_img2img.sh   # 간편 실행 스크립트
test_img2img_batch.py      # 배치 테스트 스크립트
README_inference.md        # 사용법 가이드 (이 파일)
```

## 사전 요구사항

1. **훈련된 체크포인트**: `output/img2img_debug/checkpoints/` 디렉토리에 체크포인트 파일이 있어야 합니다
2. **conda 환경**: `conda activate sana` 실행
3. **소스 이미지**: 변환하고 싶은 입력 이미지

## 1. 단일 이미지 inference

### 방법 1: Python 스크립트 직접 실행

```bash
python inference_img2img.py \
    --config configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml \
    --checkpoint output/img2img_debug/checkpoints/latest.pth \
    --source data/FinalInputData256/example_256.png \
    --output output/inference_results/example_generated.png \
    --steps 20 \
    --sampler flow_euler
```

### 방법 2: 간편 스크립트 사용

```bash
# 기본 설정으로 실행
./run_inference_img2img.sh --source data/FinalInputData256/example_256.png

# 고급 설정으로 실행
./run_inference_img2img.sh \
    --source data/FinalInputData256/example_256.png \
    --steps 28 \
    --sampler dpm-solver \
    --output output/my_results
```

## 2. 배치 테스트

여러 이미지를 한 번에 테스트하고 비교 그리드를 생성:

```bash
python test_img2img_batch.py \
    --config configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml \
    --checkpoint output/img2img_debug/checkpoints/latest.pth \
    --source_dir data/FinalInputData256 \
    --target_dir data/FinalTargetData256 \
    --output_dir output/batch_test_results \
    --num_images 10 \
    --create_grids \
    --steps 20
```

## 3. 파라미터 설명

### 공통 파라미터

- `--config`: 모델 설정 파일 경로
- `--checkpoint`: 훈련된 모델 체크포인트 경로
- `--steps`: 샘플링 스텝 수 (기본값: 20, 더 높으면 품질 향상)
- `--cfg_scale`: CFG 스케일 (기본값: 1.0)
- `--sampler`: 샘플러 종류
  - `flow_euler`: 빠른 생성 (기본값)
  - `dpm-solver`: 더 높은 품질
- `--device`: 사용할 디바이스 (기본값: cuda)

### 단일 inference 파라미터

- `--source`: 입력 이미지 경로
- `--output`: 출력 이미지 경로

### 배치 테스트 파라미터

- `--source_dir`: 소스 이미지 디렉토리
- `--target_dir`: 타겟 이미지 디렉토리 (선택사항, 비교용)
- `--output_dir`: 출력 디렉토리
- `--num_images`: 테스트할 이미지 개수
- `--create_grids`: 비교 그리드 생성 여부

## 4. 샘플링 설정 가이드

### 빠른 생성 (5-10초)
```bash
--steps 14 --sampler flow_euler
```

### 균형잡힌 품질 (10-20초)
```bash
--steps 20 --sampler flow_euler
```

### 높은 품질 (20-30초)
```bash
--steps 28 --sampler dpm-solver
```

## 5. 결과 확인

생성된 이미지는 다음 위치에 저장됩니다:
- 단일 inference: 지정한 `--output` 경로
- 배치 테스트: `output_dir/generated/` 디렉토리
- 비교 그리드: `output_dir/comparison_grids/` 디렉토리

## 6. 문제 해결

### 일반적인 오류

1. **체크포인트 없음**: 
   - 훈련이 완료되었는지 확인
   - `output/img2img_debug/checkpoints/` 디렉토리 확인

2. **CUDA 메모리 부족**:
   - `--steps` 수를 줄이기 (예: 14)
   - 이미지 크기 확인

3. **이미지 로딩 오류**:
   - 이미지 파일 형식 확인 (.png, .jpg 지원)
   - 파일 경로 확인

### 성능 최적화

- **더 빠른 생성**: `--steps 14 --sampler flow_euler`
- **GPU 메모리 절약**: 배치 크기를 1로 유지
- **더 좋은 품질**: `--steps 28 --sampler dpm-solver`

## 7. 예시 실행

기본 데이터셋으로 테스트:

```bash
# conda 환경 활성화
conda activate sana

# 단일 이미지 테스트
./run_inference_img2img.sh --source data/FinalInputData256/$(ls data/FinalInputData256/ | head -1)

# 5개 이미지 배치 테스트
python test_img2img_batch.py \
    --config configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml \
    --checkpoint output/img2img_debug/checkpoints/latest.pth \
    --source_dir data/FinalInputData256 \
    --target_dir data/FinalTargetData256 \
    --num_images 5 \
    --create_grids
```

성공적으로 실행되면 고품질의 이미지 투 이미지 변환 결과를 얻을 수 있습니다! 