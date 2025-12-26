# 🍓 딸기 OK/NG 분류 시스템

YOLO 기반 딸기 품질 검사 자동화 시스템입니다. SAHI(Slicing Aided Hyper Inference)를 활용하여 대용량 이미지에서 정확한 딸기 검출 및 품질 분류를 수행합니다.

## 📋 목차

- [프로젝트 구조](#프로젝트-구조)
- [설치](#설치)
- [사용법](#사용법)
  - [1. 전체 파이프라인 실행](#1-전체-파이프라인-실행)
  - [2. 단계별 실행](#2-단계별-실행)
  - [3. SAHI 기반 추론](#3-sahi-기반-추론)
  - [4. 배치 테스트](#4-배치-테스트)
- [성능 평가](#성능-평가)
- [주요 설정](#주요-설정)
- [트러블슈팅](#트러블슈팅)

## 📁 프로젝트 구조

```
ComputerVision/
├── 📋 핵심 스크립트
│   ├── config.py                    # 전역 설정 (경로, 클래스 등)
│   ├── utils.py                     # 유틸리티 함수
│   ├── dataset_preparation.py       # 데이터셋 준비 및 변환
│   ├── train.py                     # YOLO 모델 학습
│   ├── evaluate.py                  # 모델 성능 평가
│   ├── predict.py                   # 단일 이미지 예측
│   ├── visualize.py                 # 학습 결과 시각화
│   └── main.py                      # 전체 파이프라인 실행
│
├── 🔍 추론 및 분석
│   ├── test_sahi.py                 # SAHI 기반 추론 (권장)
│   ├── batch_test_images.py         # 배치 이미지 테스트
│   ├── diagnose_with_sahi.py        # 오분류 진단 도구
│   ├── compare_models.py            # PyTorch vs ONNX 비교
│   └── visualize_augmentation.py    # 데이터 증강 시각화
│
├── 📊 데이터
│   ├── 딸기이미지/                  # 원본 이미지 (OK/NG)
│   ├── 딸기_binary_dataset/         # 학습용 YOLO 포맷 데이터
│   └── strawberry_ok_ng.yaml       # 데이터셋 설정
│
├── 🤖 모델
│   └── runs/detect/strawberry_ok_ng/weights/
│       ├── best.pt                  # 최고 성능 PyTorch 모델
│       └── best.onnx               # ONNX 변환 모델
│
└── 📄 기타
    ├── README.md                    # 이 파일
    ├── requirements.txt             # 필요 패키지
    └── train_on_colab.ipynb         # Google Colab 학습 노트북

```

## 🚀 설치

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 주요 패키지
- `ultralytics` (YOLO11)
- `sahi` (대용량 이미지 추론)
- `opencv-python`
- `numpy`
- `matplotlib`
- `tqdm`
- `onnxruntime`

## 💻 사용법

### 🌟 Google Colab에서 학습 (권장)

GPU가 없거나 빠른 학습을 원한다면 Google Colab을 사용하세요!

1. **노트북 열기**: `train_on_colab.ipynb` 파일을 Google Colab에 업로드
   - 또는 GitHub에서 직접 열기: [Open in Colab](https://colab.research.google.com/github/adjective123/ComputerVision/blob/main/train_on_colab.ipynb)

2. **런타임 설정**: 런타임 > 런타임 유형 변경 > **T4 GPU** 선택

3. **순서대로 실행**: 모든 셀을 순서대로 실행하면 자동으로:
   - Google Drive 마운트
   - 필요한 패키지 설치
   - 프로젝트 코드 클론
   - **데이터셋 자동 생성** (JSON → YOLO 포맷 변환)
   - 모델 학습 (100 에포크, 약 1-2시간)
   - 학습 결과 확인
   - ONNX 모델 내보내기

**장점:**
- ✅ 무료 GPU 사용 (T4 GPU)
- ✅ 빠른 학습 속도 (로컬 CPU 대비 10배 이상)
- ✅ 데이터셋 자동 생성
- ✅ Google Drive에 자동 저장

---

### 1. 전체 파이프라인 실행 (로컬)

데이터 준비부터 학습, 평가까지 한 번에 실행:

```bash
python main.py
```

### 2. 단계별 실행

#### Step 1: 데이터셋 준비

```bash
python dataset_preparation.py
```

JSON 라벨을 YOLO 포맷으로 변환하고 train/val로 분할합니다.

#### Step 2: 모델 학습

```bash
python train.py --epochs 100 --batch 16 --imgsz 640
```

**주요 옵션:**
- `--epochs`: 학습 에폭 수 (기본: 100)
- `--batch`: 배치 크기 (기본: 16)
- `--imgsz`: 이미지 크기 (기본: 640)
- `--device`: 디바이스 (자동 선택)

**🎨 데이터 증강 (Data Augmentation):**
모델은 다양한 색상의 딸기를 학습할 수 있도록 강력한 데이터 증강이 적용되어 있습니다:

- **색상 증강 (HSV)**:
  - Hue (색상): ±3% - 빨강/주황/노랑 딸기 대응
  - Saturation (채도): ±80% - 채도 변화에 강건
  - Value (명도): ±50% - 조명 변화에 강건

- **기하학적 증강**:
  - 회전: ±15도
  - 이동: ±15%
  - 크기 조절: ±60%
  - 좌우 반전: 50%

- **고급 증강**:
  - Mosaic (4개 이미지 결합)
  - MixUp (이미지 혼합, 15%)
  - Copy-Paste (객체 복사, 10%)

이 설정은 `config.py`의 `TRAINING_CONFIG`에서 수정할 수 있습니다.

#### Step 3: 모델 평가

```bash
python evaluate.py --model runs/detect/strawberry_ok_ng/weights/best.pt --data strawberry_ok_ng.yaml
```

#### Step 4: 예측

```bash
python predict.py --model runs/detect/strawberry_ok_ng/weights/best.pt --source sample.jpg
```

#### Step 5: 결과 시각화

```bash
python visualize.py --results runs/detect/strawberry_ok_ng
```

### 3. SAHI 기반 추론 (권장) ⭐

대용량 이미지에서 높은 정확도로 딸기를 검출합니다:

```bash
# 기본 사용
python test_sahi.py --image sample.jpg

# 설정 커스터마이징
python test_sahi.py \
  --image sample.jpg \
  --conf 0.85 \
  --slice-size 640 \
  --overlap 0.3 \
  --output result.jpg
```

**주요 파라미터:**
- `--conf`: Confidence threshold (기본: 0.85)
  - 높을수록 정확하지만 검출 개수 감소
  - 권장: 0.85 (OK 100%, NG 85% 정확도)
- `--slice-size`: 슬라이스 크기 (기본: 640)
- `--overlap`: 오버랩 비율 (기본: 0.3)
- `--output`: 결과 저장 경로

**생성되는 파일:**
- `*_custom.jpg`: 검출 결과 시각화 (OK=초록, NG=빨강)
- `*_slices.jpg`: 슬라이스 영역 시각화
- `*.jpg`: SAHI 기본 시각화

### 4. 배치 테스트

여러 이미지를 한 번에 테스트하고 정확도를 측정:

```bash
# 20개 랜덤 샘플 테스트
python batch_test_images.py --num-samples 20 --conf 0.85

# 전체 이미지 테스트
python batch_test_images.py --conf 0.85

# 커스텀 경로
python batch_test_images.py \
  --ok-dir "딸기이미지/정상" \
  --ng-dir "딸기이미지/NG" \
  --num-samples 50 \
  --conf 0.85
```

**결과 저장:**
- JSON: `batch_test_results_[timestamp].json`
- 이미지: `batch_test_results/` 폴더
  - `ok_correct/`: OK를 정확히 분류한 이미지
  - `ok_wrong/`: OK를 잘못 분류한 이미지
  - `ng_correct/`: NG를 정확히 분류한 이미지
  - `ng_wrong/`: NG를 잘못 분류한 이미지

## 📊 성능 평가

### 검증 데이터셋 성능
- **mAP50**: 99.5%
- **mAP50-95**: 92.5%
- **Precision**: 99.8%
- **Recall**: 100%

### 실제 이미지 테스트 (40개 랜덤 샘플, Conf=0.85)
- **전체 정확도**: 92.5%
- **OK 정확도**: 100% (20/20)
- **NG 정확도**: 85% (17/20)

### Confusion Matrix
```
              예측 OK    예측 NG
실제 OK:         20          0
실제 NG:          3         17
```

## ⚙️ 주요 설정

### Confidence Threshold 선택 가이드

| Threshold | OK 정확도 | NG 정확도 | 용도 |
|-----------|-----------|-----------|------|
| 0.85 | 100% | 85% | **권장** - 정확도 우선 |
| 0.75 | 100% | ~90% | 균형잡힌 선택 |
| 0.5 | ~95% | ~92% | 검출률 우선 |

### SAHI 설정

**고해상도 이미지 (3000x4000+):**
```bash
python test_sahi.py --slice-size 640 --overlap 0.3 --conf 0.85
```

**중간 해상도 (1920x1080):**
```bash
python test_sahi.py --slice-size 640 --overlap 0.3 --conf 0.75
```

## 🛠️ 트러블슈팅

### 1. FileNotFoundError: YAML 파일 경로 오류

**문제**: 환경이 바뀌면 YAML 파일의 절대 경로가 맞지 않을 수 있습니다.

**해결**:
```bash
# evaluate.py와 predict.py 실행 시 --data 인자 명시
python evaluate.py --model best.pt --data strawberry_ok_ng.yaml
python predict.py --model best.pt --source image.jpg --data strawberry_ok_ng.yaml
```

또는 `strawberry_ok_ng.yaml` 파일의 경로를 현재 환경에 맞게 수정:
```yaml
path: /your/current/path/ComputerVision/딸기_binary_dataset
train: images/train
val: images/val
```

### 2. 낮은 검출률

**원인**: Confidence threshold가 너무 높거나 이미지 해상도가 낮음

**해결**:
1. Confidence를 낮추기: `--conf 0.7` 또는 `--conf 0.5`
2. 원본 고해상도 이미지 사용 (리사이즈 금지)
3. Overlap 증가: `--overlap 0.5`

### 3. NG 오분류 (NG를 OK로 잘못 판단)

**원인**: 모델이 일부 NG 특징을 학습하지 못함

**해결**:
1. Confidence를 낮추기: `--conf 0.75`
2. 오분류된 이미지 확인:
   ```bash
   # 진단 실행
   python diagnose_with_sahi.py --image sample.jpg --conf 0.7
   
   # 배치 테스트로 오분류 패턴 확인
   python batch_test_images.py --num-samples 20 --conf 0.75
   ```
3. 학습 데이터에 유사한 NG 샘플 추가 후 재학습

### 4. PyTorch vs ONNX 성능 비교

```bash
python compare_models.py \
  --image sample.jpg \
  --pt-model runs/detect/strawberry_ok_ng/weights/best.pt \
  --onnx-model runs/detect/strawberry_ok_ng/weights/best.onnx
```

## 🎯 사용 예시

### 예시 1: 단일 이미지 검출

```bash
python test_sahi.py --image sample.jpg --conf 0.85
```

**결과**: `final_sample_result_custom.jpg` 생성

### 예시 2: 품질 검사 배치 처리

```bash
python batch_test_images.py \
  --ok-dir "딸기이미지/정상" \
  --ng-dir "딸기이미지/NG" \
  --num-samples 50 \
  --conf 0.85
```

**결과**: 
- JSON 보고서
- 각 카테고리별 결과 이미지 저장

### 예시 3: 모델 성능 검증

```bash
# 1. 검증 데이터로 평가
python evaluate.py --model runs/detect/strawberry_ok_ng/weights/best.pt --data strawberry_ok_ng.yaml

# 2. 실제 이미지로 진단
python diagnose_with_sahi.py --image sample.jpg --conf 0.85

# 3. 배치 테스트
python batch_test_images.py --num-samples 20 --conf 0.85
```

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. `requirements.txt`의 모든 패키지가 설치되었는지
2. Python 버전이 3.8 이상인지
3. YAML 파일의 경로가 올바른지
4. 충분한 메모리가 있는지 (대용량 이미지 처리 시)

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.

---

**최종 업데이트**: 2024년 12월 23일
**버전**: 2.0 (SAHI 통합)
