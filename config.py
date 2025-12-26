"""
딸기 OK/NG 이진 분류 YOLO 학습 파이프라인 - 설정 파일

데이터 구조:
- OK (class 0): 정상 딸기
- NG (class 1): 병해, 생리장해, 작물보호제처리반응
"""

import os
from pathlib import Path
import platform

# 환경 설정
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# 경로 설정
BASE_DIR = Path("/Users/temp/내 드라이브(codejeteho123@gmail.com)/ComputerVision")
# BASE_DIR = Path("/content/drive/MyDrive/ComputerVision")
JSON_DIR = BASE_DIR / "딸기 라벨링 JSON"
IMAGE_DIR = BASE_DIR / "딸기이미지"
OUTPUT_DIR = BASE_DIR / "딸기_binary_dataset"

# 클래스 매핑: 정상 -> OK(0), 병해/생리장해/작물보호제처리반응 -> NG(1)
CLASS_MAPPING = {
    "정상": 0,      # OK
    "병해": 1,      # NG
    "생리장해": 1,  # NG
    "작물보호제처리반응": 1  # NG
}

CLASS_NAMES = ["OK", "NG"]

# 데이터 분할 설정
VAL_RATIO = 0.2  # 80% train, 20% val
RANDOM_SEED = 42

# 학습 설정
TRAINING_CONFIG = {
    'model': 'yolo11s.pt',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
    'patience': 20,
    'workers': 4,
    'name': 'strawberry_ok_ng',
    'project': 'runs/detect',
    
    # 최적화 설정
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    
    # 데이터 증강
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0
}

# 예측 설정
PREDICTION_CONFIG = {
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
}

# Matplotlib 한글 폰트 설정
def setup_matplotlib_font():
    """플랫폼에 따라 한글 폰트 설정"""
    import matplotlib.pyplot as plt
    
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:  # Linux (Colab)
        plt.rcParams['font.family'] = 'NanumGothic'
    
    plt.rcParams['axes.unicode_minus'] = False

