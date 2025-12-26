"""
유틸리티 함수 모음
- 디바이스 설정 (GPU/MPS/CPU 자동 선택)
- 메모리 관리
- JSON to YOLO 변환
"""

import json
import torch
import gc
from pathlib import Path


def get_device():
    """사용 가능한 최적의 디바이스를 자동으로 선택"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ GPU 사용: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Apple Silicon GPU(MPS) 사용")
    else:
        device = 'cpu'
        print("⚠️ CPU 사용 (GPU 없음)")
    return device


def reset_unused_memory():
    """GPU, MPS 등의 사용하지 않는 메모리를 리셋하는 함수"""
    # CUDA 리셋
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # MPS 리셋
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass  # 일부 torch 버전에서는 empty_cache가 없을 수 있음


def clear_memory_callback(trainer):
    """
    각 epoch 종료 후 메모리 정리 콜백
    YOLO 학습 시 콜백으로 등록하여 사용
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 CUDA 메모리 정리 완료 (Epoch {trainer.epoch + 1})")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print(f"🧹 MPS 메모리 정리 완료 (Epoch {trainer.epoch + 1})")


def json_to_yolo(json_path, class_label):
    """
    JSON 라벨링 파일을 YOLO 포맷으로 변환

    Args:
        json_path: JSON 파일 경로
        class_label: 클래스 레이블 (0: OK, 1: NG)

    Returns:
        YOLO 포맷 문자열 (class_id x_center y_center width height)
        모든 값은 이미지 크기로 정규화 (0~1 범위)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 이미지 크기 가져오기
        img_width = data['description']['width']
        img_height = data['description']['height']

        # bbox 정보 가져오기
        bbox_list = data['annotations']['bbox']

        if not bbox_list:
            return None

        # 첫 번째 bbox만 사용 (딸기 하나당 하나의 bbox)
        bbox = bbox_list[0]
        x = bbox['x']
        y = bbox['y']
        w = bbox['w']
        h = bbox['h']

        # YOLO 포맷으로 변환 (중심점 기준, 정규화)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height

        # YOLO 포맷: class_id x_center y_center width height
        yolo_line = f"{class_label} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

        return yolo_line

    except Exception as e:
        print(f"❌ Error processing {json_path}: {e}")
        return None


def create_yaml_config(output_dir, yaml_path, class_names):
    """
    YOLO 학습을 위한 YAML 설정 파일 생성
    
    Args:
        output_dir: 데이터셋 디렉토리 경로
        yaml_path: 저장할 YAML 파일 경로
        class_names: 클래스 이름 리스트
    """
    yaml_content = f"""# 딸기 OK/NG 이진 분류 데이터셋
path: {output_dir.absolute()}
train: images/train
val: images/val

# 클래스 설정
nc: {len(class_names)}
names: {class_names}

# 추가 설정
save_dir: runs/detect/strawberry_ok_ng
"""
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ YAML 설정 파일 생성: {yaml_path}")


def register_memory_callback():
    """메모리 정리 콜백을 YOLO에 등록"""
    from ultralytics.utils import callbacks
    callbacks.default_callbacks['on_train_epoch_end'].append(clear_memory_callback)
    print("✅ 메모리 정리 콜백 등록 완료!")

