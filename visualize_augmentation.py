"""
데이터 증강 시각화 도구

학습 시 적용되는 데이터 증강을 시각적으로 확인합니다.
"""

import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations, LetterBox
import matplotlib.pyplot as plt
from config import setup_matplotlib_font


def apply_yolo_augmentation(image, imgsz=640):
    """
    YOLO 증강을 직접 적용
    
    Args:
        image: 입력 이미지 (BGR)
        imgsz: 이미지 크기
    
    Returns:
        증강된 이미지
    """
    from ultralytics.data.augment import Compose, Format, RandomFlip, RandomHSV
    
    # 증강 파이프라인 구성
    transforms = [
        RandomHSV(hgain=0.03, sgain=0.8, vgain=0.5),  # HSV 증강
        RandomFlip(p=0.5, direction='horizontal'),     # 좌우 반전
        LetterBox(imgsz),                              # 리사이즈
        Format(bbox_format='xywh', normalize=True)     # 포맷 변환
    ]
    
    # 증강 적용을 위한 라벨 더미 생성
    h, w = image.shape[:2]
    labels = {
        'img': image,
        'cls': np.array([0]),  # 더미 클래스
        'instances': type('obj', (object,), {
            'bboxes': np.array([[w//2, h//2, w//4, h//4]]),  # 더미 박스
            'normalized': False
        })(),
        'ori_shape': (h, w),
        'resized_shape': (imgsz, imgsz)
    }
    
    # HSV와 Flip만 적용
    hsv_aug = RandomHSV(hgain=0.03, sgain=0.8, vgain=0.5)
    labels = hsv_aug(labels)
    
    flip_aug = RandomFlip(p=1.0, direction='horizontal')  # 확률 100%
    labels = flip_aug(labels)
    
    return labels['img']


def simple_hsv_augmentation(image, h_gain=0.03, s_gain=0.8, v_gain=0.5):
    """
    간단한 HSV 증강 적용
    
    Args:
        image: 입력 이미지 (BGR)
        h_gain: Hue gain
        s_gain: Saturation gain
        v_gain: Value gain
    
    Returns:
        증강된 이미지
    """
    # BGR to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 랜덤 gain 생성
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    
    # HSV 채널에 gain 적용
    h, s, v = cv2.split(img_hsv)
    
    dtype = image.dtype
    x = np.arange(0, 256, dtype=np.int16)
    
    # Hue
    lut_h = ((x * r[0]) % 180).astype(dtype)
    h = cv2.LUT(h.astype(dtype), lut_h)
    
    # Saturation
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    s = cv2.LUT(s.astype(dtype), lut_s)
    
    # Value
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    v = cv2.LUT(v.astype(dtype), lut_v)
    
    img_hsv = cv2.merge([h, s, v])
    
    # HSV to BGR
    augmented = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return augmented


def visualize_augmentations(image_path, output_path=None, num_augmentations=8):
    """
    원본 이미지와 증강된 이미지들을 시각화
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        num_augmentations: 생성할 증강 이미지 수
    """
    setup_matplotlib_font()
    
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 그리드 생성
    rows = (num_augmentations + 2) // 3  # 3열
    cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_augmentations > 1 else [axes]
    
    # 원본 이미지
    axes[0].imshow(image_rgb)
    axes[0].set_title('원본 이미지', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 증강된 이미지들
    for i in range(1, num_augmentations + 1):
        augmented = simple_hsv_augmentation(
            image.copy(),
            h_gain=0.03,
            s_gain=0.8,
            v_gain=0.5
        )
        
        # 랜덤 좌우 반전
        if random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
            flip_text = ' + 좌우반전'
        else:
            flip_text = ''
        
        augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        axes[i].imshow(augmented_rgb)
        axes[i].set_title(f'증강 #{i}{flip_text}', fontsize=12)
        axes[i].axis('off')
    
    # 남은 subplot 제거
    for i in range(num_augmentations + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # 저장
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 시각화 저장: {output_path}")
    
    plt.show()


def compare_augmentation_strength(image_path, output_path=None):
    """
    다양한 증강 강도를 비교 시각화
    
    Args:
        image_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
    """
    setup_matplotlib_font()
    
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 증강 설정들
    augmentation_configs = [
        {'h': 0.0, 's': 0.0, 'v': 0.0, 'name': '증강 없음'},
        {'h': 0.015, 's': 0.4, 'v': 0.2, 'name': '약한 증강'},
        {'h': 0.03, 's': 0.8, 'v': 0.5, 'name': '현재 설정 (강함)'},
        {'h': 0.05, 's': 1.0, 'v': 0.7, 'name': '매우 강한 증강'},
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, config in enumerate(augmentation_configs):
        # 원본
        axes[0, idx].imshow(image_rgb)
        axes[0, idx].set_title(f'{config["name"]}\n(원본)', fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
        
        # 증강 적용
        augmented = simple_hsv_augmentation(
            image.copy(),
            h_gain=config['h'],
            s_gain=config['s'],
            v_gain=config['v']
        )
        augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        
        axes[1, idx].imshow(augmented_rgb)
        axes[1, idx].set_title(
            f'H:{config["h"]:.3f}, S:{config["s"]:.1f}, V:{config["v"]:.1f}',
            fontsize=10
        )
        axes[1, idx].axis('off')
    
    plt.suptitle('데이터 증강 강도 비교', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 비교 시각화 저장: {output_path}")
    
    plt.show()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='데이터 증강 시각화')
    
    parser.add_argument('--image', type=str, required=True,
                        help='입력 이미지 경로')
    parser.add_argument('--output', type=str, default=None,
                        help='출력 이미지 경로')
    parser.add_argument('--num-aug', type=int, default=8,
                        help='생성할 증강 이미지 수 (기본: 8)')
    parser.add_argument('--compare', action='store_true',
                        help='증강 강도 비교 모드')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    output_path = args.output
    if output_path is None:
        if args.compare:
            output_path = f"augmentation_comparison_{image_path.stem}.jpg"
        else:
            output_path = f"augmentation_samples_{image_path.stem}.jpg"
    
    print("\n" + "=" * 70)
    print("🎨 데이터 증강 시각화")
    print("=" * 70)
    print(f"📁 입력 이미지: {image_path}")
    print(f"💾 출력 경로: {output_path}")
    print("=" * 70 + "\n")
    
    if args.compare:
        compare_augmentation_strength(image_path, output_path)
    else:
        visualize_augmentations(image_path, output_path, args.num_aug)
    
    print("\n✅ 완료!")


if __name__ == "__main__":
    main()

