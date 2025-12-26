"""
학습 결과 시각화 스크립트
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random
import cv2
from ultralytics import YOLO

from config import OUTPUT_DIR, setup_matplotlib_font


def visualize_training_results(results_dir, show=True, save_path=None):
    """
    학습 결과 시각화 (학습 곡선, confusion matrix 등)
    
    Args:
        results_dir: 학습 결과 디렉토리 경로
        show: 화면에 표시 여부
        save_path: 저장할 경로 (None이면 저장하지 않음)
    """
    # 한글 폰트 설정
    setup_matplotlib_font()
    
    results_dir = Path(results_dir)
    
    # 결과 이미지 파일들
    result_images = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png']
    
    # 존재하는 이미지만 필터링
    existing_images = [img for img in result_images if (results_dir / img).exists()]
    
    if not existing_images:
        print(f"⚠️ {results_dir}에서 결과 이미지를 찾을 수 없습니다.")
        return
    
    # 서브플롯 설정
    n_images = len(existing_images)
    n_cols = 2
    n_rows = (n_images + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
    
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 이미지 표시
    for idx, img_name in enumerate(existing_images):
        img_path = results_dir / img_name
        img = Image.open(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(img_name.replace('.png', '').replace('_', ' ').title(), 
                           fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    # 남은 서브플롯 숨기기
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 시각화 결과 저장: {save_path}")
    
    # 표시
    if show:
        plt.show()
    else:
        plt.close()
    
    print("✅ 학습 결과 시각화 완료!")


def visualize_sample_predictions(model_path, data_dir=None, n_samples=5, show=True, save_dir=None):
    """
    샘플 이미지에 대한 예측 결과 시각화
    
    Args:
        model_path: 모델 경로
        data_dir: 이미지 디렉토리 (None이면 validation 데이터셋 사용)
        n_samples: 샘플 개수
        show: 화면에 표시 여부
        save_dir: 저장할 디렉토리 (None이면 저장하지 않음)
    """
    # 한글 폰트 설정
    setup_matplotlib_font()
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 이미지 디렉토리 설정
    if data_dir is None:
        data_dir = OUTPUT_DIR / 'images' / 'val'
    else:
        data_dir = Path(data_dir)
    
    # 이미지 파일 수집
    image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
    
    if not image_files:
        print(f"⚠️ {data_dir}에서 이미지를 찾을 수 없습니다.")
        return
    
    # 랜덤 샘플 선택
    sample_images = random.sample(image_files, min(n_samples, len(image_files)))
    
    # 저장 디렉토리 생성
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🖼️ {len(sample_images)}개 샘플 이미지 예측 중...\n")
    
    for img_path in sample_images:
        # 예측
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        
        # 결과 시각화
        result_img = results[0].plot()
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # 표시
        plt.figure(figsize=(10, 8))
        plt.imshow(result_rgb)
        plt.title(f"예측 결과: {img_path.name}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # 저장
        if save_dir:
            save_path = save_dir / f"pred_{img_path.name}"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 저장: {save_path}")
        
        # 표시
        if show:
            plt.show()
        else:
            plt.close()
    
    print("\n✅ 샘플 예측 시각화 완료!")


def plot_class_distribution(data_dir=None):
    """
    데이터셋의 클래스 분포 시각화
    
    Args:
        data_dir: 데이터셋 디렉토리 (None이면 OUTPUT_DIR 사용)
    """
    # 한글 폰트 설정
    setup_matplotlib_font()
    
    if data_dir is None:
        data_dir = OUTPUT_DIR
    else:
        data_dir = Path(data_dir)
    
    # Train/Val 분할별 클래스 카운트
    splits = ['train', 'val']
    class_counts = {split: {'OK': 0, 'NG': 0} for split in splits}
    
    for split in splits:
        label_dir = data_dir / 'labels' / split
        if not label_dir.exists():
            continue
        
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_name = 'OK' if class_id == 0 else 'NG'
                    class_counts[split][class_name] += 1
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, split in enumerate(splits):
        counts = class_counts[split]
        classes = list(counts.keys())
        values = list(counts.values())
        
        colors = ['green', 'red']
        axes[idx].bar(classes, values, color=colors, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{split.upper()} 데이터셋 클래스 분포', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('샘플 수')
        axes[idx].grid(axis='y', alpha=0.3)
        
        # 값 표시
        for i, v in enumerate(values):
            axes[idx].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 클래스 분포:")
    for split in splits:
        counts = class_counts[split]
        total = sum(counts.values())
        print(f"\n{split.upper()}:")
        for cls, count in counts.items():
            pct = (count / total * 100) if total > 0 else 0
            print(f"  • {cls}: {count}개 ({pct:.1f}%)")


def main():
    """메인 함수 (커맨드라인 인터페이스)"""
    parser = argparse.ArgumentParser(description='딸기 OK/NG 분류 결과 시각화')
    
    subparsers = parser.add_subparsers(dest='command', help='시각화 명령')
    
    # 학습 결과 시각화
    training_parser = subparsers.add_parser('training', help='학습 결과 시각화')
    training_parser.add_argument('--results-dir', type=str, required=True,
                                help='학습 결과 디렉토리')
    training_parser.add_argument('--save', type=str, default=None,
                                help='저장할 파일 경로')
    training_parser.add_argument('--no-show', action='store_true',
                                help='화면에 표시하지 않음')
    
    # 샘플 예측 시각화
    prediction_parser = subparsers.add_parser('predictions', help='샘플 예측 시각화')
    prediction_parser.add_argument('--model', type=str, required=True,
                                  help='모델 경로')
    prediction_parser.add_argument('--data-dir', type=str, default=None,
                                  help='이미지 디렉토리')
    prediction_parser.add_argument('--n-samples', type=int, default=5,
                                  help='샘플 개수')
    prediction_parser.add_argument('--save-dir', type=str, default=None,
                                  help='저장할 디렉토리')
    prediction_parser.add_argument('--no-show', action='store_true',
                                  help='화면에 표시하지 않음')
    
    # 클래스 분포 시각화
    distribution_parser = subparsers.add_parser('distribution', help='클래스 분포 시각화')
    distribution_parser.add_argument('--data-dir', type=str, default=None,
                                    help='데이터셋 디렉토리')
    
    args = parser.parse_args()
    
    if args.command == 'training':
        visualize_training_results(
            results_dir=args.results_dir,
            show=not args.no_show,
            save_path=args.save
        )
    elif args.command == 'predictions':
        visualize_sample_predictions(
            model_path=args.model,
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            show=not args.no_show,
            save_dir=args.save_dir
        )
    elif args.command == 'distribution':
        plot_class_distribution(data_dir=args.data_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

