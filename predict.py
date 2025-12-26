"""
예측 및 추론 스크립트
"""

import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from config import CLASS_NAMES, PREDICTION_CONFIG, setup_matplotlib_font


def quick_predict(image_path, model_path, conf_threshold=None, visualize=True, save_result=False):
    """
    빠른 예측 함수

    Args:
        image_path: 예측할 이미지 경로
        model_path: 학습된 모델 경로
        conf_threshold: 신뢰도 임계값 (None이면 기본값 사용)
        visualize: 결과 시각화 여부
        save_result: 결과 저장 여부

    Returns:
        predictions: 예측 결과 리스트
    """
    # 한글 폰트 설정
    setup_matplotlib_font()
    
    # 신뢰도 임계값 설정
    if conf_threshold is None:
        conf_threshold = PREDICTION_CONFIG['conf_threshold']
    
    # 모델 로드
    model = YOLO(model_path)

    # 예측
    results = model.predict(image_path, conf=conf_threshold, verbose=False)

    # 이미지 읽기
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictions = []

    # 예측 결과 처리
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            predictions.append({
                'class_id': cls,
                'class_name': CLASS_NAMES[cls],
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })

            # 시각화를 위한 바운딩 박스 그리기
            if visualize or save_result:
                color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                label = f"{CLASS_NAMES[cls]} {conf:.2f}"

                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_rgb, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # 결과 시각화
    if visualize:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f"예측 결과: {Path(image_path).name}", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # 결과 저장
    if save_result:
        output_path = Path(image_path).parent / f"predicted_{Path(image_path).name}"
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_bgr)
        print(f"💾 결과 저장: {output_path}")

    # 예측 정보 출력
    print("=" * 70)
    print("🔍 예측 결과")
    print("=" * 70)
    if predictions:
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. 클래스: {pred['class_name']} | 신뢰도: {pred['confidence']:.4f}")
    else:
        print("⚠️ 검출된 객체 없음")
    print("=" * 70)

    return predictions


def batch_predict(image_dir, model_path, conf_threshold=None, save_results=False, output_dir=None):
    """
    디렉토리 내 모든 이미지에 대해 예측 수행
    
    Args:
        image_dir: 이미지 디렉토리 경로
        model_path: 학습된 모델 경로
        conf_threshold: 신뢰도 임계값
        save_results: 결과 저장 여부
        output_dir: 결과 저장 디렉토리 (None이면 원본 디렉토리)
    
    Returns:
        모든 예측 결과 딕셔너리
    """
    image_dir = Path(image_dir)
    
    # 이미지 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(image_dir.glob(ext)))
    
    print(f"\n📁 총 {len(image_files)}개 이미지 발견")
    
    # 출력 디렉토리 설정
    if save_results and output_dir is None:
        output_dir = image_dir / 'predictions'
        output_dir.mkdir(exist_ok=True)
    elif save_results:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 신뢰도 임계값 설정
    if conf_threshold is None:
        conf_threshold = PREDICTION_CONFIG['conf_threshold']
    
    # 배치 예측
    all_predictions = {}
    
    print("\n🔄 예측 진행 중...\n")
    for img_path in image_files:
        predictions = quick_predict(
            img_path, 
            model_path, 
            conf_threshold, 
            visualize=False,
            save_result=save_results
        )
        all_predictions[str(img_path)] = predictions
    
    print(f"\n✅ 배치 예측 완료! 총 {len(all_predictions)}개 이미지 처리")
    
    return all_predictions


def main():
    """메인 함수 (커맨드라인 인터페이스)"""
    parser = argparse.ArgumentParser(description='딸기 OK/NG 분류 YOLO 모델 예측')
    
    parser.add_argument('--image', type=str, default=None,
                        help='예측할 이미지 파일 경로')
    parser.add_argument('--dir', type=str, default=None,
                        help='예측할 이미지 디렉토리 경로')
    parser.add_argument('--model', type=str, required=True,
                        help='학습된 모델 경로 (예: runs/detect/strawberry_ok_ng/weights/best.pt)')
    parser.add_argument('--conf', type=float, default=None,
                        help='신뢰도 임계값')
    parser.add_argument('--save', action='store_true',
                        help='결과를 이미지로 저장')
    parser.add_argument('--no-visualize', action='store_true',
                        help='시각화 비활성화')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 입력 검증
    if args.image is None and args.dir is None:
        parser.error("--image 또는 --dir 중 하나는 필수입니다.")
    
    if args.image is not None and args.dir is not None:
        parser.error("--image와 --dir은 동시에 사용할 수 없습니다.")
    
    # 예측 실행
    if args.image:
        # 단일 이미지 예측
        quick_predict(
            image_path=args.image,
            model_path=args.model,
            conf_threshold=args.conf,
            visualize=not args.no_visualize,
            save_result=args.save
        )
    else:
        # 배치 예측
        batch_predict(
            image_dir=args.dir,
            model_path=args.model,
            conf_threshold=args.conf,
            save_results=args.save,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()

