"""
YOLO 모델 학습 스크립트
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

from config import BASE_DIR, TRAINING_CONFIG, setup_matplotlib_font
from utils import get_device, register_memory_callback, reset_unused_memory


def train_model(yaml_path=None, model_path=None, device=None, **kwargs):
    """
    YOLO 모델 학습
    
    Args:
        yaml_path: 데이터셋 YAML 파일 경로
        model_path: 사용할 YOLO 모델 경로 (기본: yolo11s.pt)
        device: 사용할 디바이스 (None이면 자동 선택)
        **kwargs: 추가 학습 파라미터
    
    Returns:
        학습 결과 객체
    """
    # 한글 폰트 설정
    setup_matplotlib_font()
    
    # 디바이스 선택
    if device is None:
        device = get_device()
    
    # 메모리 정리
    reset_unused_memory()
    
    # 메모리 정리 콜백 등록
    register_memory_callback()
    
    # YAML 경로 설정
    if yaml_path is None:
        yaml_path = BASE_DIR / 'strawberry_ok_ng.yaml'
    
    # 모델 경로 설정
    if model_path is None:
        model_path = TRAINING_CONFIG['model']
    
    print("\n" + "=" * 70)
    print("🚀 YOLO 모델 학습 시작")
    print("=" * 70)
    print(f"📁 데이터셋 설정: {yaml_path}")
    print(f"🤖 모델: {model_path}")
    print(f"💻 디바이스: {device}")
    print("=" * 70 + "\n")
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 학습 파라미터 병합
    train_params = TRAINING_CONFIG.copy()
    train_params.update(kwargs)
    train_params['data'] = str(yaml_path)
    train_params['device'] = device
    
    # 학습 시작
    results = model.train(**train_params)
    
    print("\n" + "=" * 70)
    print("✅ 학습 완료!")
    print("=" * 70)
    
    # 최고 성능 모델 경로 출력
    best_model_path = Path(train_params['project']) / train_params['name'] / 'weights' / 'best.pt'
    print(f"\n📁 저장된 모델 경로:")
    print(f"  ✅ 최고 성능: {best_model_path}")
    print(f"  ✅ 마지막: {best_model_path.parent / 'last.pt'}")
    print("=" * 70 + "\n")
    
    return results


def main():
    """메인 함수 (커맨드라인 인터페이스)"""
    parser = argparse.ArgumentParser(description='딸기 OK/NG 분류 YOLO 모델 학습')
    
    parser.add_argument('--yaml', type=str, default=None,
                        help='데이터셋 YAML 파일 경로')
    parser.add_argument('--model', type=str, default=None,
                        help='YOLO 모델 경로 (예: yolo11s.pt, yolo11m.pt)')
    parser.add_argument('--device', type=str, default=None,
                        help='사용할 디바이스 (cuda/mps/cpu, 기본값: 자동 선택)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='에포크 수')
    parser.add_argument('--batch', type=int, default=None,
                        help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='이미지 크기')
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    
    # 추가 파라미터 수집
    extra_params = {}
    if args.epochs is not None:
        extra_params['epochs'] = args.epochs
    if args.batch is not None:
        extra_params['batch'] = args.batch
    if args.imgsz is not None:
        extra_params['imgsz'] = args.imgsz
    if args.patience is not None:
        extra_params['patience'] = args.patience
    
    # 학습 실행
    train_model(
        yaml_path=args.yaml,
        model_path=args.model,
        device=args.device,
        **extra_params
    )


if __name__ == "__main__":
    main()

