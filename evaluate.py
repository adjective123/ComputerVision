"""
모델 평가 스크립트
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

from config import setup_matplotlib_font


def evaluate_model(model_path, data_yaml=None, split='val', **kwargs):
    """
    YOLO 모델 평가
    
    Args:
        model_path: 평가할 모델 경로
        data_yaml: 데이터셋 YAML 파일 경로 (옵션)
        split: 평가할 데이터셋 ('val' 또는 'test')
        **kwargs: 추가 평가 파라미터
    
    Returns:
        평가 결과 객체
    """
    # 한글 폰트 설정
    setup_matplotlib_font()
    
    print("\n" + "=" * 70)
    print("📊 YOLO 모델 평가")
    print("=" * 70)
    print(f"🤖 모델: {model_path}")
    print(f"📁 데이터셋: {split}")
    print("=" * 70 + "\n")
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 평가 파라미터 설정
    eval_params = {'split': split}
    if data_yaml is not None:
        eval_params['data'] = str(data_yaml)
    eval_params.update(kwargs)
    
    # 평가 실행
    metrics = model.val(**eval_params)
    
    print("\n" + "=" * 70)
    print("✅ 평가 완료!")
    print("=" * 70)
    
    # 주요 지표 출력
    print("\n📈 주요 성능 지표:")
    try:
        # Object Detection 지표
        if hasattr(metrics, 'box'):
            print(f"  • mAP50: {metrics.box.map50:.4f}")
            print(f"  • mAP50-95: {metrics.box.map:.4f}")
            print(f"  • Precision: {metrics.box.mp:.4f}")
            print(f"  • Recall: {metrics.box.mr:.4f}")
        
        # 클래스별 성능
        if hasattr(metrics, 'results_dict'):
            results = metrics.results_dict
            print(f"\n📊 상세 결과:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"  • {key}: {value:.4f}")
    except Exception as e:
        print(f"⚠️ 지표 출력 중 오류: {e}")
    
    print("=" * 70 + "\n")
    
    return metrics


def main():
    """메인 함수 (커맨드라인 인터페이스)"""
    parser = argparse.ArgumentParser(description='딸기 OK/NG 분류 YOLO 모델 평가')
    
    parser.add_argument('--model', type=str, required=True,
                        help='평가할 모델 경로 (예: runs/detect/strawberry_ok_ng/weights/best.pt)')
    parser.add_argument('--data', type=str, default=None,
                        help='데이터셋 YAML 파일 경로')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='평가할 데이터셋 분할')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='이미지 크기')
    parser.add_argument('--batch', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='신뢰도 임계값')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU 임계값 (NMS)')
    
    args = parser.parse_args()
    
    # 평가 실행
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou
    )


if __name__ == "__main__":
    main()

