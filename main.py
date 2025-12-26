"""
전체 파이프라인 실행 스크립트
데이터 준비 → 학습 → 평가 → 시각화를 한 번에 실행
"""

import argparse
from pathlib import Path

from dataset_preparation import main as prepare_dataset
from train import train_model
from evaluate import evaluate_model
from visualize import visualize_training_results, visualize_sample_predictions
from config import BASE_DIR


def run_full_pipeline(
    skip_data_prep=False,
    model_name='yolo11s.pt',
    epochs=100,
    batch=16,
    visualize_results=True
):
    """
    전체 파이프라인 실행
    
    Args:
        skip_data_prep: 데이터 준비 단계 건너뛰기
        model_name: 사용할 YOLO 모델
        epochs: 에포크 수
        batch: 배치 크기
        visualize_results: 결과 시각화 여부
    """
    print("\n" + "=" * 70)
    print("🚀 딸기 OK/NG 분류 YOLO 전체 파이프라인 시작")
    print("=" * 70 + "\n")
    
    # 1. 데이터셋 준비
    if not skip_data_prep:
        print("\n" + "=" * 70)
        print("📦 [1/4] 데이터셋 준비")
        print("=" * 70)
        stats, yaml_path = prepare_dataset()
    else:
        print("\n⏭️ 데이터 준비 단계 건너뛰기")
        yaml_path = BASE_DIR / 'strawberry_ok_ng.yaml'
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"YAML 파일을 찾을 수 없습니다: {yaml_path}\n"
                "데이터 준비를 먼저 실행하거나 --skip-data-prep 옵션을 제거하세요."
            )
    
    # 2. 모델 학습
    print("\n" + "=" * 70)
    print("🤖 [2/4] 모델 학습")
    print("=" * 70)
    train_results = train_model(
        yaml_path=yaml_path,
        model_path=model_name,
        epochs=epochs,
        batch=batch
    )
    
    # 학습된 모델 경로
    best_model_path = Path("runs/detect/strawberry_ok_ng/weights/best.pt")
    
    # 3. 모델 평가
    print("\n" + "=" * 70)
    print("📊 [3/4] 모델 평가")
    print("=" * 70)
    eval_metrics = evaluate_model(
        model_path=str(best_model_path),
        data_yaml=str(yaml_path)
    )
    
    # 4. 결과 시각화
    if visualize_results:
        print("\n" + "=" * 70)
        print("🎨 [4/4] 결과 시각화")
        print("=" * 70)
        
        # 학습 결과 시각화
        results_dir = Path("runs/detect/strawberry_ok_ng")
        visualize_training_results(results_dir, show=True)
        
        # 샘플 예측 시각화
        visualize_sample_predictions(
            model_path=str(best_model_path),
            n_samples=5,
            show=True
        )
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("🎉 전체 파이프라인 완료!")
    print("=" * 70)
    print(f"\n📁 저장된 파일:")
    print(f"  ✅ 최고 성능 모델: {best_model_path}")
    print(f"  ✅ YAML 설정: {yaml_path}")
    print(f"  ✅ 학습 결과: {results_dir}")
    print("\n💡 다음 단계:")
    print(f"  • 예측 실행: python predict.py --model {best_model_path} --image <이미지경로>")
    print(f"  • 결과 시각화: python visualize.py training --results-dir {results_dir}")
    print("=" * 70 + "\n")


def main():
    """메인 함수 (커맨드라인 인터페이스)"""
    parser = argparse.ArgumentParser(
        description='딸기 OK/NG 분류 YOLO 전체 파이프라인 실행'
    )
    
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='데이터 준비 단계 건너뛰기')
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                        help='YOLO 모델 (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11x.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='에포크 수')
    parser.add_argument('--batch', type=int, default=16,
                        help='배치 크기')
    parser.add_argument('--no-visualize', action='store_true',
                        help='결과 시각화 비활성화')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    run_full_pipeline(
        skip_data_prep=args.skip_data_prep,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        visualize_results=not args.no_visualize
    )


if __name__ == "__main__":
    main()

