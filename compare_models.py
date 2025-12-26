"""
PyTorch 모델과 ONNX 모델의 추론 결과를 비교하는 스크립트
"""
import numpy as np
import cv2
from ultralytics import YOLO
from berryModel import YOLOPredictor

def compare_models(image_path: str, pt_model_path: str, onnx_model_path: str):
    """
    PyTorch 모델과 ONNX 모델의 추론 결과를 비교합니다.
    
    Args:
        image_path: 테스트 이미지 경로
        pt_model_path: PyTorch 모델 (.pt) 경로
        onnx_model_path: ONNX 모델 (.onnx) 경로
    """
    print("=" * 70)
    print("모델 비교 검증")
    print("=" * 70)
    
    # 1. PyTorch 모델로 추론
    print("\n1️⃣ PyTorch 모델 추론")
    print("-" * 70)
    pt_model = YOLO(pt_model_path)
    pt_results = pt_model.predict(
        image_path,
        conf=0.3,
        iou=0.3,
        save=True,
        save_txt=False,
        project="compare_results",
        name="pytorch",
        exist_ok=True
    )
    
    pt_detections = []
    for r in pt_results:
        print(f"✅ PyTorch 검출 개수: {len(r.boxes)}")
        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            pt_detections.append({
                'class': cls,
                'confidence': conf,
                'bbox': xyxy
            })
            print(f"   [{i+1}] 클래스: {cls}, 신뢰도: {conf:.3f}, bbox: {xyxy}")
    
    # 2. ONNX 모델로 추론
    print("\n2️⃣ ONNX 모델 추론")
    print("-" * 70)
    onnx_predictor = YOLOPredictor(onnx_model_path)
    onnx_results = onnx_predictor.predict(
        image_path,
        conf_threshold=0.3,
        iou_threshold=0.3,
        crop_size=640,
        overlap_ratio=0.3,
        visualize=True,
        save_path="compare_results/onnx_result.jpg"
    )
    
    print(f"✅ ONNX 검출 개수: {len(onnx_results)}")
    for i, det in enumerate(onnx_results):
        print(f"   [{i+1}] 클래스: {det['class']}, 신뢰도: {det['confidence']:.3f}, bbox: {det['bbox']}")
    
    # 3. 비교 결과
    print("\n3️⃣ 비교 결과")
    print("=" * 70)
    print(f"PyTorch 검출 개수: {len(pt_detections)}")
    print(f"ONNX 검출 개수: {len(onnx_results)}")
    print(f"차이: {abs(len(pt_detections) - len(onnx_results))}개")
    
    if len(pt_detections) > 0 and len(onnx_results) == 0:
        print("\n⚠️ 경고: PyTorch는 검출했지만 ONNX는 검출 실패")
        print("   → ONNX 변환 또는 전처리에 문제가 있을 수 있습니다")
    elif len(pt_detections) == 0 and len(onnx_results) == 0:
        print("\n⚠️ 경고: 두 모델 모두 검출 실패")
        print("   → 모델 자체의 성능 또는 이미지에 문제가 있을 수 있습니다")
    elif abs(len(pt_detections) - len(onnx_results)) > 5:
        print("\n⚠️ 경고: 검출 개수 차이가 큽니다")
        print("   → ONNX 변환 또는 후처리에 문제가 있을 수 있습니다")
    else:
        print("\n✅ 두 모델의 검출 개수가 비슷합니다")
    
    # 4. Confidence 분포 비교
    if len(pt_detections) > 0:
        print("\n4️⃣ Confidence 분포")
        print("-" * 70)
        pt_confs = [d['confidence'] for d in pt_detections]
        print(f"PyTorch:")
        print(f"   - 평균: {np.mean(pt_confs):.3f}")
        print(f"   - 최대: {np.max(pt_confs):.3f}")
        print(f"   - 최소: {np.min(pt_confs):.3f}")
        
        if len(onnx_results) > 0:
            onnx_confs = [d['confidence'] for d in onnx_results]
            print(f"ONNX:")
            print(f"   - 평균: {np.mean(onnx_confs):.3f}")
            print(f"   - 최대: {np.max(onnx_confs):.3f}")
            print(f"   - 최소: {np.min(onnx_confs):.3f}")
    
    print("\n" + "=" * 70)
    print(f"💾 결과 이미지:")
    print(f"   - PyTorch: compare_results/pytorch/")
    print(f"   - ONNX: compare_results/onnx_result.jpg")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch 모델과 ONNX 모델 비교")
    parser.add_argument("--image", type=str, default="/Users/temp/내 드라이브(codejeteho123@gmail.com)/ComputerVision/sample_1920x1080.jpg",
                       help="테스트 이미지 경로")
    parser.add_argument("--pt-model", type=str, 
                       default="runs/detect/strawberry_ok_ng/weights/best.pt",
                       help="PyTorch 모델 경로")
    parser.add_argument("--onnx-model", type=str,
                       default="runs/detect/strawberry_ok_ng/weights/best.onnx",
                       help="ONNX 모델 경로")
    
    args = parser.parse_args()
    
    compare_models(args.image, args.pt_model, args.onnx_model)

