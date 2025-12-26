"""
SAHI를 사용하여 오분류를 진단하는 스크립트
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import cv2
import numpy as np
from pathlib import Path

def diagnose_sahi_predictions(
    model_path: str,
    image_path: str,
    slice_size: int = 640,
    overlap: float = 0.3,
    conf_threshold: float = 0.3
):
    """
    SAHI로 예측한 결과를 분석하여 오분류를 진단합니다.
    """
    print("=" * 70)
    print("SAHI 오분류 진단")
    print("=" * 70)
    print(f"\n📷 이미지: {image_path}")
    print(f"🎯 실제 라벨: 모든 딸기 = OK")
    print(f"🔍 Confidence threshold: {conf_threshold}")
    
    # 모델 로드
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device="cpu"
    )
    
    # 이미지 로드
    image = read_image(image_path)
    print(f"   이미지 크기: {image.shape}")
    
    # 슬라이스 예측
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False
    )
    
    # 결과 분석
    ok_detections = []
    ng_detections = []
    
    for pred in result.object_prediction_list:
        class_id = pred.category.id
        class_name = pred.category.name if hasattr(pred.category, 'name') else ("OK" if class_id == 0 else "NG")
        confidence = pred.score.value
        bbox = pred.bbox.to_voc_bbox()
        
        detection = {
            'class_name': class_name,
            'class_id': class_id,
            'confidence': confidence,
            'bbox': bbox,
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        }
        
        if class_id == 0:
            ok_detections.append(detection)
        else:
            ng_detections.append(detection)
    
    # 결과 출력
    print(f"\n📊 검출 결과:")
    print(f"   총 검출: {len(result.object_prediction_list)}개")
    print(f"   ✅ OK: {len(ok_detections)}개 (정답)")
    print(f"   ❌ NG: {len(ng_detections)}개 (오분류)")
    
    accuracy = len(ok_detections) / max(len(result.object_prediction_list), 1) * 100
    print(f"\n📈 정확도: {accuracy:.1f}%")
    
    # OK 검출 상세
    if len(ok_detections) > 0:
        print(f"\n✅ OK 검출 (정답):")
        ok_confs = [d['confidence'] for d in ok_detections]
        ok_areas = [d['area'] for d in ok_detections]
        for i, det in enumerate(sorted(ok_detections, key=lambda x: -x['confidence'])[:10]):
            print(f"   [{i+1}] 신뢰도: {det['confidence']:.3f}, 면적: {det['area']:.0f}px")
        print(f"   평균 신뢰도: {np.mean(ok_confs):.3f}")
        print(f"   평균 면적: {np.mean(ok_areas):.0f}px")
    
    # NG 검출 상세 (오분류)
    if len(ng_detections) > 0:
        print(f"\n❌ NG 검출 (오분류 - 실제로는 OK):")
        ng_confs = [d['confidence'] for d in ng_detections]
        ng_areas = [d['area'] for d in ng_detections]
        for i, det in enumerate(sorted(ng_detections, key=lambda x: -x['confidence'])[:10]):
            print(f"   [{i+1}] 신뢰도: {det['confidence']:.3f}, 면적: {det['area']:.0f}px")
        print(f"   평균 신뢰도: {np.mean(ng_confs):.3f}")
        print(f"   평균 면적: {np.mean(ng_areas):.0f}px")
    
    # 개별 검출 이미지 저장
    save_detection_crops(image_path, ok_detections, ng_detections)
    
    # 분석
    print("\n" + "=" * 70)
    print("🔬 분석")
    print("=" * 70)
    
    if len(ok_detections) > 0 and len(ng_detections) > 0:
        ok_avg_conf = np.mean([d['confidence'] for d in ok_detections])
        ng_avg_conf = np.mean([d['confidence'] for d in ng_detections])
        
        print(f"\n신뢰도 비교:")
        print(f"   OK (정답): {ok_avg_conf:.3f}")
        print(f"   NG (오분류): {ng_avg_conf:.3f}")
        
        if ng_avg_conf < ok_avg_conf:
            print(f"   → NG의 신뢰도가 더 낮음: Threshold를 {ng_avg_conf:.2f} 이상으로 설정하면 오분류 감소")
        else:
            print(f"   → NG의 신뢰도가 높음: 모델 재학습 필요")
    
    # 권장사항
    print("\n💡 권장사항:")
    if len(ng_detections) > len(ok_detections):
        print("   ⚠️ 오분류가 정답보다 많음 - 심각한 문제!")
        print("   1. 학습 데이터의 라벨 확인")
        print("   2. OK/NG 기준 재정의")
        print("   3. 모델 재학습 필요")
    elif len(ng_detections) > 0:
        # NG 중 최소 신뢰도 찾기
        min_ng_conf = min([d['confidence'] for d in ng_detections])
        print(f"   1. Confidence threshold를 {min_ng_conf:.2f} 이상으로 설정")
        print(f"      python test_sahi.py --conf {min_ng_conf:.2f}")
        print(f"   2. 또는 모델 재학습으로 OK/NG 구분력 향상")
    else:
        print("   ✅ 오분류 없음 - 모델 성능 양호!")
    
    print("=" * 70)
    
    return ok_detections, ng_detections


def save_detection_crops(image_path: str, ok_detections: list, ng_detections: list):
    """
    검출된 영역을 개별 이미지로 저장
    """
    image = cv2.imread(image_path)
    output_dir = Path("detection_crops")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 검출 영역 저장: {output_dir}/")
    
    # OK 저장
    for i, det in enumerate(ok_detections[:10]):  # 최대 10개
        x1, y1, x2, y2 = map(int, det['bbox'])
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            filename = f"OK_{i+1}_conf{det['confidence']:.3f}.jpg"
            cv2.imwrite(str(output_dir / filename), crop)
    
    # NG 저장 (오분류)
    for i, det in enumerate(ng_detections[:10]):  # 최대 10개
        x1, y1, x2, y2 = map(int, det['bbox'])
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            filename = f"NG_WRONG_{i+1}_conf{det['confidence']:.3f}.jpg"
            cv2.imwrite(str(output_dir / filename), crop)
    
    print(f"   ✓ OK (정답): {min(len(ok_detections), 10)}개")
    print(f"   ✓ NG (오분류): {min(len(ng_detections), 10)}개")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAHI 오분류 진단")
    parser.add_argument("--model", type=str,
                       default="runs/detect/strawberry_ok_ng/weights/best.pt",
                       help="모델 경로")
    parser.add_argument("--image", type=str,
                       default="sample.jpg",
                       help="전부 OK인 테스트 이미지")
    parser.add_argument("--slice-size", type=int, default=640,
                       help="슬라이스 크기")
    parser.add_argument("--overlap", type=float, default=0.3,
                       help="오버랩 비율")
    parser.add_argument("--conf", type=float, default=0.3,
                       help="Confidence threshold")
    
    args = parser.parse_args()
    
    ok_dets, ng_dets = diagnose_sahi_predictions(
        args.model,
        args.image,
        args.slice_size,
        args.overlap,
        args.conf
    )

