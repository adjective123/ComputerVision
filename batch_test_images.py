"""
딸기이미지 폴더의 이미지들을 배치로 테스트하는 스크립트
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import random

def visualize_detections(image: np.ndarray, predictions: list):
    """
    검출 결과를 시각화합니다.
    
    Args:
        image: RGB 이미지 (SAHI read_image 출력)
        predictions: 검출 결과 리스트
    """
    # RGB -> BGR 변환 (OpenCV는 BGR 사용)
    vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 클래스별 색상 (0: OK=초록, 1: NG=빨강)
    colors = {
        0: (0, 255, 0),    # OK: 초록
        1: (0, 0, 255),    # NG: 빨강
    }
    
    class_names = {0: "OK", 1: "NG"}
    
    # 이미지 크기에 따라 동적 조정
    img_area = vis_image.shape[0] * vis_image.shape[1]
    scale_factor = np.sqrt(img_area / (640 * 640))
    thickness = max(1, int(2 * scale_factor))
    font_scale = max(0.5, 0.6 * scale_factor)
    
    for pred in predictions:
        class_id = pred.category.id
        class_name = class_names.get(class_id, str(class_id))
        confidence = pred.score.value
        bbox = pred.bbox.to_voc_bbox()  # [x1, y1, x2, y2]
        
        x1, y1, x2, y2 = map(int, bbox)
        color = colors.get(class_id, (255, 0, 0))
        
        # 박스 그리기
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
        
        # 라벨 그리기
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 배경 박스
        cv2.rectangle(vis_image, 
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)
        
        # 텍스트
        cv2.putText(vis_image, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale,
                   (255, 255, 255),
                   thickness)
        
        # 중심점
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis_image, (cx, cy), max(3, int(5 * scale_factor)), color, -1)
    
    return vis_image


def batch_test_images(
    model_path: str,
    ok_dir: str,
    ng_dir: str,
    slice_size: int = 640,
    overlap: float = 0.3,
    conf_threshold: float = 0.85,
    num_samples: int = None,
    save_results: bool = True,
    save_images: bool = True
):
    """
    OK와 NG 폴더의 이미지들을 배치로 테스트합니다.
    
    Args:
        model_path: 모델 경로
        ok_dir: OK 이미지 폴더
        ng_dir: NG 이미지 폴더
        slice_size: 슬라이스 크기
        overlap: 오버랩 비율
        conf_threshold: Confidence threshold
        num_samples: 각 클래스당 테스트할 샘플 수 (None이면 전체)
        save_results: 결과를 JSON으로 저장할지 여부
    """
    print("=" * 70)
    print("배치 이미지 테스트")
    print("=" * 70)
    print(f"🎯 Confidence threshold: {conf_threshold}")
    print(f"📦 Slice size: {slice_size}x{slice_size}")
    print(f"🔄 Overlap: {overlap*100:.0f}%")
    
    # 모델 로드
    print(f"\n📦 모델 로드: {model_path}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device="cpu"
    )
    
    # 이미지 파일 수집 (랜덤 샘플링)
    all_ok_images = list(Path(ok_dir).glob("*.jpg"))
    all_ng_images = list(Path(ng_dir).glob("*.jpg"))
    
    if num_samples is not None:
        ok_images = random.sample(all_ok_images, min(num_samples, len(all_ok_images)))
        ng_images = random.sample(all_ng_images, min(num_samples, len(all_ng_images)))
    else:
        ok_images = all_ok_images
        ng_images = all_ng_images
    
    print(f"\n📂 테스트 이미지:")
    print(f"   ✅ OK: {len(ok_images)}개")
    print(f"   ❌ NG: {len(ng_images)}개")
    print(f"   총: {len(ok_images) + len(ng_images)}개")
    
    # 결과 저장용
    results = {
        'ok_images': [],
        'ng_images': [],
        'summary': {}
    }
    
    # 이미지 저장 디렉토리 생성
    if save_images:
        output_dir = Path("batch_test_results")
        output_dir.mkdir(exist_ok=True)
        (output_dir / "ok_correct").mkdir(exist_ok=True)
        (output_dir / "ok_wrong").mkdir(exist_ok=True)
        (output_dir / "ng_correct").mkdir(exist_ok=True)
        (output_dir / "ng_wrong").mkdir(exist_ok=True)
        print(f"\n💾 결과 이미지 저장 디렉토리: {output_dir}/")
    
    # OK 이미지 테스트
    print("\n" + "=" * 70)
    print("✅ OK 이미지 테스트 (실제 라벨: OK)")
    print("=" * 70)
    
    ok_correct = 0
    ok_wrong = 0
    ok_details = []
    
    for img_path in tqdm(ok_images, desc="OK 테스트"):
        try:
            image = read_image(str(img_path))
            
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
                postprocess_class_agnostic=False,
                verbose=0
            )
            
            # 결과 분석
            ok_count = sum(1 for pred in result.object_prediction_list if pred.category.id == 0)
            ng_count = sum(1 for pred in result.object_prediction_list if pred.category.id == 1)
            
            # 정확도 계산 (OK가 더 많으면 정답)
            is_correct = ok_count >= ng_count if len(result.object_prediction_list) > 0 else True
            
            if is_correct:
                ok_correct += 1
            else:
                ok_wrong += 1
            
            detail = {
                'image': str(img_path.name),
                'total_detections': len(result.object_prediction_list),
                'ok_count': ok_count,
                'ng_count': ng_count,
                'correct': is_correct
            }
            ok_details.append(detail)
            results['ok_images'].append(detail)
            
            # 결과 이미지 저장
            if save_images:
                vis_image = visualize_detections(image, result.object_prediction_list)
                subdir = "ok_correct" if is_correct else "ok_wrong"
                save_path = output_dir / subdir / f"{img_path.stem}_result.jpg"
                cv2.imwrite(str(save_path), vis_image)
            
        except Exception as e:
            print(f"\n⚠️ 오류 ({img_path.name}): {e}")
    
    # NG 이미지 테스트
    print("\n" + "=" * 70)
    print("❌ NG 이미지 테스트 (실제 라벨: NG)")
    print("=" * 70)
    
    ng_correct = 0
    ng_wrong = 0
    ng_details = []
    
    for img_path in tqdm(ng_images, desc="NG 테스트"):
        try:
            image = read_image(str(img_path))
            
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
                postprocess_class_agnostic=False,
                verbose=0
            )
            
            # 결과 분석
            ok_count = sum(1 for pred in result.object_prediction_list if pred.category.id == 0)
            ng_count = sum(1 for pred in result.object_prediction_list if pred.category.id == 1)
            
            # 정확도 계산 (NG가 더 많으면 정답)
            is_correct = ng_count > ok_count if len(result.object_prediction_list) > 0 else False
            
            if is_correct:
                ng_correct += 1
            else:
                ng_wrong += 1
            
            detail = {
                'image': str(img_path.name),
                'total_detections': len(result.object_prediction_list),
                'ok_count': ok_count,
                'ng_count': ng_count,
                'correct': is_correct
            }
            ng_details.append(detail)
            results['ng_images'].append(detail)
            
            # 결과 이미지 저장
            if save_images:
                vis_image = visualize_detections(image, result.object_prediction_list)
                subdir = "ng_correct" if is_correct else "ng_wrong"
                save_path = output_dir / subdir / f"{img_path.stem}_result.jpg"
                cv2.imwrite(str(save_path), vis_image)
            
        except Exception as e:
            print(f"\n⚠️ 오류 ({img_path.name}): {e}")
    
    # 결과 요약
    print("\n" + "=" * 70)
    print("📊 테스트 결과 요약")
    print("=" * 70)
    
    total_images = len(ok_images) + len(ng_images)
    total_correct = ok_correct + ng_correct
    total_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    
    ok_accuracy = (ok_correct / len(ok_images) * 100) if len(ok_images) > 0 else 0
    ng_accuracy = (ng_correct / len(ng_images) * 100) if len(ng_images) > 0 else 0
    
    print(f"\n✅ OK 이미지 (실제: OK):")
    print(f"   정답: {ok_correct}/{len(ok_images)}개 ({ok_accuracy:.1f}%)")
    print(f"   오답: {ok_wrong}개")
    
    print(f"\n❌ NG 이미지 (실제: NG):")
    print(f"   정답: {ng_correct}/{len(ng_images)}개 ({ng_accuracy:.1f}%)")
    print(f"   오답: {ng_wrong}개")
    
    print(f"\n🎯 전체 정확도: {total_correct}/{total_images}개 ({total_accuracy:.1f}%)")
    
    # Confusion Matrix
    print(f"\n📊 Confusion Matrix:")
    print(f"                예측 OK    예측 NG")
    print(f"   실제 OK:     {ok_correct:4d}       {ok_wrong:4d}")
    print(f"   실제 NG:     {ng_wrong:4d}       {ng_correct:4d}")
    
    # 오분류 사례 출력
    if ok_wrong > 0:
        print(f"\n⚠️ OK를 NG로 오분류한 사례:")
        wrong_ok = [d for d in ok_details if not d['correct']]
        for i, detail in enumerate(wrong_ok[:5], 1):
            print(f"   [{i}] {detail['image']}: OK {detail['ok_count']}개, NG {detail['ng_count']}개")
    
    if ng_wrong > 0:
        print(f"\n⚠️ NG를 OK로 오분류한 사례:")
        wrong_ng = [d for d in ng_details if not d['correct']]
        for i, detail in enumerate(wrong_ng[:5], 1):
            print(f"   [{i}] {detail['image']}: OK {detail['ok_count']}개, NG {detail['ng_count']}개")
    
    # 결과 저장
    if save_results:
        results['summary'] = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'conf_threshold': conf_threshold,
            'slice_size': slice_size,
            'overlap': overlap,
            'total_images': total_images,
            'ok_images': len(ok_images),
            'ng_images': len(ng_images),
            'ok_correct': ok_correct,
            'ok_wrong': ok_wrong,
            'ok_accuracy': ok_accuracy,
            'ng_correct': ng_correct,
            'ng_wrong': ng_wrong,
            'ng_accuracy': ng_accuracy,
            'total_correct': total_correct,
            'total_accuracy': total_accuracy
        }
        
        output_file = f"batch_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {output_file}")
    
    if save_images:
        print(f"\n🖼️ 결과 이미지:")
        print(f"   ✅ OK 정답: batch_test_results/ok_correct/ ({ok_correct}개)")
        print(f"   ❌ OK 오답: batch_test_results/ok_wrong/ ({ok_wrong}개)")
        print(f"   ✅ NG 정답: batch_test_results/ng_correct/ ({ng_correct}개)")
        print(f"   ❌ NG 오답: batch_test_results/ng_wrong/ ({ng_wrong}개)")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="배치 이미지 테스트")
    parser.add_argument("--model", type=str,
                       default="runs/detect/strawberry_ok_ng/weights/best.pt",
                       help="모델 경로")
    parser.add_argument("--ok-dir", type=str,
                       default="딸기이미지/정상",
                       help="OK 이미지 폴더")
    parser.add_argument("--ng-dir", type=str,
                       default="딸기이미지/NG",
                       help="NG 이미지 폴더")
    parser.add_argument("--slice-size", type=int, default=640,
                       help="슬라이스 크기")
    parser.add_argument("--overlap", type=float, default=0.3,
                       help="오버랩 비율")
    parser.add_argument("--conf", type=float, default=0.85,
                       help="Confidence threshold")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="각 클래스당 테스트할 샘플 수 (None이면 전체)")
    parser.add_argument("--save-images", action="store_true", default=True,
                       help="결과 이미지 저장 여부")
    
    args = parser.parse_args()
    
    results = batch_test_images(
        model_path=args.model,
        ok_dir=args.ok_dir,
        ng_dir=args.ng_dir,
        slice_size=args.slice_size,
        overlap=args.overlap,
        conf_threshold=args.conf,
        num_samples=args.num_samples,
        save_results=True,
        save_images=args.save_images
    )

