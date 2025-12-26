"""
SAHI 라이브러리를 사용한 YOLO 추론
대용량 이미지를 자동으로 슬라이싱하여 객체 탐지 수행
"""
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
import cv2
import numpy as np
from pathlib import Path

def predict_with_sahi(
    image_path: str,
    model_path: str,
    model_type: str = "yolov8",  # yolov8 또는 yolov5
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.3,
    overlap_width_ratio: float = 0.3,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.3,
    postprocess_match_threshold: float = 0.5,
    save_path: str = "sahi_result.jpg"
):
    """
    SAHI를 사용하여 대용량 이미지에서 객체 탐지를 수행합니다.
    
    Args:
        image_path: 입력 이미지 경로
        model_path: YOLO 모델 경로 (.pt 파일)
        model_type: 모델 타입 ('yolov8' 또는 'yolov5')
        slice_height: 슬라이스 높이
        slice_width: 슬라이스 너비
        overlap_height_ratio: 높이 오버랩 비율 (0.0~1.0)
        overlap_width_ratio: 너비 오버랩 비율 (0.0~1.0)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        postprocess_match_threshold: 슬라이스 간 매칭을 위한 IoU threshold
        save_path: 결과 이미지 저장 경로
    """
    print("=" * 70)
    print("SAHI를 사용한 객체 탐지")
    print("=" * 70)
    
    # 1. 모델 로드
    print(f"\n📦 모델 로드: {model_path}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device="cpu"  # 'cuda:0' for GPU
    )
    
    # 2. 이미지 로드
    print(f"📷 이미지 로드: {image_path}")
    image = read_image(image_path)
    print(f"   크기: {image.shape}")
    
    # 3. 슬라이스 예측 수행
    print(f"\n🔍 슬라이스 추론 시작...")
    print(f"   - 슬라이스 크기: {slice_width}x{slice_height}")
    print(f"   - 오버랩 비율: {overlap_width_ratio*100:.0f}%")
    print(f"   - Confidence threshold: {conf_threshold}")
    print(f"   - IoU threshold: {iou_threshold}")
    
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        postprocess_type="NMS",  # 'NMS' 또는 'GREEDYNMM'
        postprocess_match_metric="IOS",  # 'IOU' 또는 'IOS'
        postprocess_match_threshold=postprocess_match_threshold,
        postprocess_class_agnostic=False  # False로 설정하여 클래스별 NMS
    )
    
    # 4. 결과 분석
    print(f"\n✅ 추론 완료")
    print(f"🎯 검출 개수: {len(result.object_prediction_list)}")
    
    # 클래스별 통계
    class_counts = {}
    confidences = []
    
    for i, pred in enumerate(result.object_prediction_list):
        class_id = pred.category.id
        class_name = pred.category.name
        confidence = pred.score.value
        bbox = pred.bbox.to_voc_bbox()  # [x1, y1, x2, y2]
        
        # 통계 수집
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
        confidences.append(confidence)
        
        print(f"   [{i+1}] 클래스: {class_name} (ID: {class_id}), "
              f"신뢰도: {confidence:.3f}, bbox: {bbox}")
    
    # 5. 통계 출력
    if len(result.object_prediction_list) > 0:
        print(f"\n📊 검출 통계:")
        for class_name, count in class_counts.items():
            print(f"   - {class_name}: {count}개")
        print(f"\n📈 Confidence 분포:")
        print(f"   - 평균: {np.mean(confidences):.3f}")
        print(f"   - 최대: {np.max(confidences):.3f}")
        print(f"   - 최소: {np.min(confidences):.3f}")
    
    # 6. 슬라이스 영역 시각화
    slice_vis_path = save_path.replace(".jpg", "_slices.jpg")
    visualize_slices(image_path, slice_height, slice_width, 
                    overlap_height_ratio, overlap_width_ratio,
                    slice_vis_path)
    
    # 7. 결과 시각화 및 저장
    print(f"\n💾 결과 저장: {save_path}")
    result.export_visuals(export_dir=str(Path(save_path).parent))
    
    # SAHI의 기본 시각화 파일명 변경
    default_export_path = Path(save_path).parent / "prediction_visual.png"
    if default_export_path.exists():
        default_export_path.rename(save_path)
        print(f"   → {save_path}")
    
    # 추가 커스텀 시각화 (선택사항)
    vis_image = visualize_custom(image_path, result.object_prediction_list, save_path.replace(".jpg", "_custom.jpg"))
    
    print("=" * 70)
    
    return result.object_prediction_list


def visualize_slices(image_path: str, slice_height: int, slice_width: int, 
                     overlap_height_ratio: float, overlap_width_ratio: float,
                     save_path: str):
    """
    슬라이스 영역을 시각화
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ 이미지를 불러올 수 없습니다: {image_path}")
        return None
    
    height, width = image.shape[:2]
    
    # 슬라이스 계산
    stride_h = int(slice_height * (1 - overlap_height_ratio))
    stride_w = int(slice_width * (1 - overlap_width_ratio))
    
    # 반투명 오버레이 생성
    overlay = image.copy()
    
    slice_count = 0
    for y in range(0, height, stride_h):
        for x in range(0, width, stride_w):
            # 슬라이스 경계 계산
            x1 = x
            y1 = y
            x2 = min(x + slice_width, width)
            y2 = min(y + slice_height, height)
            
            slice_count += 1
            
            # 교차하는 색상으로 경계 그리기
            color = (0, 255, 255) if (slice_count % 2 == 0) else (255, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            
            # 슬라이스 번호 표시
            font_scale = max(0.6, min(width, height) / 2000)
            thickness = max(1, int(2 * font_scale))
            text = f"#{slice_count}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # 텍스트 배경
            text_x = x1 + 10
            text_y = y1 + text_size[1] + 10
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 5, text_y + 5),
                         (0, 0, 0), -1)
            
            # 텍스트
            cv2.putText(overlay, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # 반투명 블렌딩
    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    cv2.imwrite(save_path, result)
    print(f"💾 슬라이스 시각화 저장: {save_path} (총 {slice_count}개 슬라이스)")
    
    return result


def visualize_custom(image_path: str, predictions: list, save_path: str):
    """
    커스텀 시각화 (OK/NG 색상 구분)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ 이미지를 불러올 수 없습니다: {image_path}")
        return None
    
    # 클래스별 색상 (0: OK=초록, 1: NG=빨강)
    colors = {
        0: (0, 255, 0),    # OK: 초록
        1: (0, 0, 255),    # NG: 빨강
        "default": (255, 0, 0)  # 기본: 파랑
    }
    
    class_names = {0: "OK", 1: "NG"}
    
    # 이미지 크기에 따라 동적 조정
    img_area = image.shape[0] * image.shape[1]
    scale_factor = np.sqrt(img_area / (640 * 640))
    thickness = max(1, int(2 * scale_factor))
    font_scale = max(0.5, 0.6 * scale_factor)
    
    for pred in predictions:
        class_id = pred.category.id
        class_name = pred.category.name if hasattr(pred.category, 'name') else class_names.get(class_id, str(class_id))
        confidence = pred.score.value
        bbox = pred.bbox.to_voc_bbox()  # [x1, y1, x2, y2]
        
        x1, y1, x2, y2 = map(int, bbox)
        color = colors.get(class_id, colors["default"])
        
        # 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 라벨 그리기
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # 배경 박스
        cv2.rectangle(image, 
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)
        
        # 텍스트
        cv2.putText(image, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale,
                   (255, 255, 255),
                   thickness)
        
        # 중심점
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(image, (cx, cy), max(3, int(5 * scale_factor)), color, -1)
    
    cv2.imwrite(save_path, image)
    print(f"💾 커스텀 시각화 저장: {save_path}")
    
    return image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAHI를 사용한 YOLO 추론")
    parser.add_argument("--image", type=str, 
                       default="/Users/temp/내 드라이브(codejeteho123@gmail.com)/ComputerVision/sample_1920x1080.jpg",
                       help="테스트 이미지 경로")
    parser.add_argument("--model", type=str,
                       default="runs/detect/strawberry_ok_ng/weights/best.pt",
                       help="YOLO 모델 경로 (.pt 파일)")
    parser.add_argument("--model-type", type=str, default="yolov8",
                       choices=["yolov8", "yolov5"],
                       help="모델 타입")
    parser.add_argument("--slice-size", type=int, default=640,
                       help="슬라이스 크기 (정사각형)")
    parser.add_argument("--overlap", type=float, default=0.3,
                       help="오버랩 비율 (0.0~1.0)")
    parser.add_argument("--conf", type=float, default=0.85,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3,
                       help="IoU threshold")
    parser.add_argument("--output", type=str, default="sahi_result.jpg",
                       help="출력 이미지 경로")
    
    args = parser.parse_args()
    
    try:
        predictions = predict_with_sahi(
            image_path=args.image,
            model_path=args.model,
            model_type=args.model_type,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_path=args.output
        )
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n💡 SAHI 설치 확인:")
        print("   pip install sahi")
        raise

