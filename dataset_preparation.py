"""
데이터셋 준비 및 변환
- JSON 파일 수집
- Train/Val 분할
- YOLO 포맷으로 변환
- 멀티프로세싱 지원
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from config import (
    JSON_DIR, IMAGE_DIR, OUTPUT_DIR, CLASS_MAPPING, 
    VAL_RATIO, RANDOM_SEED, CLASS_NAMES, BASE_DIR
)
from utils import json_to_yolo, create_yaml_config


def collect_json_files():
    """JSON 파일 수집 및 매핑"""
    dataset_info = {}

    for category, class_id in CLASS_MAPPING.items():
        category_dir = JSON_DIR / category
        if category_dir.exists():
            json_files = list(category_dir.glob('*.json'))
            dataset_info[category] = {
                'files': json_files,
                'count': len(json_files),
                'class_id': class_id
            }
            print(f"📁 {category}: {len(json_files)}개 파일")

    # 모든 JSON 파일 수집
    all_json_files = []
    for category, info in dataset_info.items():
        all_json_files.extend([(f, info['class_id']) for f in info['files']])

    return dataset_info, all_json_files


def create_directory_structure():
    """데이터셋 디렉토리 구조 생성"""
    if OUTPUT_DIR.exists():
        print(f"⚠️ 기존 출력 디렉토리 삭제: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 디렉토리 구조 생성 완료: {OUTPUT_DIR}")


def process_single_file(args):
    """
    단일 파일 처리 함수 (멀티프로세싱용)
    
    Args:
        args: (json_file, class_id, split, image_dir, output_dir) 튜플
    
    Returns:
        (처리 결과, 클래스 ID) 튜플
    """
    json_file, class_id, split, image_dir, output_dir = args

    img_filename = json_file.stem + '.jpg'

    # 이미지 파일 찾기
    if class_id == 0:  # 정상
        img_src = image_dir / "정상" / img_filename
    else:  # NG
        img_src = image_dir / "NG" / img_filename

    if not img_src.exists():
        return ('fail', class_id)

    # YOLO 라벨 변환
    yolo_label = json_to_yolo(json_file, class_id)
    if yolo_label is None:
        return ('fail', class_id)

    # 이미지 복사
    img_dst = output_dir / 'images' / split / img_filename
    shutil.copy2(img_src, img_dst)

    # 라벨 저장
    label_dst = output_dir / 'labels' / split / (json_file.stem + '.txt')
    with open(label_dst, 'w') as f:
        f.write(yolo_label)

    return ('success', class_id)


def prepare_dataset(dataset_info, num_workers=None):
    """
    Train/Val 분할 및 데이터 변환 (CPU 멀티프로세싱)
    
    Args:
        dataset_info: collect_json_files()에서 반환된 데이터셋 정보
        num_workers: 사용할 워커 수 (None이면 자동 설정)
    
    Returns:
        처리 통계 딕셔너리
    """
    print("\n" + "=" * 70)
    print("🔄 데이터 변환 및 분할 진행 중... (CPU 멀티프로세싱)")
    print("=" * 70)

    # 시드 설정
    random.seed(RANDOM_SEED)

    # 통계 변수
    stats = {
        'train': {'ok': 0, 'ng': 0, 'success': 0, 'fail': 0},
        'val': {'ok': 0, 'ng': 0, 'success': 0, 'fail': 0}
    }

    # CPU 코어 수 확인
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # 최대 8개 프로세스
    print(f"💻 CPU 코어 수: {cpu_count()}개, 사용할 워커: {num_workers}개")

    # 각 카테고리별로 처리
    for category, info in dataset_info.items():
        print(f"\n처리 중: {category} ({info['count']}개)")

        json_files = info['files']
        class_id = info['class_id']

        # 랜덤 셔플 및 분할
        random.shuffle(json_files)
        split_idx = int(len(json_files) * (1 - VAL_RATIO))

        train_files = json_files[:split_idx]
        val_files = json_files[split_idx:]

        # Train 데이터 병렬 처리
        train_args = [(f, class_id, 'train', IMAGE_DIR, OUTPUT_DIR) for f in train_files]
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, train_args),
                total=len(train_args),
                desc=f"  Train {category}"
            ))

        for result, cid in results:
            if result == 'success':
                stats['train']['success'] += 1
                if cid == 0:
                    stats['train']['ok'] += 1
                else:
                    stats['train']['ng'] += 1
            else:
                stats['train']['fail'] += 1

        # Validation 데이터 병렬 처리
        val_args = [(f, class_id, 'val', IMAGE_DIR, OUTPUT_DIR) for f in val_files]
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, val_args),
                total=len(val_args),
                desc=f"  Val {category}"
            ))

        for result, cid in results:
            if result == 'success':
                stats['val']['success'] += 1
                if cid == 0:
                    stats['val']['ok'] += 1
                else:
                    stats['val']['ng'] += 1
            else:
                stats['val']['fail'] += 1

    # 통계 출력
    print("\n" + "=" * 70)
    print("✅ 데이터 변환 완료!")
    print("=" * 70)
    print(f"\n📊 Train 데이터셋:")
    print(f"   ✅ OK: {stats['train']['ok']}개")
    print(f"   ❌ NG: {stats['train']['ng']}개")
    print(f"   📈 총: {stats['train']['success']}개")
    print(f"   ⚠️  실패: {stats['train']['fail']}개")

    print(f"\n📊 Validation 데이터셋:")
    print(f"   ✅ OK: {stats['val']['ok']}개")
    print(f"   ❌ NG: {stats['val']['ng']}개")
    print(f"   📈 총: {stats['val']['success']}개")
    print(f"   ⚠️  실패: {stats['val']['fail']}개")

    print("\n" + "=" * 70)
    total_success = stats['train']['success'] + stats['val']['success']
    total_fail = stats['train']['fail'] + stats['val']['fail']
    print(f"🎯 전체 성공: {total_success}개 / 전체: {total_success + total_fail}개")
    print(f"⚡ CPU 멀티프로세싱으로 약 {num_workers}배 빠르게 처리되었습니다!")
    print("=" * 70)

    return stats


def main():
    """데이터셋 준비 메인 함수"""
    print("=" * 70)
    print("📦 딸기 OK/NG 데이터셋 준비 시작")
    print("=" * 70)
    
    # 1. JSON 파일 수집
    print("\n[1/4] JSON 파일 수집 중...")
    dataset_info, all_json_files = collect_json_files()
    
    # 2. 디렉토리 구조 생성
    print("\n[2/4] 디렉토리 구조 생성 중...")
    create_directory_structure()
    
    # 3. 데이터 변환 및 분할
    print("\n[3/4] 데이터 변환 및 분할 시작...")
    stats = prepare_dataset(dataset_info)
    
    # 4. YAML 설정 파일 생성
    print("\n[4/4] YAML 설정 파일 생성 중...")
    yaml_path = BASE_DIR / 'strawberry_ok_ng.yaml'
    create_yaml_config(OUTPUT_DIR, yaml_path, CLASS_NAMES)
    
    print("\n" + "=" * 70)
    print("✅ 데이터셋 준비 완료!")
    print("=" * 70)
    
    return stats, yaml_path


if __name__ == "__main__":
    main()

