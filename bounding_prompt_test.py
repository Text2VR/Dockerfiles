#!/usr/bin/env python3
"""
SAM 세그멘테이션 문제 분석 및 디버깅
GPT-4o 힌트와 SAM 결과 비교 분석
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2

def analyze_detection_results(gpt4o_json_path, sam_json_path, image_path):
    """GPT-4o와 SAM 결과 비교 분석"""
    print("🔍 SAM 결과 문제 분석 시작...")
    
    # JSON 파일 로드
    if not os.path.exists(gpt4o_json_path):
        print(f"❌ GPT-4o 결과 파일 없음: {gpt4o_json_path}")
        return
        
    if not os.path.exists(sam_json_path):
        print(f"❌ SAM 결과 파일 없음: {sam_json_path}")
        return
    
    with open(gpt4o_json_path, 'r') as f:
        gpt4o_data = json.load(f)
    
    with open(sam_json_path, 'r') as f:
        sam_data = json.load(f)
    
    # 이미지 정보
    img = Image.open(image_path)
    width, height = img.size
    print(f"📐 이미지 크기: {width}x{height}")
    
    # GPT-4o 결과 분석
    print("\n📍 GPT-4o 힌트 분석:")
    gpt4o_objects = gpt4o_data.get('objects_found', [])
    for i, obj in enumerate(gpt4o_objects):
        bbox = obj.get('bbox', [])
        if len(bbox) >= 4:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = w * h
            print(f"  객체 {i+1}: {obj.get('object_name', 'Unknown')}")
            print(f"    바운딩박스: {bbox}")
            print(f"    크기: {w}x{h} (면적: {area} 픽셀)")
            print(f"    신뢰도: {obj.get('confidence', 0):.3f}")
    
    # SAM 결과 분석
    print("\n🎯 SAM 결과 분석:")
    sam_objects = sam_data.get('objects', [])
    for i, obj in enumerate(sam_objects):
        bbox = obj.get('bbox', [])
        if len(bbox) >= 4:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = w * h
            print(f"  객체 {i+1}: {obj.get('description', 'Unknown')}")
            print(f"    바운딩박스: {bbox}")
            print(f"    크기: {w}x{h} (면적: {area} 픽셀)")
            print(f"    신뢰도: {obj.get('confidence', 0):.3f}")
            print(f"    마스크 사용 가능: {obj.get('mask_available', False)}")
            
            # 원본 힌트와 비교
            if 'original_hint' in obj:
                hint = obj['original_hint']
                hint_bbox = hint.get('bbox', [])
                if len(hint_bbox) >= 4:
                    hint_w = hint_bbox[2] - hint_bbox[0]
                    hint_h = hint_bbox[3] - hint_bbox[1]
                    hint_area = hint_w * hint_h
                    print(f"    원본 힌트 크기: {hint_w}x{hint_h} (면적: {hint_area} 픽셀)")
                    print(f"    크기 변화: {area/hint_area:.3f}배")

def create_comparison_visualization(gpt4o_json_path, sam_json_path, image_path, output_path="comparison.png"):
    """GPT-4o vs SAM 결과 비교 시각화"""
    print("\n🖼️ 비교 시각화 생성...")
    
    if not os.path.exists(gpt4o_json_path) or not os.path.exists(sam_json_path):
        print("❌ 필요한 JSON 파일이 없습니다")
        return
    
    with open(gpt4o_json_path, 'r') as f:
        gpt4o_data = json.load(f)
    
    with open(sam_json_path, 'r') as f:
        sam_data = json.load(f)
    
    # 원본 이미지 로드
    img = Image.open(image_path)
    
    # 좌우 비교 이미지 생성
    comparison_width = img.width * 2
    comparison_img = Image.new('RGB', (comparison_width, img.height), 'white')
    
    # 좌측: GPT-4o 결과
    left_img = img.copy()
    draw_left = ImageDraw.Draw(left_img)
    
    # 우측: SAM 결과  
    right_img = img.copy()
    draw_right = ImageDraw.Draw(right_img)
    
    # GPT-4o 바운딩박스 그리기 (빨간색)
    gpt4o_objects = gpt4o_data.get('objects_found', [])
    for i, obj in enumerate(gpt4o_objects):
        bbox = obj.get('bbox', [])
        if len(bbox) >= 4:
            draw_left.rectangle(bbox, outline='red', width=3)
            label = f"GPT-4o: {obj.get('object_name', 'Object')} ({obj.get('confidence', 0):.2f})"
            draw_left.text((bbox[0], bbox[1] - 20), label, fill='red')
    
    # SAM 바운딩박스 그리기 (파란색)
    sam_objects = sam_data.get('objects', [])
    for i, obj in enumerate(sam_objects):
        bbox = obj.get('bbox', [])
        if len(bbox) >= 4:
            line_width = 4 if obj.get('mask_available', False) else 2
            draw_right.rectangle(bbox, outline='blue', width=line_width)
            label = f"SAM: {obj.get('description', 'Object')} ({obj.get('confidence', 0):.2f})"
            draw_right.text((bbox[0], bbox[1] - 20), label, fill='blue')
            
            # 원본 힌트도 같이 표시 (회색, 점선 효과)
            if 'original_hint' in obj:
                hint_bbox = obj['original_hint'].get('bbox', [])
                if len(hint_bbox) >= 4:
                    # 점선 효과를 위해 여러 선분으로 그리기
                    for x in range(hint_bbox[0], hint_bbox[2], 10):
                        draw_right.line([(x, hint_bbox[1]), (min(x+5, hint_bbox[2]), hint_bbox[1])], fill='gray', width=1)
                        draw_right.line([(x, hint_bbox[3]), (min(x+5, hint_bbox[2]), hint_bbox[3])], fill='gray', width=1)
                    for y in range(hint_bbox[1], hint_bbox[3], 10):
                        draw_right.line([(hint_bbox[0], y), (hint_bbox[0], min(y+5, hint_bbox[3]))], fill='gray', width=1)
                        draw_right.line([(hint_bbox[2], y), (hint_bbox[2], min(y+5, hint_bbox[3]))], fill='gray', width=1)
    
    # 라벨 추가
    draw_left.text((10, 10), "GPT-4o Hints", fill='red', font_size=20)
    draw_right.text((10, 10), "SAM Results (with GPT hints in gray)", fill='blue', font_size=20)
    
    # 합성
    comparison_img.paste(left_img, (0, 0))
    comparison_img.paste(right_img, (img.width, 0))
    
    # 저장
    comparison_img.save(output_path)
    print(f"✅ 비교 이미지 저장: {output_path}")

def suggest_sam_improvements():
    """SAM 성능 개선 제안"""
    print("\n💡 SAM 성능 개선 제안:")
    print("1. 박스 프롬프트 추가:")
    print("   - 현재: 포인트 프롬프트만 사용")
    print("   - 개선: GPT-4o 바운딩박스를 박스 프롬프트로 직접 사용")
    
    print("\n2. 멀티스케일 세그멘테이션:")
    print("   - 현재: 단일 포인트로 예측")
    print("   - 개선: 바운딩박스 내 여러 포인트 사용")
    
    print("\n3. 파노라마 특화 처리:")
    print("   - 현재: 일반 이미지로 처리")
    print("   - 개선: 등장방형 투영 특성 고려")
    
    print("\n4. 후처리 필터링:")
    print("   - 현재: SAM 원본 결과 사용")
    print("   - 개선: 크기/위치 기반 필터링")

def main():
    print("🔧 SAM 문제 분석 도구")
    print("=" * 50)
    
    # 파일 경로 설정 (실제 파일명에 맞게 수정 필요)
    image_paths = [
        "scenario_1_panorama.png",
        "hybrid_chair_scenario_1_panorama.png"
    ]
    
    gpt4o_json_paths = [
        "test_output/detection_result.json",
        "quick_test_results/detection_result.json"
    ]
    
    sam_json_paths = [
        "hybrid_chair_scenario_1.json",
        "hybrid_chair_scenario_2.json"
    ]
    
    # 실제 존재하는 파일들로 분석
    found_files = False
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            print(f"\n📁 이미지 파일 발견: {img_path}")
            
            # 관련 JSON 파일들 찾기
            for gpt_json in gpt4o_json_paths:
                for sam_json in sam_json_paths:
                    if os.path.exists(gpt_json) and os.path.exists(sam_json):
                        print(f"📊 분석 중: {gpt_json} vs {sam_json}")
                        analyze_detection_results(gpt_json, sam_json, img_path)
                        create_comparison_visualization(gpt_json, sam_json, img_path, 
                                                     f"comparison_{os.path.basename(img_path)}")
                        found_files = True
                        break
                if found_files:
                    break
            
            if found_files:
                break
    
    if not found_files:
        print("❌ 분석할 파일들을 찾을 수 없습니다.")
        print("다음 파일들을 확인하세요:")
        print("- 파노라마 이미지 파일")
        print("- GPT-4o 검출 결과 JSON")
        print("- SAM 검출 결과 JSON")
    
    # 개선 제안
    suggest_sam_improvements()

if __name__ == "__main__":
    main()