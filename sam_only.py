#!/usr/bin/env python3
"""
SAM 단독 세그멘테이션 테스트
사용자가 지정한 포인트에서 SAM으로 바로 세그멘테이션
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import json

# SAM 관련 임포트
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("⚠️ SAM이 설치되지 않았습니다.")
    SAM_AVAILABLE = False

class SAMOnlyDetector:
    def __init__(self, sam_checkpoint):
        self.sam_predictor = None
        if SAM_AVAILABLE and os.path.exists(sam_checkpoint):
            self._init_sam(sam_checkpoint)
    
    def _init_sam(self, checkpoint_path):
        """SAM 모델 초기화"""
        try:
            print("🔧 SAM 모델 로딩 중...")
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
            sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
            self.sam_predictor = SamPredictor(sam)
            print(f"✅ SAM 로딩 완료 (GPU: {torch.cuda.is_available()})")
        except Exception as e:
            print(f"❌ SAM 로딩 실패: {e}")
    
    def segment_at_points(self, image_path, points, output_prefix="sam_result"):
        """지정된 포인트들에서 SAM 세그멘테이션"""
        if not self.sam_predictor:
            print("❌ SAM이 초기화되지 않았습니다")
            return None
        
        print(f"🎯 {len(points)}개 포인트에서 SAM 세그멘테이션 시작")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # SAM에 이미지 설정
        self.sam_predictor.set_image(image_rgb)
        
        results = {"objects": []}
        
        for i, point in enumerate(points):
            print(f"🔍 포인트 {i+1}: {point} 세그멘테이션 중...")
            
            # SAM 입력 준비
            input_point = np.array([point])
            input_label = np.array([1])  # 전경
            
            try:
                # SAM 예측
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
                
                # 가장 좋은 마스크 선택
                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                best_score = scores[best_mask_idx]
                
                # 마스크에서 바운딩 박스 추출
                bbox = self._mask_to_bbox(best_mask)
                
                results["objects"].append({
                    "point": point,
                    "bbox": bbox,
                    "confidence": float(best_score),
                    "mask_available": True
                })
                
                print(f"✅ 포인트 {i+1} 완료 (신뢰도: {best_score:.3f})")
                print(f"   바운딩박스: {bbox}")
                print(f"   크기: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
                
                # 개별 마스크 저장
                self._save_mask(best_mask, f"{output_prefix}_mask_{i+1}.png")
                
            except Exception as e:
                print(f"❌ 포인트 {i+1} 처리 실패: {e}")
        
        return results
    
    def _mask_to_bbox(self, mask):
        """마스크에서 바운딩 박스 추출"""
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return [0, 0, 1, 1]
        
        x1, x2 = int(x_indices.min()), int(x_indices.max())
        y1, y2 = int(y_indices.min()), int(y_indices.max())
        
        return [x1, y1, x2, y2]
    
    def _save_mask(self, mask, filename):
        """마스크를 이미지로 저장"""
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(filename, mask_image)
        print(f"💾 마스크 저장: {filename}")
    
    def visualize_results(self, image_path, results, points, output_path="sam_visualization.png"):
        """결과 시각화"""
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        
        for i, (obj, point) in enumerate(zip(results["objects"], points)):
            color = colors[i % len(colors)]
            bbox = obj["bbox"]
            
            # 바운딩 박스 그리기
            draw.rectangle(bbox, outline=color, width=4)
            
            # 클릭 포인트 표시
            x, y = point
            draw.ellipse([x-8, y-8, x+8, y+8], fill=color, outline='white', width=2)
            
            # 라벨
            confidence = obj.get("confidence", 0.0)
            label = f"SAM_{i+1} ({confidence:.3f})"
            
            # 텍스트
            text_y = max(bbox[1] - 25, 0)
            draw.text((bbox[0], text_y), label, fill=color)
        
        img.save(output_path)
        print(f"✅ 시각화 저장: {output_path}")
        return output_path

def main():
    print("🎯 SAM 단독 세그멘테이션 테스트")
    print("=" * 50)
    
    # SAM 체크포인트 경로
    sam_checkpoint = "sam_checkpoints/sam_vit_h_4b8939.pth"
    
    if not SAM_AVAILABLE:
        print("❌ SAM이 설치되지 않았습니다")
        return
    
    if not os.path.exists(sam_checkpoint):
        print(f"❌ SAM 체크포인트가 없습니다: {sam_checkpoint}")
        return
    
    # SAM 검출기 초기화
    detector = SAMOnlyDetector(sam_checkpoint)
    
    # 테스트 이미지
    image_path = "scenario_1_panorama.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일 없음: {image_path}")
        return
    
    # 이미지 정보 출력
    img = Image.open(image_path)
    width, height = img.size
    print(f"📐 이미지 크기: {width}x{height}")
    
    # 실제 의자 위치에 맞춘 포인트들 (파노라마 하단 가장자리의 갈색 가구들)
    test_points = [
        [100, 480],   # 맨 좌측 소파 (갈색)
        [350, 470],   # 좌측 의자
        [600, 460],   # 좌측 중앙 의자
        [1000, 470],  # 우측 중앙 의자
        [1350, 480],  # 우측 소파
        [1650, 470],  # 맨 우측 의자
        [1900, 480],  # 극우측 소파
    ]
    
    print(f"📍 테스트 포인트들: {test_points}")
    
    # SAM 세그멘테이션 실행
    results = detector.segment_at_points(image_path, test_points, "sam_chair")
    
    if results:
        # 시각화
        vis_path = detector.visualize_results(image_path, results, test_points, "sam_only_result.png")
        
        # JSON 저장
        with open("sam_only_result.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 결과 파일:")
        print(f"  - 시각화: sam_only_result.png")
        print(f"  - JSON: sam_only_result.json")
        print(f"  - 개별 마스크: sam_chair_mask_*.png")
        
        # 결과 요약
        print(f"\n📊 결과 요약:")
        for i, obj in enumerate(results["objects"]):
            bbox = obj["bbox"]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            print(f"  객체 {i+1}: {w}x{h} 픽셀 (신뢰도: {obj['confidence']:.3f})")
    else:
        print("❌ 세그멘테이션 실패")

if __name__ == "__main__":
    main()