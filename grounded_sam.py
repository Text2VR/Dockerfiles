#!/usr/bin/env python3
"""
간단한 Grounded SAM: transformers + SAM
Hugging Face transformers를 사용한 더 간단한 구현
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

# transformers 기반 GroundingDINO
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ transformers가 설치되지 않았습니다.")
    TRANSFORMERS_AVAILABLE = False

class SimpleGroundedSAM:
    def __init__(self, sam_checkpoint):
        self.sam_predictor = None
        self.object_detector = None
        
        # SAM 초기화
        if SAM_AVAILABLE and os.path.exists(sam_checkpoint):
            self._init_sam(sam_checkpoint)
        
        # Object Detection 파이프라인 초기화 (DETR 사용)
        if TRANSFORMERS_AVAILABLE:
            self._init_object_detector()
    
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
    
    def _init_object_detector(self):
        """Object Detection 모델 초기화"""
        try:
            print("🔧 객체 검출 모델 로딩 중...")
            # DETR을 사용한 객체 검출
            self.object_detector = pipeline(
                "object-detection", 
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ 객체 검출 모델 로딩 완료")
        except Exception as e:
            print(f"❌ 객체 검출 모델 로딩 실패: {e}")
    
    def detect_objects_with_detr(self, image_path, target_classes, confidence_threshold=0.5):
        """DETR로 객체 검출"""
        if not self.object_detector:
            print("❌ 객체 검출 모델이 초기화되지 않았습니다")
            return None
        
        print(f"🧠 DETR로 '{target_classes}' 검출 중...")
        
        try:
            # 이미지 로드
            image = Image.open(image_path)
            
            # DETR 예측
            predictions = self.object_detector(image)
            
            # 타겟 클래스 필터링
            filtered_results = []
            for pred in predictions:
                label = pred['label'].lower()
                score = pred['score']
                
                # 의자 관련 클래스들 체크
                chair_keywords = ['chair', 'couch', 'sofa', 'armchair', 'seat']
                if any(keyword in label for keyword in chair_keywords) and score >= confidence_threshold:
                    filtered_results.append(pred)
            
            print(f"📍 DETR 검출 결과: {len(filtered_results)}개 객체")
            for i, pred in enumerate(filtered_results):
                box = pred['box']
                print(f"  객체 {i+1}: {pred['label']} (신뢰도: {pred['score']:.3f})")
                print(f"    박스: [{box['xmin']:.1f}, {box['ymin']:.1f}, {box['xmax']:.1f}, {box['ymax']:.1f}]")
            
            return {
                "predictions": filtered_results,
                "image_size": image.size
            }
            
        except Exception as e:
            print(f"❌ DETR 검출 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def refine_with_sam(self, image_path, detr_results):
        """SAM으로 DETR 박스를 정밀한 마스크로 변환"""
        if not self.sam_predictor:
            print("❌ SAM이 초기화되지 않았습니다")
            return detr_results
        
        if not detr_results or len(detr_results["predictions"]) == 0:
            print("❌ DETR 결과가 없습니다")
            return None
        
        print("🎯 SAM으로 정밀 세그멘테이션 중...")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # SAM에 이미지 설정
        self.sam_predictor.set_image(image_rgb)
        
        refined_results = {"objects": []}
        
        for i, pred in enumerate(detr_results["predictions"]):
            print(f"🔍 객체 {i+1} ({pred['label']}) 정밀 분석 중...")
            
            try:
                # DETR 박스를 SAM 박스 프롬프트로 변환
                box = pred['box']
                input_box = np.array([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
                
                # SAM 예측
                masks, scores, logits = self.sam_predictor.predict(
                    box=input_box,
                    multimask_output=True
                )
                
                # 가장 좋은 마스크 선택
                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                best_score = scores[best_mask_idx]
                
                # 마스크에서 정제된 바운딩 박스 추출
                refined_bbox = self._mask_to_bbox(best_mask)
                
                refined_results["objects"].append({
                    "description": pred['label'],
                    "bbox": refined_bbox.tolist(),
                    "confidence": float(best_score),
                    "detr_confidence": float(pred['score']),
                    "mask_available": True,
                    "original_box": [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
                })
                
                print(f"✅ 객체 {i+1} 완료 (SAM 신뢰도: {best_score:.3f})")
                
                # 마스크 저장
                self._save_mask(best_mask, f"simple_grounded_mask_{i+1}_{pred['label'].replace(' ', '_')}.png")
                
            except Exception as e:
                print(f"❌ 객체 {i+1} SAM 처리 실패: {e}")
                # DETR 결과만 유지
                box = pred['box']
                refined_results["objects"].append({
                    "description": pred['label'],
                    "bbox": [box['xmin'], box['ymin'], box['xmax'], box['ymax']],
                    "confidence": float(pred['score']),
                    "detr_confidence": float(pred['score']),
                    "mask_available": False,
                    "original_box": [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
                })
        
        return refined_results
    
    def _mask_to_bbox(self, mask):
        """마스크에서 바운딩 박스 추출"""
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return np.array([0, 0, 1, 1])
        
        x1, x2 = int(x_indices.min()), int(x_indices.max())
        y1, y2 = int(y_indices.min()), int(y_indices.max())
        
        return np.array([x1, y1, x2, y2])
    
    def _save_mask(self, mask, filename):
        """마스크를 이미지로 저장"""
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(filename, mask_image)
        print(f"💾 마스크 저장: {filename}")
    
    def visualize_results(self, image_path, results, output_path="simple_grounded_result.png"):
        """결과 시각화"""
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for i, obj in enumerate(results["objects"]):
            color = colors[i % len(colors)]
            bbox = obj["bbox"]
            
            # 바운딩 박스 그리기
            line_width = 4 if obj.get("mask_available", False) else 2
            draw.rectangle(bbox, outline=color, width=line_width)
            
            # 라벨
            confidence = obj.get("confidence", 0.0)
            detr_conf = obj.get("detr_confidence", 0.0)
            mask_status = "SAM" if obj.get("mask_available", False) else "DETR"
            description = obj.get("description", "object")
            
            label = f"{description} D:{detr_conf:.2f} S:{confidence:.2f} [{mask_status}]"
            
            # 텍스트 배경
            text_y = max(bbox[1] - 35, 0)
            try:
                text_bbox = draw.textbbox((bbox[0], text_y), label)
                draw.rectangle(text_bbox, fill=color)
                draw.text((bbox[0], text_y), label, fill='white')
            except:
                draw.text((bbox[0], text_y), label, fill=color)
        
        img.save(output_path)
        print(f"✅ 시각화 저장: {output_path}")
        return output_path
    
    def detect_objects(self, image_path, target_classes="chair", confidence_threshold=0.5):
        """전체 파이프라인 실행"""
        print(f"🚀 Simple Grounded SAM 파이프라인 시작")
        print(f"📝 타겟 클래스: '{target_classes}'")
        
        # 1. DETR로 객체 검출
        detr_results = self.detect_objects_with_detr(
            image_path, target_classes, confidence_threshold
        )
        
        if not detr_results:
            print("❌ DETR 검출 실패")
            return None
        
        # 2. SAM으로 정밀한 세그멘테이션
        refined_results = self.refine_with_sam(image_path, detr_results)
        
        return refined_results

def main():
    print("🎯 Simple Grounded SAM 파이프라인 테스트")
    print("=" * 60)
    
    # SAM 체크포인트 경로
    sam_checkpoint = "sam_checkpoints/sam_vit_h_4b8939.pth"
    
    if not os.path.exists(sam_checkpoint):
        print(f"❌ SAM 체크포인트 없음: {sam_checkpoint}")
        return
    
    # 검출기 초기화
    detector = SimpleGroundedSAM(sam_checkpoint)
    
    # 테스트 실행
    image_path = "scenario_1_panorama.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일 없음: {image_path}")
        return
    
    # 의자 검출 테스트
    results = detector.detect_objects(
        image_path, 
        target_classes="chair", 
        confidence_threshold=0.3
    )
    
    if results:
        # 시각화
        output_name = "simple_grounded_sam_result.png"
        detector.visualize_results(image_path, results, output_name)
        
        # JSON 저장
        json_name = "simple_grounded_sam_result.json"
        with open(json_name, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 결과 파일:")
        print(f"  - 시각화: {output_name}")
        print(f"  - JSON: {json_name}")
        print(f"📊 검출된 객체: {len(results['objects'])}개")
        
        # 결과 요약
        print(f"\n📊 결과 요약:")
        for i, obj in enumerate(results["objects"]):
            bbox = obj["bbox"]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            print(f"  객체 {i+1}: {obj['description']} - {w:.0f}x{h:.0f} 픽셀")
            print(f"    DETR: {obj['detr_confidence']:.3f}, SAM: {obj['confidence']:.3f}")
    else:
        print("❌ 검출 실패")

if __name__ == "__main__":
    main()