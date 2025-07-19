#!/usr/bin/env python3
"""
진짜 GroundingDINO + SAM 파이프라인
원본 GroundingDINO 설치 후 사용
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
import json
import sys

# GroundingDINO 경로 추가
sys.path.append('GroundingDINO')

# SAM 관련 임포트
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("⚠️ SAM이 설치되지 않았습니다.")
    SAM_AVAILABLE = False

# GroundingDINO 관련 임포트 (원본)
try:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
    from groundingdino.util.inference import annotate, load_image, predict
    from groundingdino.util import box_ops
    GROUNDING_DINO_AVAILABLE = True
    print("✅ GroundingDINO import 성공")
except ImportError as e:
    print(f"⚠️ GroundingDINO 임포트 실패: {e}")
    GROUNDING_DINO_AVAILABLE = False

class ProperGroundedSAM:
    def __init__(self, sam_checkpoint, grounding_dino_config, grounding_dino_checkpoint):
        self.sam_predictor = None
        self.grounding_dino_model = None
        
        # SAM 초기화
        if SAM_AVAILABLE and os.path.exists(sam_checkpoint):
            self._init_sam(sam_checkpoint)
        
        # GroundingDINO 초기화
        if GROUNDING_DINO_AVAILABLE and os.path.exists(grounding_dino_config) and os.path.exists(grounding_dino_checkpoint):
            self._init_grounding_dino(grounding_dino_config, grounding_dino_checkpoint)
    
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
    
    def _init_grounding_dino(self, config_path, checkpoint_path):
        """GroundingDINO 모델 초기화"""
        try:
            print("🔧 GroundingDINO 모델 로딩 중...")
            print(f"  설정 파일: {config_path}")
            print(f"  체크포인트: {checkpoint_path}")
            
            # 설정 로드
            args = SLConfig.fromfile(config_path)
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 모델 빌드
            model = build_model(args)
            
            # 체크포인트 로드
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            print(f"  모델 로딩 결과: {load_res}")
            
            # 평가 모드
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            
            self.grounding_dino_model = model
            print("✅ GroundingDINO 로딩 완료")
            
        except Exception as e:
            print(f"❌ GroundingDINO 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def detect_with_grounding_dino(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """GroundingDINO로 텍스트 기반 객체 검출"""
        if not self.grounding_dino_model:
            print("❌ GroundingDINO가 초기화되지 않았습니다")
            return None
        
        print(f"🧠 GroundingDINO로 '{text_prompt}' 검출 중...")
        
        try:
            # 이미지 로드 및 전처리
            image_source, image = load_image(image_path)
            
            # 예측 수행
            boxes, logits, phrases = predict(
                model=self.grounding_dino_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # 박스 좌표 변환 (normalized → pixel)
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            boxes = box_ops.box_cxcywh_to_xyxy(boxes).cpu().numpy()
            
            print(f"📍 GroundingDINO 검출 결과: {len(boxes)}개 객체")
            for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                print(f"  객체 {i+1}: {phrase} (신뢰도: {logit:.3f})")
                print(f"    박스: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
            return {
                "boxes": boxes,
                "logits": logits.cpu().numpy(),
                "phrases": phrases,
                "image_shape": (h, w)
            }
            
        except Exception as e:
            print(f"❌ GroundingDINO 검출 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def refine_with_sam(self, image_path, grounding_results):
        """SAM으로 GroundingDINO 박스를 정밀한 마스크로 변환"""
        if not self.sam_predictor:
            print("❌ SAM이 초기화되지 않았습니다")
            return grounding_results
        
        if not grounding_results or len(grounding_results["boxes"]) == 0:
            print("❌ GroundingDINO 결과가 없습니다")
            return None
        
        print("🎯 SAM으로 정밀 세그멘테이션 중...")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # SAM에 이미지 설정
        self.sam_predictor.set_image(image_rgb)
        
        refined_results = {"objects": []}
        
        for i, (box, logit, phrase) in enumerate(zip(
            grounding_results["boxes"], 
            grounding_results["logits"], 
            grounding_results["phrases"]
        )):
            print(f"🔍 객체 {i+1} ({phrase}) 정밀 분석 중...")
            
            try:
                # GroundingDINO 박스를 SAM 박스 프롬프트로 사용
                input_box = np.array([box])
                
                # 박스 중심점도 추가 (더 안정적)
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                input_point = np.array([[center_x, center_y]])
                input_label = np.array([1])
                
                # SAM 예측 (박스 + 포인트 프롬프트)
                masks, scores, logits_sam = self.sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=True
                )
                
                # 가장 좋은 마스크 선택
                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                best_score = scores[best_mask_idx]
                
                # 마스크에서 정제된 바운딩 박스 추출
                refined_bbox = self._mask_to_bbox(best_mask)
                
                # 폴리곤 추출
                polygon = self._mask_to_polygon(best_mask)
                
                refined_results["objects"].append({
                    "description": phrase,
                    "bbox": refined_bbox.tolist(),
                    "polygon": polygon,
                    "confidence": float(best_score),
                    "grounding_confidence": float(logit),
                    "mask_available": True,
                    "original_box": box.tolist()
                })
                
                print(f"✅ 객체 {i+1} 완료 (GroundingDINO: {logit:.3f}, SAM: {best_score:.3f})")
                
                # 마스크 저장
                self._save_mask(best_mask, f"proper_grounded_mask_{i+1}_{phrase.replace(' ', '_')}.png")
                
            except Exception as e:
                print(f"❌ 객체 {i+1} SAM 처리 실패: {e}")
                # GroundingDINO 결과만 유지
                refined_results["objects"].append({
                    "description": phrase,
                    "bbox": box.tolist(),
                    "polygon": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                    "confidence": float(logit),
                    "grounding_confidence": float(logit),
                    "mask_available": False,
                    "original_box": box.tolist()
                })
        
        return refined_results
    
    def _mask_to_bbox(self, mask):
        """마스크에서 바운딩 박스 추출"""
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0:
            return np.array([0, 0, 1, 1])
        
        x1, x2 = int(x_indices.min()), int(x_indices.max())
        y1, y2 = int(y_indices.min()), int(y_indices.max())
        
        return np.array([x1, y1, x2, y2])
    
    def _mask_to_polygon(self, mask):
        """마스크에서 폴리곤 추출"""
        try:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 윤곽선 선택
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 폴리곤 근사
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
                polygon = polygon.reshape(-1, 2).tolist()
                
                return polygon
            else:
                # 기본 사각형
                h, w = mask.shape
                return [[0, 0], [w, 0], [w, h], [0, h]]
        except Exception as e:
            print(f"폴리곤 추출 실패: {e}")
            h, w = mask.shape
            return [[0, 0], [w, 0], [w, h], [0, h]]
    
    def _save_mask(self, mask, filename):
        """마스크를 이미지로 저장"""
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(filename, mask_image)
        print(f"💾 마스크 저장: {filename}")
    
    def visualize_results(self, image_path, results, output_path="proper_grounded_result.png"):
        """결과 시각화 (바운딩박스 + 폴리곤)"""
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        
        for i, obj in enumerate(results["objects"]):
            color = colors[i % len(colors)]
            bbox = obj["bbox"]
            polygon = obj["polygon"]
            
            # 바운딩 박스 그리기
            line_width = 4 if obj.get("mask_available", False) else 2
            draw.rectangle(bbox, outline=color, width=line_width)
            
            # 폴리곤 그리기
            if len(polygon) > 2:
                polygon_coords = [tuple(point) for point in polygon]
                draw.polygon(polygon_coords, outline=color, width=2)
            
            # 라벨
            grounding_conf = obj.get("grounding_confidence", 0.0)
            sam_conf = obj.get("confidence", 0.0)
            mask_status = "SAM" if obj.get("mask_available", False) else "DINO"
            description = obj.get("description", "object")
            
            label = f"{description} G:{grounding_conf:.2f} S:{sam_conf:.2f} [{mask_status}]"
            
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
    
    def detect_objects(self, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """전체 파이프라인 실행"""
        print(f"🚀 Proper Grounded SAM 파이프라인 시작")
        print(f"📝 텍스트 프롬프트: '{text_prompt}'")
        
        # 1. GroundingDINO로 텍스트 기반 검출
        grounding_results = self.detect_with_grounding_dino(
            image_path, text_prompt, box_threshold, text_threshold
        )
        
        if not grounding_results:
            print("❌ GroundingDINO 검출 실패")
            return None
        
        # 2. SAM으로 정밀한 세그멘테이션
        refined_results = self.refine_with_sam(image_path, grounding_results)
        
        return refined_results

def main():
    print("🎯 Proper Grounded SAM 파이프라인 테스트")
    print("=" * 60)
    
    # 모델 경로들
    sam_checkpoint = "sam_checkpoints/sam_vit_h_4b8939.pth"
    grounding_dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint = "GroundingDINO/groundingdino_swint_ogc.pth"
    
    # 파일 존재 확인
    files_to_check = [
        ("SAM checkpoint", sam_checkpoint),
        ("GroundingDINO config", grounding_dino_config),
        ("GroundingDINO checkpoint", grounding_dino_checkpoint)
    ]
    
    for name, path in files_to_check:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name} 없음: {path}")
            if "GroundingDINO" in path:
                print("💡 먼저 GroundingDINO를 설치해주세요:")
                print("   git clone https://github.com/IDEA-Research/GroundingDINO.git")
                print("   cd GroundingDINO && pip install -e .")
                return
    
    # 검출기 초기화
    detector = ProperGroundedSAM(sam_checkpoint, grounding_dino_config, grounding_dino_checkpoint)
    
    # 테스트 실행
    image_path = "scenario_1_panorama.png"
    
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일 없음: {image_path}")
        return
    
    # 텍스트 프롬프트들
    test_prompts = [
        "brown leather chair",
        "chair . sofa . armchair",
        "furniture",
        "chair"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"🎯 테스트: '{prompt}'")
        
        results = detector.detect_objects(image_path, prompt, box_threshold=0.25, text_threshold=0.20)
        
        if results and results["objects"]:
            # 시각화
            output_name = f"proper_grounded_{prompt.replace(' ', '_').replace('.', '')}.png"
            detector.visualize_results(image_path, results, output_name)
            
            # JSON 저장
            json_name = f"proper_grounded_{prompt.replace(' ', '_').replace('.', '')}.json"
            with open(json_name, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"📁 결과 파일:")
            print(f"  - 시각화: {output_name}")
            print(f"  - JSON: {json_name}")
            print(f"📊 검출된 객체: {len(results['objects'])}개")
            
            # 성공하면 첫 번째 프롬프트로 중단
            break
        else:
            print("❌ 검출 실패")

if __name__ == "__main__":
    main()