import json
from PIL import Image, ImageDraw

# JSON 로드
with open('hybrid_chair_scenario_1_panorama.json', 'r') as f:
    data = json.load(f)

# 이미지 로드
img = Image.open('scenario_1_panorama.png')
draw = ImageDraw.Draw(img)

# GPT-4o 힌트만 그리기
colors = ['red', 'blue', 'green','purple', 'orange', 'yellow']
for i, obj in enumerate(data['objects']):
    hint = obj['original_hint']
    bbox = hint['bbox']
    center = hint['center_pixel']

    # 바운딩박스
    draw.rectangle(bbox, outline=colors[i], width=4)

    # 중심점
    x, y = center
    draw.ellipse([x-5, y-5, x+5, y+5], fill=colors[i])

    # 영어 라벨
    label = f"Chair {i+1}"
    draw.text((bbox[0], bbox[1] - 20), label, fill=colors[i])

img.save('gpt4o_hints_only.png')
print("Saved gpt4o_hints_only.png")