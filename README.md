
GPT로 object 탐지 후, SAM으로 segmentation: gpt의 탐지 성능이 너무 좋지 않아서 pass

sam 단독도 성능이 좋지 않아서 pass

grounded + sam : 지금까지 시도해본 combi 중에서 가장 best
  - bounding box로 진행했을 때 희망이 보여서, 폴리곤 시도
  -> hgf_grounded_sam.py 코드가 gruonded를 hgf에서 불러서 쓰는것인줄 알았는데, 그게 아니라 다 못불러오고 대체로 하는 중이였음
    -> groundDINO 사용 연구 필요


panorama 생성
  - Dreamscene360의 파노라마는 text2PanoRun.py로 실행(text2pano_self_refine_pipeline.py로 생성)하는데, gpt 모델이 사라진 예전 모델이라, 현재 gpt-4o를 써야하는데 포멧이 맞지 않음 
    - text2pano_fixed.py로 연구중

  - 단순 stich_diffusion만 써서 만든 pano -> pano_generator.py -> panorama_output/panorama.png