# 🐳 Text  → 3D — Docker Collection

GPU-ready Docker environments for state-of-the-art 3D generation / reconstruction:

| Folder                  | Model                 | Task                                | CUDA | Framework           |
| ----------------------- | --------------------- | ----------------------------------- | ---- | ------------------- |
| `dreamscene_docker/`    | dreamscene            | Text → 3D Mesh (.ply)               | 11.8 | PyTorch cu118       |


---

# Quick Start
'''bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Text2VR/Dockerfiles.git
cd Dockerfiles

# DreamScene demo
cd dreamscene_docker

# Build the Docker image (CUDA 11.8 + cuDNN)
docker build -t dreamscene:cu118 .

# Run the container with GPU and mount output directory
docker run --gpus all -it --rm \
  -v $(pwd)/DreamScene:/workspace/DreamScene \
  -v $(pwd)/output:/workspace/output \
  dreamscene:cu118

# Once inside the container:
# 1. Navigate to DreamScene directory
# 2. Run training using your config:
#    CUDA_VISIBLE_DEVICES=0 python main.py --object --config configs/objects/sample.yaml
'''