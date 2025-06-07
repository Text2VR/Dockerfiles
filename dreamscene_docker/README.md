# 🚀 DreamScene Docker Setup

A GPU-ready Docker environment for **DreamScene**, a compositional 3D scene generation framework based on object-centric Gaussian Splatting.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Build the Docker Image](#build-the-docker-image)
- [Run the Docker Container](#run-the-docker-container)
- [Run DreamScene](#run-dreamscene)
- [Output Directory](#output-directory)
- [Optional Extensions](#optional-extensions)
- [License](#license)

---

## 🔍 Overview

This Docker setup provides everything needed to run DreamScene with GPU acceleration. It includes:

- **CUDA 11.8 + cuDNN 8**
- **Python 3.10**
- **PyTorch 2.2.0 (cu118)**
- Dependencies for:
  - **DreamScene**
  - **Cap3D**
  - **Point-E**
  - **Differentiable Gaussian Splatting**

---

## 📦 Build the Docker Image

```bash
cd dreamscene_docker
docker build -t dreamscene:cu118 .

🧠 Run the Docker Container
bash
복사
편집
docker run --gpus all -it --rm \
  -v $(pwd)/DreamScene:/workspace/DreamScene \
  -v $(pwd)/output:/workspace/output \
  dreamscene:cu118
  Option	Description
--gpus all	Allocates all available GPUs
-v $(pwd)/DreamScene:/workspace/DreamScene	Mounts the DreamScene repository into the container
-v $(pwd)/output:/workspace/output	Saves output results to your local output/ folder
