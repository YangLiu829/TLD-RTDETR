# TLD-RTDETR

## Abstract：This repository contains the official implementation of TLD‑RTDETR, which is a lightweight real‑time detection transformer for tomato leaf disease recognition in complex agricultural scenarios.The main contributions are as follows:

- `PConvBlock`: A lightweight backbone module that reduces parameters and computational cost while maintaining feature extraction capability.  
- `CA-HSFPN`: A feature pyramid module that suppresses background noise and enhances multi-scale feature representation.  
- `AIFI-LPE`: An encoding module that enhances positional awareness and improves key feature identification.
# Environment Configuration
## Hardware Requirements
- Operating System: Windows 10  
- CPU: Intel(R) Core i7-12400  
- GPU: NVIDIA RTX 3090 (recommended)
## Software Dependencies
- Python >= 3.8  
- PyTorch == 2.0.0
  # Create an environment
conda create -n yolo python=3.10.14

# Activate the environment
conda activate RTDETR

# Install PyTorch (match your CUDA 12.1)
conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other core dependencies
pip install ultralytics opencv-python numpy
