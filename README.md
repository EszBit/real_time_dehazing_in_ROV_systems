# Real-Time Image Enhancement in ROV Systems

This repository contains the implementation and optimization pipeline for training and deploying an underwater image enhancement model, developed as part of a master's thesis project. It builds upon the architecture introduced in [Fast Underwater Image Enhancement for Improved Visual Perception (RA-L 2020)](https://ieeexplore.ieee.org/document/9001231), adapting it for real-time inference on remotely operated vehicles (ROVs).

## 🔍 Project Overview

The goal of this project is to develop a robust and lightweight underwater dehazing model that performs well on real-world data from ROV-mounted cameras. While the original FUnIE-GAN model provides a strong foundation, additional training, fine-tuning, and optimization were necessary to adapt it for real-time, CPU-only operation in constrained environments.

Key features include:

- Support for training and fine-tuning on custom underwater datasets
- ONNX export and optimization (incl. INT8 and FP16 variants)
- FastAPI wrapper for lightweight model deployment from RUST
- Evaluation tools for testing original or optimized models

## 📁 Repository Structure
```txt
.
├── configs/
│   ├── train_model.yaml
│   └── train_target.yaml
├── nets/
│   ├── __init__.py
│   ├── commons.py
│   ├── funiegan.py
│   ├── pix2pix.py
│   └── ugan.py
├── utils/
│   ├── __init__.py
│   └── data_utils.py
├── data/  # NOT included, only for clarification of structure
│   ├── paired
│   ├── target_dataset
│   ├── test
│   └── output
│   	└──test_results
├── models/
├── convert_int8.py
├── convert_onnx.py
├── optimize_onnx_model.py
├── finetune_funiegan.py
├── test_model.py
├── test_onnx_model.py
└── test_ort_model.py
```

## 🔧 Usage

### 1. Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training (can be done on CPU or GPU (MPS))

```bash
python train_funiegan.py
```

(Optional) Fine-tune a pretrained model (can be done on CPU or GPU (MPS))

```bash
python finetune_funiegan.py
```

(Optional) ### Convert your PyTorch model into your desired format

```bash
python convert_onnx.py
python convert_uint8.py
python convert_fp16
```

(Optional )### Optimize your .onnx model

```bash
python optimize_onnx_model.py
```

### 3. Testing (can be done on CPU or GPU (MPS))

```bash
python test_model.py
python test_onnx_model.py
python test_ort_model.py
```

Edit data paths in each script as needed.

(Optional) ### Deployment with RUST (this was for testing purposes)

```bash
python fastAPI.py
```

## Reference

> **Fast Underwater Image Enhancement for Improved Visual Perception**  
> Md Jahidul Islam, Youya Xia, and Junaed Sattar  
> *IEEE Robotics and Automation Letters (RA-L), 2020*  
> [IEEE Link](https://ieeexplore.ieee.org/document/9001231)

```bibtex
@article{islam2019fast,
  title={Fast Underwater Image Enhancement for Improved Visual Perception},
  author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3227--3234},
  year={2020},
  publisher={IEEE}
}
```

## 📎 Notes

- The dataset used for training and evaluation is not included.
- Optimized for real-time CPU-only inference under 500 ms.









