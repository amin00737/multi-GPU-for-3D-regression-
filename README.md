🚀 multi‑GPU‑for‑3D‑regression

Distributed training utilities to accelerate 3D regression models across multiple GPUs.

This repository contains scripts to configure and run multi‑GPU training for 3D regression tasks using PyTorch’s distributed training paradigms (e.g., DDP) to scale performance on systems with multiple GPUs.


🚀 Features

📈 Distributed training across multiple GPUs

⚙️ Compatible with PyTorch’s torch.distributed APIs

🧪 Scales regression model training using data parallelism

🔧 Minimal code changes required for single‑GPU to multi‑GPU switch

🧩 Requirements

Make sure you have the following installed:

Python 3.8+
PyTorch (compatible with your CUDA version)
NCCL & CUDA drivers configured
torchvision (if needed depending on models)

pip install torch torchvision


📌 Notes

This project uses data parallelism to distribute training across GPUs.

You can extend this to multi‑node training using PyTorch’s distributed launch utilities.