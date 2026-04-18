# 🌵 Desert Environment Semantic Segmentation

A deep learning pipeline for semantic segmentation of desert/offroad environments using **DeepLabV3+** with a ResNet50 backbone.

---

## 🎯 Demo

> 🤗 **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Aadarshhhhhhhh/desert-segmentation)

Upload any desert environment image and get instant semantic segmentation with class distribution analysis.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Training](#training)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## 🔍 Overview

This project tackles the challenge of understanding complex desert and offroad terrain through pixel-level semantic segmentation. The model can distinguish between 10 distinct environment classes including vegetation, terrain features, and sky — enabling applications in autonomous vehicles, robotics, and environmental monitoring.

---

## 🏷️ Classes

| ID | Label | Class Name |
|----|-------|------------|
| 0 | 100 | 🌳 Trees |
| 1 | 200 | 🌿 Lush Bushes |
| 2 | 300 | 🌾 Dry Grass |
| 3 | 500 | 🪴 Dry Bushes |
| 4 | 550 | 🗂️ Ground Clutter |
| 5 | 600 | 🌸 Flowers |
| 6 | 700 | 🪵 Logs |
| 7 | 800 | 🪨 Rocks |
| 8 | 7100 | 🏜️ Landscape |
| 9 | 10000 | ☁️ Sky |

---

## 📂 Dataset

- **Name:** Offroad Segmentation Training Dataset
- **Format:** RGB images + Grayscale segmentation masks
- **Splits:** Train / Validation
- **Structure:**
```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/     # RGB input images
│   └── Segmentation/     # Grayscale masks (raw class IDs)
└── val/
    ├── Color_Images/
    └── Segmentation/
```

---

## 🏗️ Model Architecture

| Component | Detail |
|-----------|--------|
| **Model** | DeepLabV3+ |
| **Backbone** | ResNet50 |
| **Pretrained** | ImageNet |
| **Input Size** | 512 × 512 |
| **Output** | 10-class segmentation map |
| **Loss** | CrossEntropy + Dice (combined) |
| **Optimizer** | AdamW (lr=3e-4, weight_decay=1e-4) |
| **Scheduler** | Cosine Annealing |
| **Trained On** | NVIDIA T4 GPU (Google Colab) |

### Class Weights
Dominant classes (Sky, Landscape, Dry Grass) were down-weighted and rare classes (Flowers, Logs) were up-weighted to handle class imbalance:

```python
CLASS_WEIGHTS = [1.5, 2.0, 0.5, 2.0, 2.5, 3.0, 3.0, 2.5, 0.4, 0.3]
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Pixel Accuracy | - |
| Mean Accuracy | - |
| Mean IoU (mIoU) | - |

> Training curves and per-class IoU charts are saved in `/content/outputs/` during training.

---

## 📁 Project Structure

```
desert-segmentation/
│
├── cell1_setup.py              # Environment setup & dependencies
├── cell2_dataset.py            # Dataset class & dataloaders
├── cell3_train.py              # DeepLabV3+ training loop
├── cell4_inference.py          # Single/batch image inference
├── cell5_evaluate.py           # Accuracy metrics & confusion matrix
│
├── app.py                      # Gradio web app (HF Spaces)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/aadarshpuniya01-boop/desert_segmentation-model.git
cd desert-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset
Place your dataset in the following structure:
```
/data/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
└── val/
    ├── Color_Images/
    └── Segmentation/
```

### 4. Run inference on a single image
```python
from cell4_inference import load_model, test_single

model = load_model("path/to/best_model.pth")
test_single(model, "path/to/your/image.jpg")
```

---

## 🏋️ Training

### Google Colab (Recommended)
1. Open `cell1_setup.py` → run as Cell 1
2. Open `cell2_dataset.py` → run as Cell 2
3. Open `cell3_train.py` → run as Cell 3

### Local Machine
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Train
python cell3_train.py
```

> ⚠️ Recommended batch size: `8` for T4/A100, `4` for RTX 3050/3060

---

## 🚀 Deployment

The model is deployed as a **Gradio** web app on **Hugging Face Spaces**.

To redeploy:
```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj = "app.py",
    path_in_repo    = "app.py",
    repo_id         = "Aadarshhhhhhhh"/desert-segmentation",
    repo_type       = "space",
    token           = "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",
)
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-4.x-purple)

- **PyTorch** — Deep learning framework
- **segmentation-models-pytorch** — DeepLabV3+ implementation
- **Albumentations** — Image augmentation
- **Gradio** — Web app interface
- **Hugging Face Spaces** — Model deployment
- **Google Colab** — Training environment (T4 GPU)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)
- [Hugging Face](https://huggingface.co/)
