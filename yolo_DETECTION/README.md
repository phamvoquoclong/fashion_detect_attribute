# 🧥 YOLOv11 Fashion Object Detection

This directory contains all resources related to **training and evaluating the object detection model** for the Fashion AI pipeline.

The detection task is built using **YOLOv11 (Ultralytics)** and serves as the **core backbone** for:
- SAHI sliced inference
- Attribute recognition via embeddings
- Streamlit demo visualization

---

## 📌 Overview

- **Task**: Multi-class fashion object detection  
- **Model**: YOLOv11s
  <img width="1438" height="867" alt="image" src="https://github.com/user-attachments/assets/10436701-0b66-44f4-bb9a-4690136bad96" />

- **Framework**: Ultralytics YOLO  
- **Number of classes**: 46  
- **Annotation format**: YOLO (normalized bounding boxes)

The focus of this module is to provide a **reliable detector** that balances accuracy and speed, suitable for system-level integration rather than extreme fine-tuning.

---

## 📂 Directory Structure

```text
yolo_detection/
├── weights/                # Trained model checkpoints
├── classes.txt             # List of detection classes
├── data.yaml               # YOLO dataset configuration
├── fashion-yolo-train.ipynb# Training notebook
├── EDA_Train.ipynb         # Exploratory analysis (train)
├── EDA_Val.ipynb           # Exploratory analysis (val)
└── README.md               # This file
```
## 🚀 Training

You can train the YOLOv11 detection model using either the **full dataset** or a **lightweight sample dataset**, depending on your purpose.

---

### 🔹 Option 1: Train with Full Dataset (Recommended)

If you want to train the model with the **complete fashion detection dataset**, please refer to:

👉 **Full Dataset (YOLO Detection)**  
https://www.kaggle.com/datasets/ngusix/fashion-yolo

**Dataset statistics:**

| Split | Number of Images |
|------:|------------------:|
| Train | **45,623** |
| Val   | **1,158** |

This dataset is suitable for:
- achieving stable detection performance
- evaluating SAHI sliced inference
- serving as a backbone for attribute recognition

---

### 🔹 Option 2: Train with Sample Dataset (Quick Demo)

If you only want to **quickly verify the training pipeline** or run experiments with limited resources, you can use the sample dataset:

👉 **Sample Dataset (Lightweight)**  
https://www.kaggle.com/datasets/ngusix/fashion-yolo-sample

**Dataset statistics:**

| Split | Number of Images |
|------:|------------------:|
| Train | **300** |
| Val   | **30** |

This option is recommended for:
- pipeline debugging
- code verification
- demonstration purposes

---

> 💡 Both datasets follow the same **YOLO annotation format** and can be used interchangeably with the same training scripts.



After download and unzip put the data folder into the yolo_detection and fix data path in data.yaml to train.
After reparing datasource, you can refer to the below file to training. 
```bash
fashio-yolo-train.ipynb
```

