# 👗 Fashion AI Pipeline  
**YOLOv11 + SAHI + Attribute Recognition**

This repository presents a **complete Computer Vision pipeline** for fashion understanding, including:

- Object Detection using **YOLOv11**
- Small-object enhancement using **SAHI (Sliced Inference)**
- Attribute recognition using **YOLO embeddings**
- Interactive demo using **Streamlit**
- Clean, modular, and reproducible codebase

The project is designed to demonstrate **end-to-end AI system integration**, rather than focusing solely on raw model accuracy.

---

## 📌 Project Overview

### Goals
- Detect fashion items and accessories from images
- Predict multiple attributes per detected object
- Provide a clean JSON output format for downstream usage
- Build a runnable demo for visualization and evaluation

---

## 🧠 Models Used

### 1️⃣ Object Detection
- **Model**: YOLOv11 (Ultralytics)
- Variants used:
  - `yolov11n` – prototyping
  - `yolov11s` – final detection model
- Task: Detect clothing and accessory objects

### 2️⃣ Attribute Recognition
- Backbone: YOLOv11 feature extractor (frozen)
- Input: Cropped object images
- Output: Multi-label attributes
- Training: Only the attribute head is trained (YOLO frozen)

### 3️⃣ SAHI (Sliced Inference)
- Used at inference time
- Improves detection of **small fashion accessories**
- Optional toggle in demo

---

## 📦 Dataset – Object Detection (YOLO)

### Dataset Layout

```text
fashion_yolo/
└── crops/
    ├── train/
    │   ├── images/
    │   └── labels/   # attribute IDs
    │
    └── val/
        ├── images/
        └── labels/

```



### Label Format (YOLO)

Each label file contains lines of:
```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalized** to the image size.

### 📄 Example Label File
```
42 0.447871 0.456543 0.008811 0.006836
32 0.414097 0.464355 0.076358 0.034180
35 0.511013 0.466309 0.008811 0.016602
42 0.563142 0.458984 0.004405 0.005859
```
---

This visualization is used to manually verify:
- Bounding box alignment
- Class correctness
- Label quality

---

## ⚙️ Training Configuration

- **Framework**: Ultralytics YOLO
- **Version**: YOLOv11s
- **Image size**: 640 × 640
- **Optimizer**: Default (Ultralytics)
- **Epochs**: 60
- **Hardware**: GPU (CUDA)T4 Kaggle

)

## 🚀 Training Process

The YOLOv11 detection model was trained using the Ultralytics framework.
For reproducibility and clarity, all training scripts, logs, and configuration files are organized in a dedicated folder.

👉 To view the full training pipeline, including:

- **Dataset configuration**
- **Training commands**
- **Experiment logs**
- **Model checkpoints**
Please refer to the following directory:
```text
yolo_DETECTION
```

This folder documents the complete training workflow for the object detection model and serves as a reference for further experimentation or retraining.


## 📊 Detection Performance & Evaluation

After training on the **full detection dataset**, the YOLOv11 model achieved the following results on the **validation set**.

---

### 🔹 Overall Performance

| Metric        | Value |
|--------------|-------|
| **mAP@0.5**  | **0.53** |
| **Precision** | **0.632** |
| **Recall**    | **0.494** |

These results are sufficient for:

- Reliable object localization
- Serving as input for **SAHI sliced inference**
- Supporting **downstream attribute recognition**

---

### 🔹 Performance on Small Accessories

Special attention was paid to **small fashion accessories**, which are critical for this task.

| Class | Precision |
|------|-----------|
| bag, wallet | 0.757 |
| watch | 0.615 |
| hat | 0.785 |
| headband / head covering / hair accessory | 0.632 |
| glasses | **0.890** |
| belt | 0.678 |
| shoe | **0.867** |

👉 **Average precision across small accessory classes exceeds 0.7**, validating the effectiveness of the detection pipeline and motivating the integration of **SAHI** for improved recall.

---

### 🔗 Role in the Full Pipeline

This detection model is used as:

- **Primary detector** in the Streamlit demo  
- **Backbone** for SAHI sliced inference  
- **Feature extractor** for attribute recognition via embeddings  

Accuracy is considered sufficient for **system-level demonstration**, with the focus placed on **pipeline completeness and integration** rather than peak benchmark scores.

---

### ✅ Summary

The **YOLOv11-based detector** provides a strong foundation for fashion object detection, achieving competitive accuracy while remaining flexible for advanced techniques such as **SAHI** and **multi-attribute prediction**.
