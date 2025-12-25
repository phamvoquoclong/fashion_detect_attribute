
---

## 🎯 Demo Objectives

The goal of this demo is to demonstrate a **complete and practical computer vision pipeline**, including:

- Multi-object fashion detection
- Improved recall for small objects (watches, accessories)
- Multi-label attribute prediction per detected object
- Clean, standardized output format for downstream usage

This demo is intended as a **proof-of-concept system**.

---

## 🚀 Features

### ✅ Object Detection
- YOLOv11-based detection
- Supports apparel and accessory categories
- GPU acceleration via CUDA when available

### ✅ SAHI Integration
- Optional SAHI inference for small objects
- Toggleable directly from the UI
- Helps recover tiny fashion items often missed by standard inference

### ✅ Attribute Recognition
- Each detected object is cropped and passed to an attribute head
- Multi-label prediction (one object → multiple attributes)
- Attribute IDs are mapped to human-readable names

### ✅ Output Formats
- Visual output with bounding boxes
- Tabular summary (class, confidence, attributes)
- JSON output matching submission format


---

## ⚙️ Configuration (`config/settings.py`)

All important runtime configurations for the demo are centralized in a single file:
Please refer to the following directory:
```text

demo/
├── config/
│ └── settings.py


This design allows you to **easily customize paths(detection weight, attribute weight), thresholds for both detection, attribute, SAHI** without modifying the core application logic.
```
---


## Example JSON output:
```json
[
  {
    "label": "Cardigan",
    "confidence": 0.97,
    "box": [100, 175, 715, 971],
    "attributes": ["Plain pattern", "Short length", "Single breasted"]
  }
]
```
To view fully the streamlit page you can refer to Fashion Detection Demo_yellowdress.pdf.