# config/settings.py
import torch
import os

# Paths
DET_WEIGHTS = r"weights\best_yolo.pt"
ATTR_WEIGHTS = r"weights/attribute_head_best.pt"
ATTR_TXT_PATH = r"attributes.txt"
UPLOAD_DIR = r"temp/uploads"

# Detection
CONF_THRES = 0.5
IMG_SIZE = 640

# Attribute
ATTR_THRESH = 0.25

# SAHI
SAHI_SLICE = 640
SAHI_OVERLAP = 0.2

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(UPLOAD_DIR, exist_ok=True)
