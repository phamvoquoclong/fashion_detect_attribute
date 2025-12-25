from ultralytics import YOLO
from utils.attribute_mapper import load_attribute_map
from models.attribute_model import AttributePredictor
from config.settings import *

import cv2
import json
# ================= USER CONFIG =================
IMAGE_PATH = r"D:\test\fashion_yolo_sample\images\val\5f5e8dd2003101a233a31bfc1b940085.jpg"   # 👈 ĐỔI PATH ẢNH Ở ĐÂY
SAVE_OUTPUT = True
OUTPUT_IMAGE_PATH = "output_result.jpg"
# ===============================================


# ================= LOAD MODELS =================
print("🔹 Loading YOLO detection model...")
det_model = YOLO(DET_WEIGHTS)
det_model.model.to(DEVICE)
det_model.model.eval()

print("🔹 Loading attribute model...")
attr_predictor = AttributePredictor(
    yolo_model=det_model,
    attr_ckpt_path=ATTR_WEIGHTS,
    device=DEVICE,
    attr_thresh=ATTR_THRESH
)

print("🔹 Loading attribute mapping...")
attr_id2name = load_attribute_map(ATTR_TXT_PATH)


# ================= LOAD IMAGE =================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"❌ Cannot read image: {IMAGE_PATH}")

h, w = img.shape[:2]
vis_img = img.copy()

print(f"🖼️ Image loaded: {IMAGE_PATH} ({w}x{h})")


# ================= DETECTION =================
results = det_model.predict(
    source=img,
    imgsz=IMG_SIZE,
    conf=CONF_THRES,
    device=0 if DEVICE == "cuda" else "cpu",
    verbose=False
)[0]


# ================= POST-PROCESS =================
json_output = []

for box in results.boxes:
    cls_id = int(box.cls[0])
    cls_name = det_model.names[cls_id]
    score = float(box.conf[0])

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # ---- ATTRIBUTE PREDICTION ----
    crop = img[y1:y2, x1:x2]
    attrs = attr_predictor.predict_from_crop(crop)
    attr_names = [attr_id2name.get(a, str(a)) for a in attrs]

    # ---- DRAW BOX ----
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"{cls_name} {score:.2f}"
    cv2.putText(
        vis_img,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # ---- JSON FORMAT ----
    json_output.append({
        "label": cls_name,
        "confidence": round(score, 3),
        "box": [x1, y1, x2, y2],
        "attributes": attr_names
    })


# ================= SAVE / SHOW =================
if SAVE_OUTPUT:
    cv2.imwrite(OUTPUT_IMAGE_PATH, vis_img)
    print(f"💾 Output image saved to: {OUTPUT_IMAGE_PATH}")

cv2.imshow("Inference Result", vis_img)
print("➡️ Press any key to close image")
cv2.waitKey(0)
cv2.destroyAllWindows()


# ================= PRINT JSON =================
print("\n📦 OUTPUT JSON:")
print(json.dumps(json_output, indent=2, ensure_ascii=False))
