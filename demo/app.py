# app.py
from config.settings import *
from models.attribute_model import AttributePredictor
from services.detector import load_detector, run_sahi
from services.visualizer import draw_sahi
from utils.attribute_mapper import load_attribute_map

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import uuid
from PIL import Image


# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(page_title="Fashion Detection Demo", layout="wide")
st.title("🧥 Fashion Detection Demo")
st.write("YOLOv11 + SAHI + Attribute Recognition")


# ======================================================
# LOAD RESOURCES
# ======================================================
@st.cache_resource
def load_all():
    yolo, sahi_model = load_detector(
        DET_WEIGHTS, CONF_THRES, DEVICE
    )

    attr_predictor = AttributePredictor(
        yolo_model=yolo,
        attr_ckpt_path=ATTR_WEIGHTS,
        device=DEVICE,
        attr_thresh=ATTR_THRESH
    )

    attr_id2name = load_attribute_map(ATTR_TXT_PATH)
    print('LOAD MODEL : DONE')

    return yolo, sahi_model, attr_predictor, attr_id2name


yolo_model, sahi_model, attr_predictor, attr_id2name = load_all()


# ======================================================
# CONTROLS
# ======================================================
use_sahi = st.checkbox("Use SAHI (better small-object recall)", value=False)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)


# ======================================================
# INFERENCE
# ======================================================
if uploaded_file is not None:
    img_id = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_DIR, f"{img_id}.jpg")

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    rows = []
    json_output = []

    with st.spinner("Running detection..."):
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        img_bgr = img_rgb[..., ::-1]

        if use_sahi:
            sahi_res = run_sahi(
                img_path, sahi_model, SAHI_SLICE, SAHI_OVERLAP
            )

            valid_preds = []

            for obj in sahi_res.object_prediction_list:
                score = float(obj.score.value)
                if score < CONF_THRES:
                    continue

                x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
                crop = img_bgr[y1:y2, x1:x2]

                attrs = attr_predictor.predict_from_crop(crop)
                attr_names = [attr_id2name.get(a, str(a)) for a in attrs]

                rows.append({
                    "Class": obj.category.name,
                    "Confidence": round(score, 3),
                    "Attributes": ", ".join(attr_names)
                })

                json_output.append({
                    "label": obj.category.name,
                    "confidence": round(score, 3),
                    "box": [x1, y1, x2, y2],
                    "attributes": attr_names
                })

                valid_preds.append(obj)

            output_image = draw_sahi(img_rgb, valid_preds)

        else:
            results = yolo_model.predict(
                source=img_bgr,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                device=0,
                save=False,
                verbose=False
            )[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                score = float(box.conf[0])

                crop = img_bgr[y1:y2, x1:x2]
                attrs = attr_predictor.predict_from_crop(crop)
                attr_names = [attr_id2name.get(a, str(a)) for a in attrs]

                rows.append({
                    "Class": yolo_model.names[cls_id],
                    "Confidence": round(score, 3),
                    "Attributes": ", ".join(attr_names)
                })

                json_output.append({
                    "label": yolo_model.names[cls_id],
                    "confidence": round(score, 3),
                    "box": [x1, y1, x2, y2],
                    "attributes": attr_names
                })

            output_image = Image.fromarray(results.plot()[..., ::-1])

    # ======================================================
    # DISPLAY
    # ======================================================
    col_img, col_tbl = st.columns([3, 2])

    with col_img:
        st.subheader("Detection Result")
        st.image(output_image, use_container_width=True)

    with col_tbl:
        st.subheader("Detected Objects")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("📦 Output JSON (Submission Format)")
    st.code(
        json.dumps(json_output, indent=2, ensure_ascii=False),
        language="json")

else:
    st.info("Please upload an image to start detection.")
