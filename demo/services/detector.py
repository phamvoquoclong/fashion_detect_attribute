# services/detector.py
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def load_detector(weights, conf_thres, device):
    yolo = YOLO(weights)
    yolo.model.to(device)
    yolo.model.eval()

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model=yolo,
        confidence_threshold=conf_thres,
        device="cuda:0" if device == "cuda" else "cpu"
    )

    return yolo, sahi_model


def run_sahi(img_path, sahi_model, slice_size, overlap):
    return get_sliced_prediction(
        image=img_path,
        detection_model=sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )
