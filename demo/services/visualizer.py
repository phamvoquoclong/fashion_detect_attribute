# services/visualizer.py
from sahi.utils.cv import visualize_object_predictions
import numpy as np
from PIL import Image


def draw_sahi(image_rgb, predictions):
    vis = visualize_object_predictions(
        image=image_rgb,
        object_prediction_list=predictions,
        rect_th=2,
        text_size=0.7,
        text_th=2
    )

    if isinstance(vis, dict):
        vis = vis["image"]

    return Image.fromarray(vis)
