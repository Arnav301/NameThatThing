from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


@dataclass
class Detection:
    cls_id: int
    name: str
    conf: float
    box: tuple  # (x1, y1, x2, y2)


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        # Loads and caches weights locally after first run
        self.model = YOLO(model_name)
        # Map of class id to readable name
        self.names = self.model.names

    def detect(self, image: Image.Image, conf: float = 0.25) -> List[Detection]:
        # Ultralytics accepts numpy arrays or PIL Images
        results = self.model(image, conf=conf, verbose=False)
        if not results:
            return []
        r = results[0]
        detections: List[Detection] = []
        boxes = r.boxes
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, p in zip(xyxy, clss, confs):
            name = self.names.get(int(c), str(int(c)))
            detections.append(
                Detection(
                    cls_id=int(c),
                    name=name,
                    conf=float(p),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections

    def draw_boxes(self, image: Image.Image, detections: List[Detection]) -> Image.Image:
        im = image.convert("RGB").copy()
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = (0, 175, 255)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            label = f"{det.name} {det.conf:.2f}"
            if font:
                tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            else:
                # Fallback size estimate
                tw, th = len(label) * 6, 10
            pad = 2
            draw.rectangle([(x1, y1 - th - pad * 2), (x1 + tw + pad * 2, y1)], fill=color)
            draw.text((x1 + pad, y1 - th - pad), label, fill=(0, 0, 0), font=font)
        return im
