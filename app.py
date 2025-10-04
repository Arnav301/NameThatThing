from __future__ import annotations

import io
from collections import Counter
from typing import List

import numpy as np
from PIL import Image
import streamlit as st

from detectors.yolo import YOLODetector, Detection
from llm.describe import summarize_scene, describe_objects_brief, llm_available


st.set_page_config(page_title="Object Detector + LLM", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def get_detector() -> YOLODetector:
    return YOLODetector("yolov8n.pt")


def load_image(file) -> Image.Image:
    im = Image.open(file)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    return im


def main():
    st.title("Object Detection + LLM Descriptions")
    st.caption("YOLOv8 for detection. Optional Gemini LLM (via .env) for text descriptions.")

    with st.sidebar:
        st.header("Controls")
        conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        extra = st.text_input("Extra context (optional)", placeholder="e.g., indoor photo, product shot, etc.")
        use_llm = st.toggle("Use LLM (if API key available)", value=True, help="Provide GEMINI_API_KEY in a .env file")
        st.divider()
        st.write("1) Upload an image, 2) Click Detect")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if uploaded is not None:
            image = load_image(uploaded)
            st.image(image, caption="Original", use_column_width=True)
        else:
            st.info("Upload an image to get started.")
            return

        if st.button("Detect", type="primary"):
            with st.spinner("Running YOLOv8..."):
                detector = get_detector()
                detections: List[Detection] = detector.detect(image, conf=conf)
                annotated = detector.draw_boxes(image, detections)

            st.subheader("Detections")
            st.image(annotated, caption="Annotated", use_column_width=True)

            names = [d.name for d in detections]
            counts = Counter(names)
            if counts:
                st.write({k: int(v) for k, v in counts.items()})
            else:
                st.write("No objects detected above threshold.")

            with col_right:
                st.subheader("Description")
                if use_llm and llm_available():
                    txt = summarize_scene({k: int(v) for k, v in counts.items()}, extra_context=extra or None)
                    st.write(txt)

                    brief = describe_objects_brief({k: int(v) for k, v in counts.items()})
                    if brief:
                        st.caption(brief)
                else:
                    st.write(summarize_scene({k: int(v) for k, v in counts.items()}, extra_context=extra or None))
                    st.caption("LLM disabled or not configured; showing heuristic summary.")


if __name__ == "__main__":
    main()
