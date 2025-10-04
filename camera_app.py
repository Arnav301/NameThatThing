from __future__ import annotations

import time
from collections import Counter
from typing import List

import cv2
import numpy as np
from PIL import Image

from detectors.yolo import YOLODetector, Detection
from llm.describe import summarize_scene, llm_available


def pil_from_bgr(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def draw_hud(frame, counts: Counter, last_desc: str | None, use_llm: bool):
    h, w = frame.shape[:2]
    y = 24
    x = 10
    # Background rect for HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120 if last_desc else 60), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # Counts line
    counts_txt = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) if counts else "No objects"
    cv2.putText(frame, f"Counts: {counts_txt}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    y += 22
    # Help
    help_txt = "[q] quit  [d] describe scene (Gemini)  [c] toggle LLM: " + ("ON" if use_llm else "OFF")
    cv2.putText(frame, help_txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,255,200), 1, cv2.LINE_AA)
    y += 22

    if last_desc:
        # Wrap description roughly by width
        max_chars = max(20, int(w / 12))
        desc_lines = []
        s = last_desc.strip()
        while s:
            desc_lines.append(s[:max_chars])
            s = s[max_chars:]
        for line in desc_lines[:3]:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,230,180), 1, cv2.LINE_AA)
            y += 20


def main():
    print("Starting webcam object detection (no Streamlit)...")
    print("Hotkeys: q=quit, d=describe scene, c=toggle LLM on/off")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open default camera (index 0). Try another index or check permissions.")
        return

    detector = YOLODetector("yolov8n.pt")

    use_llm = llm_available()
    last_desc: str | None = None
    last_counts: Counter = Counter()

    # Process every frame; you can skip frames for speed if needed
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: Failed to read frame from camera.")
            break

        # Run detection (convert to PIL for our helper)
        pil_im = pil_from_bgr(frame)
        detections: List[Detection] = detector.detect(pil_im, conf=0.35)
        annotated_pil = detector.draw_boxes(pil_im, detections)
        annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

        # Update counts
        names = [d.name for d in detections]
        last_counts = Counter(names)

        # Draw HUD
        draw_hud(annotated_bgr, last_counts, last_desc, use_llm)

        cv2.imshow("YOLO + Gemini (press q to quit)", annotated_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            use_llm = not use_llm
            if not use_llm:
                last_desc = None
        elif key == ord('d'):
            # Generate description on-demand to avoid latency every frame
            if use_llm:
                try:
                    last_desc = summarize_scene({k: int(v) for k, v in last_counts.items()})
                except Exception as e:
                    print(f"LLM error: {e}")
                    last_desc = "(LLM error. See console.)"
            else:
                last_desc = summarize_scene({k: int(v) for k, v in last_counts.items()})

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
