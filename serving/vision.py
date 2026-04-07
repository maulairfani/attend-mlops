"""Frame decode/encode and lightweight CPU face detection helpers."""

from __future__ import annotations

import cv2
import numpy as np


class FaceDetector:
    """Simple OpenCV Haar-cascade detector for runtime face boxes."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        if self._cascade.empty():
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        detections: list[dict] = []
        for (x, y, w, h) in faces:
            detections.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "name": "Unknown",
                    "confidence": 0.0,
                    "source": "unknown",
                }
            )
        return detections


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def encode_jpeg(frame_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return encoded.tobytes()


def draw_detections(frame_bgr: np.ndarray, detections: list[dict]) -> None:
    for det in detections:
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (80, 255, 120), 2)
        cv2.putText(
            frame_bgr,
            det.get("name") or "Unknown",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )
