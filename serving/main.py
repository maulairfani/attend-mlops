"""Kamera-compatible AI runtime service.

Provides:
- /ws/camera: camera frame ingestion
- /ws/feed: viewer stream output
- /healthz and /api/status: runtime health/status
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import time
from datetime import datetime, timezone

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from serving.config import settings
from serving.state import state
from serving.vision import FaceDetector, decode_jpeg, draw_detections, encode_jpeg


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("attend-mlops-serving")

app = FastAPI(title="Attend MLOps AI Serving", version="0.1.0")

VALID_BG_MODES = {"normal", "green", "blur", "black"}
detector = FaceDetector()


def _parse_camera_packet(payload: bytes, receive_timestamp_ms: float) -> tuple[float, bytes]:
    """Return (camera_timestamp_ms, jpeg_bytes) with support for legacy payloads."""
    if len(payload) >= 2 and payload[:2] == b"\xff\xd8":
        return receive_timestamp_ms, payload

    if len(payload) <= 8:
        return receive_timestamp_ms, payload

    try:
        camera_ts_ms = struct.unpack(">d", payload[:8])[0]
        return camera_ts_ms, payload[8:]
    except struct.error:
        return receive_timestamp_ms, payload


async def _broadcast_frame_and_metadata(frame_jpeg: bytes, metadata_json: str) -> None:
    disconnected: list[WebSocket] = []
    for viewer_ws in list(state.viewers.keys()):
        try:
            await viewer_ws.send_bytes(frame_jpeg)
            await viewer_ws.send_text(metadata_json)
        except Exception:
            disconnected.append(viewer_ws)

    for viewer_ws in disconnected:
        state.viewers.pop(viewer_ws, None)


async def _broadcast_text(message_json: str) -> None:
    disconnected: list[WebSocket] = []
    for viewer_ws in list(state.viewers.keys()):
        try:
            await viewer_ws.send_text(message_json)
        except Exception:
            disconnected.append(viewer_ws)

    for viewer_ws in disconnected:
        state.viewers.pop(viewer_ws, None)


def _maybe_build_face_capture(frame_bgr, detections: list[dict]) -> str | None:
    if not detections:
        return None

    now = time.time()
    if now - state.last_face_capture_ts < settings.face_capture_interval_seconds:
        return None

    x, y, w, h = detections[0]["bbox"]
    crop = frame_bgr[max(0, y): max(0, y + h), max(0, x): max(0, x + w)]
    if crop.size == 0:
        return None

    thumb = cv2.resize(crop, (112, 112))
    thumb_jpeg = encode_jpeg(thumb, quality=70)
    if not thumb_jpeg:
        return None

    state.last_face_capture_ts = now
    payload = {
        "type": "face_capture",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": None,
        "confidence": 0.0,
        "thumbnail": base64.b64encode(thumb_jpeg).decode("ascii"),
        "bbox": [x, y, w, h],
    }
    return json.dumps(payload)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def api_status() -> dict:
    return {
        "camera_connected": state.camera_connected,
        "viewer_count": state.viewer_count,
        "uptime": state.uptime,
        "total_frames": state.frame_count,
        "faces_detected_total": state.faces_detected_total,
    }


@app.websocket("/ws/feed")
async def websocket_feed(ws: WebSocket) -> None:
    await ws.accept()
    state.viewers[ws] = "normal"
    logger.info("Viewer connected (%s total)", state.viewer_count)

    if state.latest_normal_frame:
        try:
            await ws.send_bytes(state.latest_normal_frame)
        except Exception:
            logger.debug("Failed to send initial frame", exc_info=True)

    if state.latest_metadata:
        try:
            await ws.send_text(json.dumps(state.latest_metadata))
        except Exception:
            logger.debug("Failed to send initial metadata", exc_info=True)

    try:
        while True:
            message = await ws.receive_text()
            try:
                payload = json.loads(message)
                mode = payload.get("bg")
                if mode in VALID_BG_MODES:
                    state.viewers[ws] = mode
            except (json.JSONDecodeError, AttributeError):
                logger.debug("Invalid viewer message: %s", message)
    except WebSocketDisconnect:
        pass
    finally:
        state.viewers.pop(ws, None)
        logger.info("Viewer disconnected (%s total)", state.viewer_count)


@app.websocket("/ws/camera")
async def websocket_camera(ws: WebSocket) -> None:
    await ws.accept()
    state.camera_ws = ws
    state.camera_connected = True
    logger.info("Camera connected")

    try:
        while True:
            raw_payload = await ws.receive_bytes()
            receive_ts = time.time() * 1000.0
            camera_ts, jpeg_bytes = _parse_camera_packet(raw_payload, receive_ts)

            frame = decode_jpeg(jpeg_bytes)
            if frame is None:
                continue

            state.frame_count += 1

            if state.frame_count % settings.inference_every_n_frames == 0:
                detections = detector.detect(frame)
                state.latest_detections = detections
                state.faces_detected_total += len(detections)
            else:
                detections = state.latest_detections

            draw_detections(frame, detections)
            frame_jpeg = encode_jpeg(frame, quality=settings.jpeg_quality)
            if not frame_jpeg:
                continue

            send_ts = time.time() * 1000.0
            metadata = {
                "type": "frame_metadata",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "frame": state.frame_count,
                "faces": len(detections),
                "detections": detections,
                "camera_ts": camera_ts,
                "server_ts": send_ts,
                "processing_ms": round(send_ts - receive_ts, 1),
            }
            metadata_json = json.dumps(metadata)

            state.latest_normal_frame = frame_jpeg
            state.latest_metadata = metadata

            await _broadcast_frame_and_metadata(frame_jpeg, metadata_json)

            capture_json = _maybe_build_face_capture(frame, detections)
            if capture_json:
                await _broadcast_text(capture_json)
    except WebSocketDisconnect:
        pass
    finally:
        state.camera_connected = False
        state.camera_ws = None
        logger.info("Camera disconnected")


def run() -> None:
    import uvicorn

    uvicorn.run(
        "serving.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
