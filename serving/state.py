"""Mutable runtime state for camera and feed websocket sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fastapi import WebSocket


@dataclass
class RuntimeState:
    viewers: dict[WebSocket, str] = field(default_factory=dict)
    camera_ws: WebSocket | None = None
    camera_connected: bool = False

    start_time: float = field(default_factory=time.time)
    frame_count: int = 0
    faces_detected_total: int = 0

    latest_normal_frame: bytes = b""
    latest_metadata: dict | None = None
    latest_detections: list[dict] = field(default_factory=list)

    last_face_capture_ts: float = 0.0

    @property
    def viewer_count(self) -> int:
        return len(self.viewers)

    @property
    def uptime(self) -> int:
        return int(time.time() - self.start_time)


state = RuntimeState()
