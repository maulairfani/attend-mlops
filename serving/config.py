"""Environment-backed settings for the AI serving runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    jpeg_quality: int
    inference_every_n_frames: int
    face_capture_interval_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("AI_HOST", "0.0.0.0"),
            port=_get_int("AI_PORT", 5679),
            jpeg_quality=max(40, min(95, _get_int("AI_JPEG_QUALITY", 75))),
            inference_every_n_frames=max(1, _get_int("AI_INFERENCE_EVERY_N_FRAMES", 3)),
            face_capture_interval_seconds=max(1.0, _get_float("AI_FACE_CAPTURE_INTERVAL_SECONDS", 5.0)),
        )


settings = Settings.from_env()
