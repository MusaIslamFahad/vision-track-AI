"""
video_utils.py — Helpers for video I/O and frame manipulation.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


def get_video_info(path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {}
    info = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info


def frame_generator(path: str, max_dim: int = 1280) -> Generator[np.ndarray, None, None]:
    """Yield BGR frames from a video file, optionally resized."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        yield frame
    cap.release()


def save_temp_video(uploaded_bytes: bytes, suffix: str = ".mp4") -> str:
    """Write uploaded bytes to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, max_dim: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def cleanup_temp(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass
