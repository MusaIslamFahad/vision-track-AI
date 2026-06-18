"""
tracker.py — Multi-object tracking engine.
Wraps YOLOv8's built-in BoT-SORT / ByteTrack / DeepSORT trackers and
exposes a clean frame-by-frame API used by the Streamlit app.
"""

from __future__ import annotations

import time
import colorsys
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ── Colour palette ──────────────────────────────────────────────────────────

def _generate_palette(n: int = 100) -> list[tuple[int, int, int]]:
    palette = []
    for i in range(n):
        h = i / n
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette

PALETTE = _generate_palette(100)


def get_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[int(track_id) % len(PALETTE)]


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    center: tuple[int, int]


@dataclass
class FrameResult:
    frame: np.ndarray
    detections: list[Detection]
    fps: float
    frame_idx: int
    total_tracks: int
    class_counts: dict[str, int]


@dataclass
class TrackHistory:
    centers: deque = field(default_factory=lambda: deque(maxlen=60))


# ── Main Tracker class ───────────────────────────────────────────────────────

class ObjectTracker:
    """
    Wraps YOLOv8 + built-in tracker (BoT-SORT / ByteTrack).

    Parameters
    ----------
    model_name : str
        YOLOv8 model variant: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
    tracker : str
        Tracker config: 'botsort.yaml' | 'bytetrack.yaml'
    confidence : float
        Detection confidence threshold (0–1)
    iou : float
        NMS IOU threshold (0–1)
    classes : list[int] | None
        Filter to specific COCO class IDs; None = all classes
    show_trails : bool
        Draw movement trails behind tracked objects
    trail_length : int
        Number of historical centers to draw
    show_labels : bool
        Draw class label + confidence on bounding boxes
    show_fps : bool
        Overlay FPS counter on frame
    device : str
        'cpu', 'cuda', 'mps'
    """

    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self,
        model_name: str = "yolov8n",
        tracker: str = "botsort.yaml",
        confidence: float = 0.45,
        iou: float = 0.50,
        classes: Optional[list[int]] = None,
        show_trails: bool = True,
        trail_length: int = 40,
        show_labels: bool = True,
        show_fps: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.tracker_cfg = tracker
        self.confidence = confidence
        self.iou = iou
        self.classes = classes
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.show_labels = show_labels
        self.show_fps = show_fps
        self.device = device

        self._model: Optional[YOLO] = None
        self._track_history: dict[int, TrackHistory] = defaultdict(TrackHistory)
        self._frame_times: deque = deque(maxlen=30)
        self._unique_ids: set[int] = set()

    # ── Model loading ────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Download (first time) and load the YOLO model."""
        self._model = YOLO(f"{self.model_name}.pt")

    @property
    def model(self) -> YOLO:
        if self._model is None:
            self.load_model()
        return self._model

    # ── Reset state ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._track_history.clear()
        self._frame_times.clear()
        self._unique_ids.clear()

    # ── Core inference ────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> FrameResult:
        """Run detection + tracking on a single BGR frame."""
        t0 = time.perf_counter()

        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            tracker=self.tracker_cfg,
            device=self.device,
            verbose=False,
        )

        t1 = time.perf_counter()
        self._frame_times.append(t1 - t0)
        fps = len(self._frame_times) / sum(self._frame_times)

        detections: list[Detection] = []
        class_counts: dict[str, int] = defaultdict(int)
        annotated = frame.copy()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                for box, track_id_t, cls_t, conf_t in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.id.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                ):
                    track_id = int(track_id_t)
                    class_id = int(cls_t)
                    conf = float(conf_t)
                    class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else str(class_id)

                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    self._unique_ids.add(track_id)
                    history = self._track_history[track_id]
                    history.centers.append((cx, cy))
                    class_counts[class_name] += 1

                    det = Detection(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                    )
                    detections.append(det)

                    color = get_color(track_id)
                    self._draw_detection(annotated, det, color)

            # Draw trails after all boxes (so trails appear under boxes)
            if self.show_trails:
                for det in detections:
                    self._draw_trail(annotated, det.track_id, get_color(det.track_id))

        if self.show_fps:
            self._draw_fps(annotated, fps, len(detections))

        return FrameResult(
            frame=annotated,
            detections=detections,
            fps=fps,
            frame_idx=frame_idx,
            total_tracks=len(self._unique_ids),
            class_counts=dict(class_counts),
        )

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_detection(self, frame: np.ndarray, det: Detection, color: tuple) -> None:
        x1, y1, x2, y2 = det.bbox
        thickness = 2

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if self.show_labels:
            label = f"#{det.track_id} {det.class_name} {det.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            # Label background
            cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw + 4, y1), color, -1)
            # Label text (white)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Center dot
        cv2.circle(frame, det.center, 4, color, -1)

    def _draw_trail(self, frame: np.ndarray, track_id: int, color: tuple) -> None:
        pts = list(self._track_history[track_id].centers)[-self.trail_length:]
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(ch * alpha) for ch in color)
            cv2.line(frame, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)

    def _draw_fps(self, frame: np.ndarray, fps: float, count: int) -> None:
        h, w = frame.shape[:2]
        overlay_text = f"FPS: {fps:.1f}  |  Objects: {count}"
        cv2.rectangle(frame, (0, 0), (260, 34), (0, 0, 0), -1)
        cv2.putText(frame, overlay_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    # ── Video file processing ─────────────────────────────────────────────────

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback=None,
    ) -> dict:
        """
        Process an entire video file, write annotated output, return stats.
        progress_callback(pct: float) called each frame if provided.
        """
        self.reset()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

        all_class_counts: dict[str, int] = defaultdict(int)
        frame_idx = 0
        fps_readings = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame, frame_idx)
            writer.write(result.frame)
            fps_readings.append(result.fps)

            for cls, cnt in result.class_counts.items():
                all_class_counts[cls] += cnt

            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames)

            frame_idx += 1

        cap.release()
        writer.release()

        return {
            "total_frames": frame_idx,
            "total_unique_tracks": len(self._unique_ids),
            "avg_fps": float(np.mean(fps_readings)) if fps_readings else 0.0,
            "class_counts": dict(all_class_counts),
            "output_path": output_path,
        }
