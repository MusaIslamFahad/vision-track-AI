"""
app.py — VisionTrack AI · Streamlit application entry point.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Cloud or Hugging Face Spaces.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "utils"))

load_dotenv(ROOT / ".env.example")

from tracker import ObjectTracker
from video_utils import (
    get_video_info, save_temp_video, bgr_to_rgb,
    resize_frame, ensure_dir, cleanup_temp,
)
from analytics import (
    SessionStats, objects_over_time_chart, fps_over_time_chart,
    class_distribution_chart, track_id_timeline,
)
from logger import WandbLogger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark card style */
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #9E9C95; }
div[data-testid="metric-container"] {
    background: #1E1E1E;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px 18px;
}
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 6px 18px;
    font-weight: 500;
}
h1 { font-size: 2rem !important; }
.section-title { font-size: 1.1rem; font-weight: 600; color: #FAFAF8; margin: 1rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return config dict."""
    with st.sidebar:
        st.image("https://img.shields.io/badge/VisionTrack-AI-185FA5?style=for-the-badge&logo=opencv&logoColor=white")
        st.markdown("---")

        st.markdown("### 🤖 Model")
        model_name = st.selectbox(
            "YOLOv8 variant",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            index=0,
            help="n=nano (fastest), x=extra-large (most accurate). n recommended for CPU.",
        )
        tracker_algo = st.selectbox(
            "Tracker algorithm",
            ["botsort.yaml", "bytetrack.yaml"],
            index=0,
            help="BoT-SORT: appearance + motion (accurate). ByteTrack: motion only (faster).",
        )

        st.markdown("### 🎚️ Detection")
        confidence = st.slider("Confidence threshold", 0.1, 0.95, 0.45, 0.05,
                                help="Minimum score to consider a detection valid.")
        iou = st.slider("NMS IOU threshold", 0.1, 0.95, 0.50, 0.05,
                         help="Overlap threshold for non-max suppression.")

        st.markdown("### 🎨 Visualisation")
        show_trails   = st.toggle("Movement trails", value=True)
        trail_length  = st.slider("Trail length (frames)", 10, 80, 40, 5, disabled=not show_trails)
        show_labels   = st.toggle("Bounding box labels", value=True)
        show_fps      = st.toggle("FPS overlay", value=True)

        st.markdown("### 🏷️ Class filter")
        all_classes = ObjectTracker.COCO_CLASSES
        selected_classes = st.multiselect(
            "Detect only these classes (leave empty = all)",
            options=all_classes,
            default=[],
        )
        class_ids = [all_classes.index(c) for c in selected_classes] if selected_classes else None

        st.markdown("### 📊 W&B Logging")
        wandb_key = st.text_input("W&B API key (optional)", type="password",
                                   help="Paste your key to enable experiment tracking.")
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key

        st.markdown("---")
        st.caption("Built by Md. Musa Islam Fahad · CodeAlpha AI Internship · 2025")
        st.caption("[GitHub](https://github.com/MusaIslamFahad) · [Portfolio](https://musaislamfahad.vercel.app)")

    return dict(
        model_name=model_name,
        tracker_algo=tracker_algo,
        confidence=confidence,
        iou=iou,
        show_trails=show_trails,
        trail_length=trail_length,
        show_labels=show_labels,
        show_fps=show_fps,
        class_ids=class_ids,
    )


# ── Tracker factory (cached per config) ──────────────────────────────────────

@st.cache_resource(show_spinner="⚙️ Loading YOLO model weights …")
def get_tracker(model_name: str, tracker_algo: str) -> ObjectTracker:
    t = ObjectTracker(model_name=model_name, tracker=tracker_algo)
    t.load_model()
    return t


# ── Tab 1: Image inference ────────────────────────────────────────────────────

def tab_image(cfg: dict) -> None:
    st.markdown("Upload a single image to run object detection (no tracking — single frame).")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not uploaded:
        st.info("⬆️  Upload an image to get started.")
        return

    img = Image.open(uploaded).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    tracker = get_tracker(cfg["model_name"], cfg["tracker_algo"])
    tracker.confidence = cfg["confidence"]
    tracker.iou        = cfg["iou"]
    tracker.classes    = cfg["class_ids"]
    tracker.show_labels = cfg["show_labels"]
    tracker.show_fps   = False
    tracker.show_trails = False
    tracker.reset()

    with st.spinner("Running detection …"):
        result = tracker.process_frame(frame_bgr, frame_idx=0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(img, use_container_width=True)
    with col2:
        st.markdown("**Annotated**")
        st.image(bgr_to_rgb(result.frame), use_container_width=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects detected", len(result.detections))
    c2.metric("Unique classes", len(result.class_counts))
    c3.metric("Inference FPS", f"{result.fps:.1f}")

    if result.class_counts:
        fig = class_distribution_chart(result.class_counts)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if result.detections:
        import pandas as pd
        rows = [{"Track ID": d.track_id, "Class": d.class_name,
                 "Confidence": f"{d.confidence:.3f}",
                 "BBox (x1,y1,x2,y2)": str(d.bbox)} for d in result.detections]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── Tab 2: Video file ─────────────────────────────────────────────────────────

def tab_video(cfg: dict) -> None:
    st.markdown("Upload a video to run full object detection + tracking. Annotated video is downloadable.")

    MAX_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", 200))
    uploaded = st.file_uploader(
        f"Choose a video (max {MAX_MB} MB)",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )
    if not uploaded:
        st.info("⬆️  Upload a video to get started.")
        return

    size_mb = len(uploaded.getvalue()) / 1e6
    if size_mb > MAX_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Limit is {MAX_MB} MB.")
        return

    info_placeholder = st.empty()
    tmp_in  = save_temp_video(uploaded.getvalue(), suffix=Path(uploaded.name).suffix)
    tmp_out = tmp_in.replace(Path(uploaded.name).suffix, "_tracked.mp4")

    info = get_video_info(tmp_in)
    if info:
        info_placeholder.info(
            f"📹 **{uploaded.name}** · {info['width']}×{info['height']} · "
            f"{info['fps']:.1f} FPS · {info['frames']} frames · "
            f"{info['duration_s']:.1f}s"
        )

    tracker = get_tracker(cfg["model_name"], cfg["tracker_algo"])
    tracker.confidence  = cfg["confidence"]
    tracker.iou         = cfg["iou"]
    tracker.classes     = cfg["class_ids"]
    tracker.show_trails = cfg["show_trails"]
    tracker.trail_length = cfg["trail_length"]
    tracker.show_labels = cfg["show_labels"]
    tracker.show_fps    = cfg["show_fps"]
    tracker.reset()

    stats_acc = SessionStats()
    logger    = WandbLogger()
    logger.start_run(config=cfg, run_name=f"video-{uploaded.name}")

    progress_bar = st.progress(0, text="Processing frames …")
    preview_slot = st.empty()
    metrics_slot = st.empty()

    cap = cv2.VideoCapture(tmp_in)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps_src, (w, h))

    frame_idx = 0
    preview_every = max(1, total // 40)  # update preview ~40 times

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.process_frame(frame, frame_idx)
        writer.write(result.frame)
        stats_acc.update(frame_idx, result.detections, result.fps)
        logger.log({"fps": result.fps, "objects": len(result.detections)}, step=frame_idx)

        if frame_idx % preview_every == 0:
            preview_slot.image(bgr_to_rgb(resize_frame(result.frame, 800)),
                               caption=f"Frame {frame_idx}/{total}", use_container_width=True)
            with metrics_slot.container():
                ca, cb, cc = st.columns(3)
                ca.metric("Frame", f"{frame_idx}/{total}")
                cb.metric("FPS", f"{result.fps:.1f}")
                cc.metric("Objects", len(result.detections))

        progress_bar.progress(min((frame_idx + 1) / max(total, 1), 1.0),
                               text=f"Frame {frame_idx + 1}/{total}")
        frame_idx += 1

    cap.release()
    writer.release()

    summary = stats_acc.summary()
    logger.log_summary(summary)
    logger.finish()

    progress_bar.empty()
    preview_slot.empty()
    metrics_slot.empty()

    # ── Results ──
    st.success("✅ Processing complete!")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total frames", summary.get("total_frames_processed", 0))
    c2.metric("Unique tracks", summary.get("unique_tracks", 0))
    c3.metric("Avg objects/frame", summary.get("avg_objects_per_frame", 0))
    c4.metric("Avg FPS", summary.get("avg_fps", 0))

    # Charts
    df = stats_acc.to_dataframe()
    if not df.empty:
        tab_a, tab_b, tab_c, tab_d = st.tabs(
            ["Objects over time", "FPS over time", "Class distribution", "Class timeline"]
        )
        with tab_a:
            st.plotly_chart(objects_over_time_chart(df), use_container_width=True)
        with tab_b:
            st.plotly_chart(fps_over_time_chart(df), use_container_width=True)
        with tab_c:
            agg_cls: dict[str, int] = defaultdict(int)
            for row in stats_acc.frame_data:
                for k, v in row.items():
                    if k.startswith("cls_"):
                        agg_cls[k.replace("cls_", "")] += int(v)
            fig = class_distribution_chart(dict(agg_cls))
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with tab_d:
            fig2 = track_id_timeline(df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

    # Download
    with open(tmp_out, "rb") as f:
        st.download_button(
            label="⬇️  Download annotated video",
            data=f,
            file_name=f"tracked_{uploaded.name}",
            mime="video/mp4",
        )

    cleanup_temp(tmp_in)
    cleanup_temp(tmp_out)


# ── Tab 3: Webcam (experimental) ──────────────────────────────────────────────

def tab_webcam(cfg: dict) -> None:
    st.markdown(
        "**Live webcam inference** runs best locally. "
        "On Streamlit Cloud / HF Spaces, use the video upload tab instead."
    )

    start = st.button("▶️  Start webcam", type="primary")
    stop  = st.button("⏹  Stop")

    if not start:
        st.info("Click **Start webcam** to begin live detection.")
        return

    tracker = get_tracker(cfg["model_name"], cfg["tracker_algo"])
    tracker.confidence   = cfg["confidence"]
    tracker.iou          = cfg["iou"]
    tracker.classes      = cfg["class_ids"]
    tracker.show_trails  = cfg["show_trails"]
    tracker.trail_length = cfg["trail_length"]
    tracker.show_labels  = cfg["show_labels"]
    tracker.show_fps     = cfg["show_fps"]
    tracker.reset()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Make sure your browser has granted camera permission.")
        return

    frame_slot   = st.empty()
    metrics_slot = st.empty()
    stats_acc    = SessionStats()
    frame_idx    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break

        result = tracker.process_frame(frame, frame_idx)
        stats_acc.update(frame_idx, result.detections, result.fps)

        frame_slot.image(bgr_to_rgb(result.frame), channels="RGB", use_container_width=True)
        with metrics_slot.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("FPS", f"{result.fps:.1f}")
            c2.metric("Objects", len(result.detections))
            c3.metric("Total IDs", result.total_tracks)

        frame_idx += 1

    cap.release()
    st.info("Webcam stopped.")

    summary = stats_acc.summary()
    if summary:
        st.markdown("### Session summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames processed", summary["total_frames_processed"])
        c2.metric("Unique tracks", summary["unique_tracks"])
        c3.metric("Avg FPS", summary["avg_fps"])


# ── Tab 4: About ──────────────────────────────────────────────────────────────

def tab_about() -> None:
    st.markdown("""
## VisionTrack AI 🎯

A production-grade **Object Detection & Tracking** system built for the
**CodeAlpha AI Internship** (Task 4).

### Tech stack
| Component | Tool |
|---|---|
| Detection model | YOLOv8 (Ultralytics) — industry standard |
| Tracking | BoT-SORT / ByteTrack — state-of-the-art multi-object tracking |
| Experiment logging | Weights & Biases (W&B) |
| UI | Streamlit |
| Computer vision | OpenCV |
| Visualisation | Plotly |
| Deployment | Streamlit Cloud / Hugging Face Spaces |

### Features
- 🎯 Real-time object detection with YOLOv8n/s/m/l/x
- 🔁 Multi-object tracking with persistent IDs (BoT-SORT / ByteTrack)
- 🎨 Colour-coded bounding boxes + movement trail visualisation
- 🏷️ Per-class filtering (80 COCO classes)
- 📊 Live analytics dashboard (FPS, object counts, class breakdown)
- 📥 Downloadable annotated video output
- 📡 Optional W&B experiment logging
- 📸 Image / video file / webcam input modes

### Architecture
```
Input (image / video / webcam)
        │
        ▼
  YOLOv8 Detector  ──── pre-trained COCO weights
        │
        ▼
  BoT-SORT Tracker ──── appearance + motion cues
        │
        ▼
  Annotation engine ─── bounding boxes, trails, labels, FPS
        │
        ▼
  Streamlit UI ──────── live preview + analytics + download
        │
        ▼
  W&B Logger ────────── optional experiment tracking
```

### Author
**Md. Musa Islam Fahad**  
CSE (Data Science) · Daffodil International University  
[Portfolio](https://musaislamfahad.vercel.app) · [GitHub](https://github.com/MusaIslamFahad)  
CodeAlpha AI Internship · 2025
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🎯 VisionTrack AI")
    st.caption("Real-time Object Detection & Multi-Object Tracking · YOLOv8 + BoT-SORT · CodeAlpha AI Internship")

    cfg = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image", "🎬 Video", "📷 Webcam", "ℹ️ About"])

    with tab1:
        tab_image(cfg)
    with tab2:
        tab_video(cfg)
    with tab3:
        tab_webcam(cfg)
    with tab4:
        tab_about()


if __name__ == "__main__":
    main()
