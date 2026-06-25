# 🎯 VisionTrack AI — Object Detection & Tracking

> **CodeAlpha AI Internship · Task 4**  
> Built by **Md. Musa Islam Fahad** · CSE (Data Science) · Daffodil International University

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-logging-yellow?logo=weightsandbiases)](https://wandb.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

**VisionTrack AI** is a production-grade object detection and multi-object tracking system.
It detects and tracks 80 COCO object classes across images, videos, and live webcam streams,
with a fully interactive Streamlit dashboard, real-time analytics, and optional W&B logging.


---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 Detection model | YOLOv8 (n / s / m / l / x variants) |
| 🔁 Tracking | BoT-SORT (appearance + motion) · ByteTrack (motion only) |
| 🎨 Visualisation | Bounding boxes · colour-coded track IDs · movement trails |
| 🏷️ Class filter | Filter any of 80 COCO classes from the sidebar |
| 📊 Analytics | FPS timeline · object count chart · class breakdown · track timeline |
| 📥 Export | Download annotated video (MP4) |
| 📡 Logging | W&B experiment tracking (optional) |
| 🐳 Docker | Full Dockerfile for containerised deployment |
| ☁️ Deploy | Streamlit Cloud · Hugging Face Spaces · Docker |

---

## 🗂️ Project Structure

```
object_detection_tracker/
│
├── app.py                      ← Streamlit main application
│
├── src/
│   └── tracker.py              ← ObjectTracker class (YOLO + BoT-SORT engine)
│
├── utils/
│   ├── analytics.py            ← SessionStats + Plotly chart builders
│   ├── video_utils.py          ← Video I/O helpers
│   └── logger.py               ← W&B experiment logger
│
├── .streamlit/
│   └── config.toml             ← Streamlit theme + server config
│
├── requirements.txt
├── Dockerfile
├── .env.example                ← Environment variable template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/MusaIslamFahad/CodeAlpha_ObjectDetectionTracking
cd CodeAlpha_ObjectDetectionTracking
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional)
```bash
cp .env.example .env
# Edit .env — add WANDB_API_KEY if you want experiment logging
```

### 5. Run the app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t visiontrack-ai .

# Run (CPU)
docker run -p 8501:8501 visiontrack-ai

# Run (GPU)
docker run --gpus all -p 8501:8501 visiontrack-ai
```

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** SDK
3. Push this repository:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/visiontrack-ai
git push hf main
```
4. (Optional) Apply for a **Free GPU grant** in your Space settings for real-time inference.

---

## ☁️ Deploy to Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → `app.py` → Deploy
4. Add secrets in the Streamlit Cloud dashboard if using W&B

---

## 📊 W&B Experiment Tracking

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from Settings
3. Paste it in the sidebar **W&B API key** field (or add to `.env`)

Each video processing session logs:
- Per-frame FPS and object count
- Session summary (unique tracks, avg FPS, peak objects)
- Model config (model variant, confidence, IOU, tracker)

---

## 🧠 Architecture

```
Input (Image / Video File / Webcam)
         │
         ▼
   ┌─────────────┐
   │  YOLOv8     │  ← Pre-trained on COCO (80 classes)
   │  Detector   │  ← Configurable: n/s/m/l/x variant
   └──────┬──────┘
          │  raw detections (bbox, class, confidence)
          ▼
   ┌─────────────┐
   │  BoT-SORT   │  ← Appearance embedding + Kalman filter
   │  Tracker    │  ← Persistent track IDs across frames
   └──────┬──────┘
          │  tracked detections (bbox, class, track_id)
          ▼
   ┌─────────────────────┐
   │  Annotation Engine  │  ← Bounding boxes, trails, labels, FPS HUD
   └──────┬──────────────┘
          │
          ▼
   ┌─────────────────┐     ┌────────────┐
   │  Streamlit UI   │────▶│  W&B Logger│  (optional)
   │  Dashboard      │     └────────────┘
   │  Analytics      │
   │  Download       │
   └─────────────────┘
```

---

## 🎯 Model Variants

| Model | Size | Speed (CPU) | mAP50-95 | Best for |
|---|---|---|---|---|
| yolov8n | 6.2 MB | ~8 FPS | 37.3 | CPU / edge devices |
| yolov8s | 21.5 MB | ~5 FPS | 44.9 | Balanced |
| yolov8m | 49.7 MB | ~3 FPS | 50.2 | Higher accuracy |
| yolov8l | 83.7 MB | ~2 FPS | 52.9 | High accuracy |
| yolov8x | 130.5 MB | ~1 FPS | 53.9 | Max accuracy (GPU only) |

> **Recommendation:** Use `yolov8n` for CPU/Streamlit Cloud. Use `yolov8m` or larger on GPU.

---

## 📦 COCO Classes (80)

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## 🤝 Author

**Md. Musa Islam Fahad**  
CSE (Data Science) · Daffodil International University, Dhaka  
📧 musa.islam.fahad@gmail.com  
🌐 [Portfolio](https://musaislamfahad.vercel.app) · [GitHub](https://github.com/MusaIslamFahad) · [LinkedIn](https://linkedin.com/in/md-musa-islam-fahad-b18759249)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
