<!--
---
title: VisionTrack AI
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
short_description: Real-time Object Detection & Tracking with YOLOv8 + BoT-SORT
---
-->

<div align="center">
  
# 🎯 VisionTrack AI - Object Detection & Tracking

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Analytics-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![W&B](https://img.shields.io/badge/Weights_%26_Biases-Logging-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)


**Production-grade object detection and tracking - YOLOv8 + BoT-SORT/ByteTrack, interactive Streamlit dashboard, Plotly analytics, MP4 export and optional W&B logging.**

> Built by **Md. Musa Islam Fahad** · CSE (Data Science) · Daffodil International University

</div>

---

## 📖 Overview

**VisionTrack AI** is a production-grade, multi-object detection and tracking system. It combines **YOLOv8** (Ultralytics) for fast and accurate detection with **BoT-SORT** or **ByteTrack** to assign persistent IDs and render movement trails across frames for all 80 COCO object classes.

An interactive **Streamlit dashboard** exposes full control over models, trackers, and thresholds, and renders live Plotly analytics (FPS timeline, object count chart, class breakdown, track timeline). Sessions can optionally be logged to **Weights & Biases**, and the annotated output is exportable as an **MP4 video**.

The system works on:
- 🎥 Live webcam streams
- 📁 Pre-recorded video files
- 🖼️ Static images

---

## ✨ Features

| Feature | Details |
|---|---|
| 🤖 **Detection Model** | YOLOv8 (n / s / m / l / x variants) - pre-trained on COCO 80 classes |
| 🔁 **Dual Tracker Support** | BoT-SORT (appearance + motion) · ByteTrack (motion only) - switchable from UI |
| 🎨 **Rich Visualisation** | Colour-coded bounding boxes, persistent track IDs, movement trails |
| 🏷️ **Class Filter** | Filter any of the 80 COCO classes directly from the sidebar |
| 📊 **Live Analytics** | Plotly charts - FPS timeline, object count, class breakdown, track timeline |
| 📥 **MP4 Export** | Download the fully annotated video from the dashboard |
| 📡 **W&B Logging** | Per-frame FPS, object count, session summary, model config (optional) |
| 📷 **Multi-source Input** | Webcam, video file, or image - selectable from the UI |
| 🐳 **Docker Ready** | Full Dockerfile for containerised CPU or GPU deployment |
| ☁️ **Multi-platform Deploy** | Streamlit Cloud · HuggingFace Spaces · Docker |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Detection Model | [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics) |
| Trackers | BoT-SORT (appearance + Kalman filter) · ByteTrack (motion-only) |
| Computer Vision | [OpenCV](https://opencv.org/) |
| Deep Learning Backend | PyTorch |
| UI / Dashboard | [Streamlit](https://streamlit.io/) 1.35 |
| Analytics Charts | [Plotly](https://plotly.com/python/) |
| Experiment Tracking | [Weights & Biases](https://wandb.ai/) *(optional)* |

---

## 📁 Project Structure

```
object_detection_tracker/
│
├── app.py                      # Streamlit main application — entry point
│
├── src/
│   └── tracker.py              # ObjectTracker class (YOLOv8 + BoT-SORT engine)
│
├── utils/
│   ├── analytics.py            # SessionStats + Plotly chart builders
│   ├── video_utils.py          # Video I/O helpers (read, write, frame extraction)
│   └── logger.py               # W&B experiment logger
│
├── .streamlit/
│   └── config.toml             # Streamlit theme + server configuration
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build definition
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/MusaIslamFahad/codealpha_tasks.git
cd codealpha_tasks/CodeAlpha_Object_Detection_and_Tracking
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics opencv-python streamlit plotly wandb python-dotenv
```

> **GPU Acceleration:** PyTorch is installed automatically via `ultralytics`. For CUDA support, install the GPU-enabled build from [pytorch.org](https://pytorch.org/get-started/locally/) *before* running `pip install -r requirements.txt`.

### 4. Configure environment variables *(optional - for W&B)*

```bash
cp .env.example .env
# Edit .env and add: WANDB_API_KEY=your_key_here
```

### 5. Run the app

```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## 🚀 Usage

### Streamlit Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Use the sidebar to:
- Select input source (webcam / video file / image)
- Choose model variant (`yolov8n` → `yolov8x`)
- Select tracker (`BoT-SORT` or `ByteTrack`)
- Set confidence and IOU thresholds
- Filter specific COCO classes
- Enter your W&B API key *(optional)*

The main panel shows the annotated live feed, Plotly analytics charts below, and a download button for the MP4 export.

### Keyboard Controls *(webcam mode)*

| Key | Action |
|-----|--------|
| `q` | Quit / stop stream |

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t visiontrack-ai .

# Run on CPU
docker run -p 8501:8501 visiontrack-ai

# Run with GPU
docker run --gpus all -p 8501:8501 visiontrack-ai

# Open http://localhost:8501
```

---

## ☁️ Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Streamlit** SDK
3. Push this repository to the Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/visiontrack-ai
git push hf main
```

4. *(Optional)* Apply for a **Free GPU grant** in your Space settings for real-time inference

---

## ☁️ Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Select repo → `app.py` → Deploy
4. Add secrets in the Streamlit Cloud dashboard if using W&B:
   - **Key:** `WANDB_API_KEY` → **Value:** your API key

---

## 📊 W&B Experiment Tracking

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from **Settings → API Keys**
3. Paste it in the sidebar **W&B API key** field, or add it to `.env`

Each processing session automatically logs:

| Metric | Description |
|--------|-------------|
| Per-frame FPS | Inference speed over time |
| Object count | Number of detections per frame |
| Unique tracks | Total distinct objects tracked per session |
| Avg FPS / Peak objects | Session-level summary stats |
| Model config | Variant, confidence, IOU, tracker, and class filter settings |

---

## 🧠 Architecture

```
Input (Image / Video File / Webcam)
         │
         ▼
   ┌─────────────┐
   │  YOLOv8     │  ← Pre-trained on COCO (80 classes)
   │  Detector   │  ← Configurable: n / s / m / l / x variant
   └──────┬──────┘
          │  raw detections (bbox, class, confidence)
          ▼
   ┌──────────────────────────┐
   │  Tracker                 │  ← BoT-SORT: appearance embedding + Kalman filter
   │  (selectable from UI)    │  ← ByteTrack: motion-only, lighter & faster
   └──────┬───────────────────┘
          │  tracked detections (bbox, class, track_id)
          ▼
   ┌─────────────────────┐
   │  Annotation Engine  │  ← Bounding boxes, colour-coded IDs, trails, FPS HUD
   └──────┬──────────────┘
          │
          ▼
   ┌───────────────────────────────┐     ┌────────────────┐
   │  Streamlit Dashboard (app.py) │────▶│   W&B Logger   │ (optional)
   │  ├─ Live annotated feed       │     │  (logger.py)   │
   │  ├─ Plotly analytics charts   │     └────────────────┘
   │  │   ├─ FPS timeline          │
   │  │   ├─ Object count chart    │
   │  │   ├─ Class breakdown       │
   │  │   └─ Track timeline        │
   │  └─ MP4 download button       │
   └───────────────────────────────┘
```

**Step-by-step:**

1. **Detection**: YOLOv8 processes each frame and returns bounding boxes, class labels, and confidence scores.
2. **Tracking**: The selected tracker (BoT-SORT or ByteTrack) matches detections across frames and assigns persistent IDs.
3. **Annotation**: The annotation engine draws bounding boxes, colour-coded track IDs, and movement trails on each frame.
4. **Dashboard**: Streamlit renders the annotated frames live alongside Plotly charts for FPS, object count, class breakdown, and track timelines.
5. **Export**: The processed video is written to disk by `video_utils.py` and made available as an MP4 download.
6. **Logging**: If W&B is configured, `logger.py` pushes per-frame and session-level metrics in real time.

---

## 🎯 Model Variants

```python
# Swap model by changing one line in app.py or src/tracker.py
model = YOLO("yolov8n.pt")  # Nano         - fastest, lowest VRAM
model = YOLO("yolov8s.pt")  # Small
model = YOLO("yolov8m.pt")  # Medium
model = YOLO("yolov8l.pt")  # Large
model = YOLO("yolov8x.pt")  # Extra-large  - most accurate
```

| Model | Size | Speed (CPU) | mAP50-95 | Best For |
|-------|------|-------------|----------|----------|
| yolov8n | 6.2 MB | ~8 FPS | 37.3 | CPU / edge / Streamlit Cloud |
| yolov8s | 21.5 MB | ~5 FPS | 44.9 | Balanced speed + accuracy |
| yolov8m | 49.7 MB | ~3 FPS | 50.2 | Higher accuracy |
| yolov8l | 83.7 MB | ~2 FPS | 52.9 | High accuracy |
| yolov8x | 130.5 MB | ~1 FPS | 53.9 | Max accuracy (GPU recommended) |

> **Recommendation:** Use `yolov8n` on CPU or Streamlit Cloud. Use `yolov8m` or larger on a dedicated GPU.

---

## 🔁 Tracker Comparison

| Tracker | Algorithm | Speed | ID Stability | Best For |
|---------|-----------|-------|--------------|----------|
| **BoT-SORT** | Appearance embedding + Kalman filter | Medium | ⭐⭐⭐⭐⭐ | Crowded scenes, re-identification |
| **ByteTrack** | Motion-only (IoU matching) | Fast | ⭐⭐⭐ | Sparse scenes, low-VRAM environments |

---

## 📋 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
streamlit>=1.35.0
plotly>=5.18.0
wandb>=0.16.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

Python version: **3.10 or higher**

---

## 📦 COCO Classes (80)

<details>
<summary>Click to expand full class list</summary>

`person` `bicycle` `car` `motorcycle` `airplane` `bus` `train` `truck` `boat` `traffic light` `fire hydrant` `stop sign` `parking meter` `bench` `bird` `cat` `dog` `horse` `sheep` `cow` `elephant` `bear` `zebra` `giraffe` `backpack` `umbrella` `handbag` `tie` `suitcase` `frisbee` `skis` `snowboard` `sports ball` `kite` `baseball bat` `baseball glove` `skateboard` `surfboard` `tennis racket` `bottle` `wine glass` `cup` `fork` `knife` `spoon` `bowl` `banana` `apple` `sandwich` `orange` `broccoli` `carrot` `hot dog` `pizza` `donut` `cake` `chair` `couch` `potted plant` `bed` `dining table` `toilet` `tv` `laptop` `mouse` `remote` `keyboard` `cell phone` `microwave` `oven` `toaster` `sink` `refrigerator` `book` `clock` `vase` `scissors` `teddy bear` `hair drier` `toothbrush`

</details>

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👤 Author

**Md. Musa Islam Fahad**  
CSE (Data Science) · Daffodil International University, Dhaka  
📧 musa.islam.fahad@gmail.com  
🌐 [Portfolio](https://musaislamfahad.vercel.app) · [GitHub](https://github.com/MusaIslamFahad) · [LinkedIn](https://linkedin.com/in/md-musa-islam-fahad-b18759249)

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8, BoT-SORT, and ByteTrack integration
- [OpenCV](https://opencv.org/) for computer vision utilities
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [Plotly](https://plotly.com/python/) for interactive analytics charts
- [Weights & Biases](https://wandb.ai/) for experiment tracking infrastructure
- [CodeAlpha](https://www.codealpha.tech/) for the internship opportunity and project brief

---

<div align="center">

⭐ If you found this useful or built something cool on top of it, drop a star. It helps a lot!

**[⬆ Back to Top](#-visiontrack-ai--object-detection--tracking)**

</div>
