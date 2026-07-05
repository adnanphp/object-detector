
#  Safety Equipment Detection System

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/adnanphp/object-detector)
[![Gradio](https://img.shields.io/badge/Gradio-Live%20Demo-orange)](https://f18d2b91a5b6fb1189.gradio.live)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-11-green)](https://ultralytics.com)

##  Overview

A real-time object detection system for workplace safety monitoring using **YOLO11** (You Only Look Once). Detects safety equipment violations and helps enforce workplace safety protocols.

### 🎯 Detects:
-  **Helmets** - Safety headgear compliance
- 🚫 **No-Helmet** - Violation detection (Red alert!)
- 🦺 **Safety Vests** - High-visibility clothing
- 👷 **People** - Worker detection

##  Live Demo

| Platform | URL | Status |
|----------|-----|--------|
| **Hugging Face Spaces** | https://huggingface.co/spaces/adnanphp/object-detector | ✅ Live |
| **Gradio Share** | https://f18d2b91a5b6fb1189.gradio.live | ✅ Live (1 week) |

## 🚀 Features

-  **Real-time detection** with YOLO11
-  **Image & Video processing**
-  **Confidence scores** for each detection
-  **Detection statistics** with class counts
-  **Interactive web interface** with Gradio
- **Color-coded bounding boxes**
-  **Temporary public URL** for sharing

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Ultralytics YOLO11** | Object Detection Model |
| **OpenCV** | Image/Video Processing |
| **Gradio** | Web Interface |
| **PyTorch** | Deep Learning Framework |
| **Python 3.10** | Programming Language |

##  Detection Classes & Colors

| Class | Color | Description |
|-------|-------|-------------|
| Helmet |  Green | Safety compliance |
| No-Helmet | 🔴 Red | Safety violation |
| Vest |  Yellow | Safety equipment |
| Person |  Blue | Worker detection |

## 📁 Project Structure

```
object-detector/
├── app.py              # Gradio web interface
├── detector.py         # Detection engine
├── train_yolo.py       # Training script
├── yolo11n.pt          # Pretrained YOLO model
├── requirements.txt    # Python dependencies
├── Dockerfile          # Containerization
├── README.md           # Documentation
└── screenshot/         # Demo images
```

##  Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/adnanphp/object-detector.git
cd object-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Access the App
- **Local:** http://localhost:7860
- **Public:** https://f18d2b91a5b6fb1189.gradio.live

### Docker (Optional - CPU Only)

```bash
# Build the image (CPU version)
docker build -f Dockerfile.cpu -t object-detector-cpu .

# Run the container
docker run -p 7860:7860 object-detector-cpu
```

## 🎯 Use Cases

-  **Construction Site Safety**
-  **Industrial Workplace Monitoring**
-  **Corporate Safety Compliance**
-  **Safety Audit Assistance**

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Model** | YOLO11 Nano |
| **Classes** | 4 (Helmet, No-Helmet, Vest, Person) |
| **Input Size** | 640x640 |
| **Inference Speed** | Real-time |

##  Testing

Upload images or videos to test the detection:
1. Click on **"Image Detection"** tab
2. Upload an image
3. Click **"Detect Objects"**
4. View annotated results with statistics

##  Troubleshooting

### Common Issues:

**Slow performance on CPU:**
```bash
# Use smaller model
model = YOLO('yolo11n.pt')  # Already configured
```

**Out of memory:**
```bash
# Reduce batch size in training
batch = 4  # In train_yolo.py
```

## Future Improvements

- [ ] Training on custom dataset
- [ ] Real-time video streaming
- [ ] Alert system for violations
- [ ] Dashboard for safety metrics
- [ ] Mobile app integration

##  License

MIT License - Feel free to use and modify!

##  Author

**Adnan** - [GitHub](https://github.com/adnanphp) | [LinkedIn](https://linkedin.com/in/adnanphp)

##  Acknowledgments

- [Ultralytics](https://ultralytics.com) for YOLO
- [Gradio](https://gradio.app) for the UI
- [Roboflow](https://roboflow.com) for dataset


---
**Live Demo:** https://huggingface.co/spaces/adnanphp/object-detector
EOF
```

## 🎯 NOW COPY THIS ENTIRE BLOCK and paste it into your terminal!

That's it! Just copy everything from `cat > README.md << 'EOF'` to the final `EOF` and paste it. Your README.md file will be created with all the content.

After that, run:
```bash
git add README.md
git commit -m "docs: add complete README"
git push
```
