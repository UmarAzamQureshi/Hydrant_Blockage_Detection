# 🔥 Hydrant Blockage Detection

Smart computer vision system to detect fire hydrants and determine whether they are blocked (e.g., by parked vehicles) using YOLO object detection and simple overlap logic.

![Status](https://img.shields.io/badge/status-active-brightgreen) ![YOLO](https://img.shields.io/badge/YOLO-v8-blue) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)

---

### ✨ Features
- **Hydrant detection** in images and video
- **Blockage classification** using bounding-box overlap (IoU) with vehicles
- **Visual overlays**:
  - Green = hydrant clear
  - Red = hydrant blocked
  - Blue = detected vehicle

---

### 📁 Repository Structure (key items)
```
Hydrant-sentinel/
├─ script/
│  ├─ auto_label_hydrants.py
│  ├─ auto_label_car.py
│  ├─ blockage_detected.py       # Blockage logic + visualization
│  └─ splitdata.py               # Train/val split helper
├─ hydrant_data.yaml             # YOLO dataset config
├─ hydrant_dataset/
│  ├─ images/{train,val}/
│  └─ labels/{train,val}/
├─ runs/detect/train*/weights/   # Trained YOLO weights 
├─ Car/, Hydrant/                # Raw images (optional)
└─ README.md
```

If you are working from a clean copy, your structure may vary. Adjust paths in scripts accordingly.

---

### 📦 Requirements
Install dependencies (CPU-only shown; GPU users should install the appropriate Torch build):

```bash
pip install ultralytics opencv-python numpy pandas matplotlib
```

YOLO requires PyTorch. If not installed, visit the official PyTorch site and follow the selector for your platform, or for CPU you can typically do:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> Tip (Windows): Activate your virtual environment first, e.g. `venv\Scripts\activate`.

---

### 🚀 Quick Start
1) Prepare your dataset
- Label hydrants (class id 0) and vehicles (class id 1) in YOLO format
- Split into `train`/`val` (use `script/splitdata.py` if helpful)
- Ensure `hydrant_data.yaml` points to the right image/label directories

2) Train a YOLO model (example)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="hydrant_data.yaml", epochs=50, imgsz=640)
# Trained weights will appear under runs/detect/train*/weights/best.pt
```

3) Run inference + blockage logic (example)

```python
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")
result = model("path_to_test_image.jpg")[0]

boxes = result.boxes.xyxy.cpu().numpy()
classes = result.boxes.cls.cpu().numpy().astype(int)

hydrants = [b for b, c in zip(boxes, classes) if c == 0]
cars = [b for b, c in zip(boxes, classes) if c == 1]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

img = cv2.imread("path_to_test_image.jpg")

for car in cars:
    x1, y1, x2, y2 = map(int, car)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue car

for h in hydrants:
    blocked = any(iou(h, car) > 0.20 for car in cars)  # tune threshold
    color = (0, 0, 255) if blocked else (0, 255, 0)    # Red if blocked, else Green
    x1, y1, x2, y2 = map(int, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

cv2.imwrite("output.jpg", img)
```

> You can also use `script/blockage_detected.py` as a starting point to batch-process images and save visualizations.

---

### ⚙️ Blockage Logic (Overview)
- Compute IoU between each hydrant box and each vehicle box
- If IoU exceeds a threshold (e.g., **0.20**), mark hydrant as **blocked**
- Optional refinements:
  - Vertical alignment or y-distance constraints
  - Distance-to-curb or perspective heuristics per camera

Visualization helps calibrate thresholds for your camera angles and environment.

---

### 📊 Tips for Better Results
- Augment data (rotation, brightness, flips) to improve robustness
- Try different YOLO sizes (nano/small/medium) to balance speed vs accuracy
- Use early stopping and LR scheduling while training
- Calibrate thresholds per deployment camera and scene

---

### 🧪 Reproducibility Notes
- Keep a record of training runs under `runs/detect/` (CSV + images)
- Track dataset splits and seeds for consistent evaluation

---

### 📝 License & Acknowledgements
- Thanks to the Ultralytics team for YOLOv8
- If you build on this, please credit the original authors and document enhancements

---

### 🙋 Troubleshooting
- No detections: verify class IDs match your label schema and model classes
- Distorted colors or blank frames: confirm `cv2.imread` path and image format
- Poor blockage classification: adjust IoU threshold and consider y-axis filters


