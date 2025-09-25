## 🔥 Hydrant Blockage Detection

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
├─ runs/detect/train*/weights/   # Trained YOLO weights (best.pt, last.pt)
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

3) Run inference + blockage logic (reference)

Use the following scripts as reference for batch processing and blockage evaluation.

```python
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ---------------- Config ----------------
INPUT_FOLDER = "/content/hydrant_pano_images"
OUTPUT_FOLDER = "/content/result_out"
DEBUG_FOLDER = "/content/result_debug"

COVER_FRAC = 0.25          # % hydrant covered by car
FOREGROUND_Y_MARGIN = 10   # pixels, car centroid must be below hydrant centroid

# Class IDs (⚠️ adjust these to your trained model’s IDs)
HYDRANT_CLASS = 10   # change if hydrant is not class 0
CAR_CLASS = 2       # change if car is not class 1

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ---------------- Utils ----------------
def get_box_and_mask(result, class_id):
    """Extract bounding boxes & masks for a given class_id."""
    objects = []
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy().astype(bool)
    else:
        masks = [None] * len(result.boxes)

    for i, box in enumerate(result.boxes):
        cls = int(box.cls)
        if cls != class_id:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        centroid = ((y1 + y2) // 2, (x1 + x2) // 2)
        mask = masks[i] if masks[i] is not None else None
        objects.append({
            "box": (x1, y1, x2, y2),
            "mask": mask,
            "centroid": centroid
        })
    return objects


def hydrant_blocked(hydrants, cars, use_masks=True, cover_thresh=0.15, horiz_thresh=0.3, prox_thresh=30):
    results = [False] * len(hydrants)

    for hi, h in enumerate(hydrants):
        hx1, hy1, hx2, hy2 = h["box"]
        h_w, h_h = hx2 - hx1, hy2 - hy1
        h_cy = (hy1 + hy2) / 2

        # Expand hydrant region to be tolerant
        pad = 20
        hx1e, hy1e, hx2e, hy2e = hx1-pad, hy1-pad, hx2+pad, hy2+pad

        for ci, c in enumerate(cars):
            cx1, cy1, cx2, cy2 = c["box"]

            blocked = False
            overlap_ratio = 0.0
            horiz_frac = 0.0
            horiz_gap = 9999
            vertical_ok = cy2 >= h_cy

            if use_masks and h.get("mask") is not None and c.get("mask") is not None:
                # mask overlap
                h_mask = h["mask"].astype(np.uint8)
                c_mask = c["mask"].astype(np.uint8)
                inter = cv2.bitwise_and(h_mask, c_mask)
                overlap_ratio = inter.sum() / (h_mask.sum() + 1e-6)
                if overlap_ratio >= cover_thresh:
                    blocked = True

            # Fallback: bbox overlap
            if not blocked:
                overlap_w = max(0, min(hx2e, cx2) - max(hx1e, cx1))
                horiz_frac = overlap_w / (h_w + 1e-6)
                horiz_gap = min(abs(cx1 - hx2e), abs(hx1e - cx2))
                if (horiz_frac >= horiz_thresh and vertical_ok) or (horiz_gap <= prox_thresh and vertical_ok):
                    blocked = True

            # 🔎 Debug print
            print(f"\nHydrant {hi} vs Car {ci}:")
            print(f"  Hydrant box: {h['box']}")
            print(f"  Car box:     {c['box']}")
            print(f"  horiz_frac={horiz_frac:.2f}, vertical_ok={vertical_ok}, prox_gap={horiz_gap}")
            print(f"  overlap_ratio={overlap_ratio:.2f}")

            if blocked:
                results[hi] = True
                break

    return results


def draw_overlay(img, hydrants, cars, blocked_status):
    """Draw results on image."""
    out = img.copy()

    # cars
    for c in cars:
        x1, y1, x2, y2 = c["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(out, "Car", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 2)

    # hydrants
    for idx, (h, blocked) in enumerate(zip(hydrants, blocked_status)):
        x1, y1, x2, y2 = h["box"]
        color = (0,0,255) if blocked else (255,0,0)
        label = f"Hydrant {idx} BLOCKED" if blocked else f"Hydrant {idx} CLEAR"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    return out


def draw_debug(img, hydrants, cars):
    """Draw hydrants (blue) and cars (red) for debugging."""
    debug_img = img.copy()
    for h in hydrants:
        x1, y1, x2, y2 = h["box"]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue = hydrant
    for c in cars:
        x1, y1, x2, y2 = c["box"]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = car
    return debug_img


# ---------------- Main ----------------
def process_image(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load {img_path}")

    results = model.predict(img, verbose=False)[0]

    hydrants = get_box_and_mask(results, HYDRANT_CLASS)
    cars = get_box_and_mask(results, CAR_CLASS)

    blocked_status = hydrant_blocked(hydrants, cars)

    # Overlay + debug
    out = draw_overlay(img, hydrants, cars, blocked_status)
    debug_img = draw_debug(img, hydrants, cars)

    return out, debug_img, blocked_status


if __name__ == "__main__":
    # 🔧 load your trained model
    model = YOLO("yolov8s.pt")   # change path if needed

    for img_file in os.listdir(INPUT_FOLDER):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_FOLDER, img_file)
        out, debug_img, status = process_image(model, img_path)

        blocked_ids = [i for i, b in enumerate(status) if b]
        if blocked_ids:
            print(f"🚨 Blocked: {img_file} — hydrants {blocked_ids}")
        else:
            print(f"✅ Clear: {img_file}")

        # Save overlays
        out_path = os.path.join(OUTPUT_FOLDER, img_file)
        debug_path = os.path.join(DEBUG_FOLDER, "debug_" + img_file)
        cv2.imwrite(out_path, out)
        cv2.imwrite(debug_path, debug_img)
```

```python
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ---------------- Config ----------------
INPUT_FOLDER = "/content/hydrant_pano_images"
OUTPUT_FOLDER = "/content/result_out"

COVER_FRAC = 0.25          # % hydrant covered by car
FOREGROUND_Y_MARGIN = 10   # pixels, car centroid must be below hydrant centroid

# Class IDs (⚠️ adjust these to your trained model’s IDs)
HYDRANT_CLASS = 10  # change if hydrant is not class 10
CAR_CLASS = 2       # change if car is not class 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------- Utils ----------------
def get_box_and_mask(result, class_id):
    """Extract bounding boxes & masks for a given class_id."""
    objects = []
    masks = result.masks.data.cpu().numpy().astype(bool) if result.masks is not None else [None] * len(result.boxes)

    for i, box in enumerate(result.boxes):
        cls = int(box.cls)
        if cls != class_id:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        centroid = ((y1 + y2) // 2, (x1 + x2) // 2)
        mask = masks[i] if masks[i] is not None else None
        objects.append({
            "box": (x1, y1, x2, y2),
            "mask": mask,
            "centroid": centroid
        })
    return objects


def hydrant_blocked(hydrants, cars, use_masks=True, cover_thresh=0.15, horiz_thresh=0.3, prox_thresh=30):
    """Determine if hydrants are blocked by cars."""
    results = [False] * len(hydrants)

    for hi, h in enumerate(hydrants):
        hx1, hy1, hx2, hy2 = h["box"]
        h_w, h_h = hx2 - hx1, hy2 - hy1
        h_cy = (hy1 + hy2) / 2
        pad = 20
        hx1e, hy1e, hx2e, hy2e = hx1 - pad, hy1 - pad, hx2 + pad, hy2 + pad

        for c in cars:
            cx1, cy1, cx2, cy2 = c["box"]
            blocked = False
            vertical_ok = cy2 >= h_cy

            # Mask overlap check
            if use_masks and h.get("mask") is not None and c.get("mask") is not None:
                h_mask = h["mask"].astype(np.uint8)
                c_mask = c["mask"].astype(np.uint8)
                inter = cv2.bitwise_and(h_mask, c_mask)
                overlap_ratio = inter.sum() / (h_mask.sum() + 1e-6)
                if overlap_ratio >= cover_thresh:
                    blocked = True

            # Bounding box overlap fallback
            if not blocked:
                overlap_w = max(0, min(hx2e, cx2) - max(hx1e, cx1))
                horiz_frac = overlap_w / (h_w + 1e-6)
                horiz_gap = min(abs(cx1 - hx2e), abs(hx1e - cx2))
                if (horiz_frac >= horiz_thresh and vertical_ok) or (horiz_gap <= prox_thresh and vertical_ok):
                    blocked = True

            if blocked:
                results[hi] = True
                break

    return results


def draw_overlay(img, hydrants, cars, blocked_status):
    """Draw results on image."""
    out = img.copy()

    # Draw cars
    for c in cars:
        x1, y1, x2, y2 = c["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "Car", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw hydrants
    for idx, (h, blocked) in enumerate(zip(hydrants, blocked_status)):
        x1, y1, x2, y2 = h["box"]
        color = (0, 0, 255) if blocked else (255, 0, 0)
        label = f"Hydrant {idx} BLOCKED" if blocked else f"Hydrant {idx} CLEAR"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return out


# ---------------- Main ----------------
def process_image(model, img_path):
    """Run detection and overlay for a single image."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load {img_path}")

    results = model.predict(img, verbose=False)[0]

    hydrants = get_box_and_mask(results, HYDRANT_CLASS)
    cars = get_box_and_mask(results, CAR_CLASS)

    blocked_status = hydrant_blocked(hydrants, cars)
    out = draw_overlay(img, hydrants, cars, blocked_status)

    return out, blocked_status


if __name__ == "__main__":
    # Load YOLO model
    model = YOLO("yolov8s.pt")  # Change path if needed

    for img_file in os.listdir(INPUT_FOLDER):
        if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_FOLDER, img_file)
        out, status = process_image(model, img_path)

        blocked_ids = [i for i, b in enumerate(status) if b]
        if blocked_ids:
            print(f"🚨 Blocked: {img_file} — hydrants {blocked_ids}")
        else:
            print(f"✅ Clear: {img_file}")

        # Save overlay
        out_path = os.path.join(OUTPUT_FOLDER, img_file)
        cv2.imwrite(out_path, out)
```

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
- MIT License – feel free to fork, adapt, and modify
- Thanks to the Ultralytics team for YOLOv8
- If you build on this, please credit the original authors and document enhancements

---

### 🙋 Troubleshooting
- No detections: verify class IDs match your label schema and model classes
- Distorted colors or blank frames: confirm `cv2.imread` path and image format
- Poor blockage classification: adjust IoU threshold and consider y-axis filters


