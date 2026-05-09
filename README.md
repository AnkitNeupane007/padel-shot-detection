# Padel Match Analysis System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-red.svg)

An automated, computer-vision-based system for analyzing Padel matches. This project tracks players, the ball, and rackets to automatically detect and classify shots (Forehand, Backhand, Volley, Smash) using physics heuristics and pose estimation.

---

## 📽️ Input and Output Examples

### Input Video

<!-- <img src="input/input.mp4" width="600"> -->
[Watch the input video](./input/input.mp4)

### Output Video

[Watch the output video](./output/padel_analysis_output.mp4)

---

## 🧠 System Architecture & Flow

The core system is implemented in the `PadelAnalyzer` class, which processes the match video frame by frame. The standard pipeline follows these steps:

1. **Player Tracking:** Detects players using a pre-trained YOLOv8 model and tracks them across frames using the BotSORT tracker.
2. **Racket Detection:** Uses a YOLO model to identify player rackets.
3. **Ball Tracking:** Employs a custom-trained YOLOv8 model to pinpoint the ball's location in each frame.
4. **Trajectory & Physics Analysis:** Monitors the ball's trajectory vector. A drastic change in the ball's direction accompanied by high speed triggers a "shot event".
5. **Shot Attribution:** Once a shot is detected, the system calculates the distance to the nearest tracked player to assign the shot.
6. **Pose Classification:** A cropped region of the hitting player is passed through **MediaPipe Pose**. By analyzing the wrist and shoulder landmarks at the moment of impact, the system classifies the shot as a **Forehand**, **Backhand**, **Volley**, or **Smash**.
7. **Logging:** Shot data (frame, timestamp, player ID, shot type) is exported to a CSV report (`padel_report.csv`).

---

## 🔬 Methods & Methodologies

### The Ball Detection Journey

Detecting a small, fast-moving ball in a complex background is challenging.

- **Initial Attempt (Raw OpenCV):** The first iteration relied on raw OpenCV techniques such as color masking, background subtraction, and contour detection. This approach quickly failed because it was too sensitive to noise—every small, moving pixel cluster (shoes, reflections, court debris) was falsely detected as a ball.
- **Current Solution (Custom YOLOv8):** To achieve robustness, the system now uses a **custom-trained YOLOv8 model** specifically fine-tuned on tennis/padel ball datasets. This deep learning approach drastically reduced false positives and gracefully handles motion blur. The players and racket detection however is done through the base YOLO model.

### Pose Estimation for Shot Classification

Instead of deploying a complex temporal action recognition model, this system uses an efficient heuristic-based approach. Using **MediaPipe**, we extract skeletal landmarks at the exact frame the physics engine detects a hit:

- **Smash:** Detected if the average wrist height is significantly above the shoulder height.
- **Volley:** Detected if the wrist height is roughly horizontal/equal to the shoulder height.
- **Forehand / Backhand:** Determined by the relative X-coordinates of the right wrist and right shoulder (depending on the camera angle and handedness).

---

## ⚠️ Limitations & Known Issues

- **Player ID Switching:** The BotSORT tracking relies heavily on spatial continuity. If a player leaves the frame or is heavily occluded and later reappears, the tracker loses the original ID and assigns a new one.
- **2D Distance Heuristics:** Shot attribution relies on 2D pixel distance between the ball and the player. In specific camera angles, a ball flying high in the air might incorrectly appear "close" to a player standing in the background.
- **Camera Angle Dependency:** The left/right heuristics for forehand and backhand classifications currently assume a fixed perspective and a right-handed player.

---

## 🚀 Future Enhancements

To make the system even more robust and capable for professional-grade analytics, the following improvements are planned:

1. **Re-Identification (ReID) Integration:** Implement a robust visual feature extractor (like OSNet) to maintain player IDs perfectly across occlusions and side-switches.
2. **Perspective Transformation (Homography):** Map the 2D video coordinates to a top-down 2D court mini-map. This will allow for true 3D spatial distancing, heatmaps, and tactical tactical positioning analysis.
3. **Advanced Shot Classification:** Train a lightweight sequential model (e.g., LSTM or Transformer) on a sequence of pose landmarks rather than a single frame to classify complex shots like slices, drop shots, and topspin drives.
4. **Automated Highlight Clipping:** Automatically generate short video clips of the longest rallies or fastest smashes based on the generated event log.

---

## 💻 Quick Start

**1. Install Dependencies:**

```bash
pip install ultralytics mediapipe opencv-python numpy
```

**2. Run the Analyzer:**
Ensure you have the required YOLO weights (`yolov8s.pt` and your custom ball model `best.pt`) in the correct paths.

```bash
python shot_type.py
```

**3. Output:**
Check `padel_analysis_output.mp4` for the annotated video and `padel_report.csv` for the statistical breakdown.
