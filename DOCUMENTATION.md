# SignSpeak – Detailed Project Documentation

This document describes the **Indian Sign Language (ISL)** recognition project in depth: models, architecture, data flow, and how to obtain and use the output.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Models](#2-models)
3. [System Architecture](#3-system-architecture)
4. [How It Works (Data Flow)](#4-how-it-works-data-flow)
5. [Getting Output](#5-getting-output)
6. [Training Pipeline](#6-training-pipeline)
7. [Project Structure](#7-project-structure)
8. [API Reference](#8-api-reference)
9. [Configuration & Paths](#9-configuration--paths)
10. [Docker Deployment](#10-docker-deployment)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

**SignSpeak** is a real-time **Indian Sign Language (ISL)** recognition system that:

- Captures live video from a webcam.
- Detects hand(s) and extracts **hand landmarks** using **MediaPipe**.
- Classifies gestures using the **ISL Keras model** only (trained on Indian Sign Language).
- Streams the video and sends **predictions** to a web UI via **Flask** and **WebSocket**.
- Displays recognized characters in a text area and supports **text-to-speech** (Speak All).

**Recognized classes:** 35 in total — digits **1–9** and letters **A–Z**. A special gesture (left-hand open palm “C”) is mapped to **space** for word separation.

---

## 2. Models

**Prediction uses only the ISL Keras landmark model** (trained on Indian Sign Language). The same **84-dimensional** feature vector is used (see [Feature extraction](#feature-extraction)).

### 2.1 ISL Keras Landmark Model (used for prediction)

| Property | Value |
|----------|--------|
| **Type** | Dense neural network (Keras/TensorFlow) |
| **File** | `model/indian_sign_model.h5` or `checkpoints/best_model_*.h5` |
| **Input** | 84 features (normalized landmark vector) |
| **Output** | 35 classes (softmax probabilities) |
| **Labels** | `['1','2',…,'9','A','B',…,'Z']` |

**Architecture (layers):**

```
Input(84)
    → Dense(256) + ReLU
    → BatchNormalization
    → Dropout(0.3)
    → Dense(128) + ReLU
    → BatchNormalization
    → Dropout(0.3)
    → Dense(64) + ReLU
    → BatchNormalization
    → Dropout(0.2)
    → Dense(35, softmax)
```

- **Parameters:** ~67K trainable.
- **Inference:** ~2–5 ms per prediction on CPU (see `profile_models.py`).
- **Role:** Only model used for live prediction (Indian Sign Language).

### 2.2 ISL Skeleton Model (RandomForest) — not used for prediction

The skeleton model (`model/model.p`) may still be loaded for compatibility but **is not used for prediction**. All live output comes from the Keras model.

### 2.3 Feature Extraction (84-D Vector)

The Keras model expects a **single 84-dimensional** vector per frame:

- **One hand:** 42 values (21 landmarks × 2 coordinates) for that hand + **42 zeros** (padding).
- **Two hands:** 42 values for hand 1 + 42 values for hand 2.

**Keras landmark features (used for prediction):**

- 21 landmarks per hand converted to **pixel** coordinates via `calc_landmark_list()`.
- **Pre-processing:** relative to landmark 0 (wrist), flattened to 42 values, then normalized by max absolute value (`pre_process_landmark()`).
- Combined for two hands (or padded with zeros for one hand) → 84 values (wrist-relative, normalized by scale).

---

## 3. System Architecture

High-level layout:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Web Browser (Client)                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  index.html     │  │  Socket.IO      │  │  MJPEG video stream      │ │
│  │  (UI, TTS)      │◄─┤  (predictions) │  │  /video_feed             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ HTTP / WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Flask + SocketIO Server (app.py)                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  generate_frames()                                                │  │
│  │    → open_camera() → cv2.VideoCapture                              │  │
│  │    → MediaPipe Hands (landmarks)                                   │  │
│  │    → Feature building (84-D skeleton + 84-D keras)                  │  │
│  │    → predict_with_skeleton_model() / predict_with_keras_landmark()  │  │
│  │    → get_combined_prediction()                                     │  │
│  │    → Temporal smoothing → socketio.emit('prediction')             │  │
│  │    → Encode frame → yield MJPEG                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  Model: indian_sign_model.h5 (Keras only, for prediction)                │
└─────────────────────────────────────────────────────────────────────────┘
```

**Components:**

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| **Frontend** | HTML, JS, Bootstrap, Socket.IO client | Page UI, video display, receive predictions, show text, TTS (Speak All / Pause / Clear). |
| **Backend** | Flask, Flask-SocketIO | Serve `/`, `/video_feed`, `/api/status`; WebSocket server for `prediction` events. |
| **Vision** | OpenCV, MediaPipe | Camera capture, frame resize (e.g. 640×480), hand detection, 21 landmarks per hand. |
| **ML** | TensorFlow/Keras | Load `.h5`; run inference on 84-D vectors; temporal smoothing of predictions. |

---

## 4. How It Works (Data Flow)

### 4.1 Frame Loop (High Level)

1. **Capture:** Read frame from camera (flip horizontally for mirror view).
2. **Detect hands:** `MediaPipe Hands` on RGB frame → `multi_hand_landmarks` (up to 2 hands).
3. **Per hand:**  
   - Draw landmarks.  
   - Build **Keras** 42-D (pixel landmarks → relative to wrist → normalized).
4. **Combine hands:** 42 + 42 or 42 + zero-padding → **84-D** for Keras.
5. **Predict:** `keras_model.predict` on 84-D (Keras model only).
6. **Temporal smoothing:**  
   - Only consider predictions with confidence ≥ 0.75.  
   - Keep last 5 predictions; require 60% consistency (e.g. 3/5 same).  
   - Emit via `socketio.emit('prediction', {...})` only when stable and different from last emitted.
7. **Special rule:** If the stable gesture is **’C’** (left-hand open palm), emit **space** for the text area.
8. **Encode:** Frame → JPEG → MJPEG chunk; yield to `/video_feed`.

Prediction is run only every **N-th frame** (e.g. every 3rd) to reduce CPU/TensorFlow load while keeping smooth video.

### 4.2 Output to the User

- **Video:** Continuous MJPEG stream at `/video_feed` (mirrored, with landmarks and bounding boxes).
- **Text:** Characters (or space) sent over WebSocket as `prediction` events; the frontend appends them to the “Recognized text” area.
- **Stability:** Shown on the frame as “Stability: k/5” so users see when the system is about to commit a character.

---

## 5. Getting Output

### 5.1 Running the Application

#### Option A: Docker Deployment (Recommended for Easy Setup)

1. **Prerequisites:** Docker and Docker Compose installed.

2. **Place model file:**
   - Copy `indian_sign_model.h5` to `./model/` directory.

3. **Build and run:**
   ```bash
   docker-compose up --build
   ```

4. **Access:** Open http://localhost:5000 in your browser.

See **DOCKER.md** for detailed Docker deployment instructions, camera access setup, and troubleshooting.

#### Option B: Local Python Installation

1. **Environment:**  
   - Python 3.8+.  
   - Create and activate venv, then:
   ```bash
   pip install -r requirements.txt
   ```
   (Or use `requirements_venv.txt` if that matches your environment.)

2. **Model:**  
   - Place the Keras model as `./model/indian_sign_model.h5` (or one of the paths in `KERAS_MODEL_PATHS` in `app.py`). Prediction uses only this model.

3. **Start server:**
   ```bash
   python app.py
   ```
   Server runs at **http://0.0.0.0:5000** (or http://localhost:5000).

4. **Browser:** Open http://localhost:5000, allow camera access. You should see:
   - Left: live video with hand landmarks and bounding boxes.
   - Right: “Recognized text” area and controls (Speak All, Pause, Clear).

### 5.2 Interpreting Output

- **Live text:** Characters (and spaces) appear in the right panel as you sign. Space is inserted when the system detects the left-hand “C” (open palm) gesture.
- **Confidence:** The UI shows a confidence bar; the backend only emits after stability and minimum confidence (0.75).
- **Speak All:** Reads the current recognized text using the browser’s speech synthesis (language/speed selectors apply).
- **Clear:** Clears the recognized text area.

### 5.3 Programmatic Output

- **HTTP:**  
  - `GET /api/status` → JSON with `skeleton_model`, `keras_model`, `tensorflow` (booleans).  
  - `GET /video_feed` → MJPEG stream.
- **WebSocket (Socket.IO):**  
  - Event `prediction` with payload, e.g.:
    - `text`: character (or space).
    - `confidence`: 0–1.
    - `model`: `"ISL-Keras"` (only model used).
    - `num_hands`, `stability`, `gesture` (internal gesture label).

You can build your own client that listens to `prediction` and uses `text` (and optionally `confidence`) as the main output.

---

## 6. Training Pipeline

**Script:** `train_indian_model.py`

### 6.1 Dataset

- **Layout:** Directory `Indian/` with one folder per class (e.g. `1`, `2`, … `9`, `A`, `B`, … `Z`).
- **Content:** Each folder contains images (e.g. `.jpg`) of that sign.
- **Usage:** Script finds `Indian/` (or path from `POSSIBLE_DATASET_PATHS`), loads all images, runs MediaPipe in **static_image_mode**, and extracts:
  - **Keras-style** 84-D (preprocessed landmarks),
  - **Skeleton-style** 84-D (bbox-normalized, same as `model.p`).

### 6.2 Training Steps

1. **Load dataset:** Scan class folders → list of image paths and labels.
2. **Extract features:** For each image, run MediaPipe; for each detected hand, compute 42-D skeleton and 42-D preprocessed; combine to 84-D (padding with zeros if one hand). Discard images where no hand is detected.
3. **Split:** Train / validation / test (e.g. 70% / 20% / 10%), stratified.
4. **Keras model:** Build Dense(256)→BN→Dropout(0.3) → Dense(128)→BN→Dropout(0.3) → Dense(64)→BN→Dropout(0.2) → Dense(35, softmax). Train with Adam, categorical cross-entropy, validation-based checkpointing and early stopping.
5. **Checkpoints:** Best model (by validation accuracy), best weights, and periodic checkpoints (e.g. every 5 epochs) saved under `checkpoints/`.
6. **Skeleton model:** Train a RandomForest (e.g. 400 trees) on the **skeleton-style** 84-D features and labels; save as `model.p` (same format as `app.py` expects).
7. **Metadata:** Save `model_metadata.json` (class names, input_dim, etc.) next to the Keras model.

### 6.3 Output of Training

- **Keras:** `model/indian_sign_model.h5` (or path set in script) and `checkpoints/best_model_<timestamp>.h5`.
- **Skeleton:** `model/model.p`.
- **Metadata:** `model/model_metadata.json`.
- **Plots:** e.g. `checkpoints/training_history_<timestamp>.png`.

After training, point `app.py` at these files (see [Configuration & Paths](#9-configuration--paths)) and restart the app to use the new models.

---

## 7. Project Structure

```
sign-to-text-and-speech/
├── app.py                    # Main Flask app: camera, MediaPipe, Keras model, WebSocket, MJPEG
├── train_indian_model.py     # Dataset load, feature extraction, Keras + RandomForest training
├── profile_models.py         # Inference-time profiling (skeleton vs Keras)
├── requirements.txt         # Pip dependencies (Flask, OpenCV, MediaPipe, TensorFlow, etc.)
├── requirements_venv.txt    # Alternative/full venv snapshot
├── README.md                 # User-facing overview and quick start
├── DOCUMENTATION.md          # This file
├── DOCKER.md                 # Docker deployment guide
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Docker Compose configuration
└── .dockerignore             # Files excluded from Docker build
├── templates/
│   └── index.html            # Single-page UI: video, Socket.IO, text area, TTS, Clear
├── model/                    # Not in git (large/binary); expected contents:
│   ├── model.p               # Pickled RandomForest + metadata
│   ├── indian_sign_model.h5  # Keras model
│   └── model_metadata.json   # Class names, input_dim, etc.
├── checkpoints/              # Training checkpoints and plots (optional)
├── Indian/                   # Dataset: class folders 1..9, A..Z (optional, often not in git)
├── dataset_classes_visualization.png
├── venv_activation.ps1       # Helper to activate venv (e.g. Windows)
└── .gitignore
```

---

## 8. API Reference

### HTTP

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main HTML page (SignSpeak UI). |
| `/video_feed` | GET | MJPEG stream of the camera with overlaid landmarks and boxes. |
| `/api/status` | GET | JSON: `skeleton_model`, `keras_model`, `tensorflow` (booleans). |

### WebSocket (Socket.IO)

- **Connection:** Client connects to same origin (e.g. `http://localhost:5000`); Socket.IO auto-connects.
- **Server → Client event: `prediction`**
  - Emitted when a gesture is stable and above confidence threshold.
  - Payload example:
    ```json
    {
      "text": "A",
      "confidence": 0.92,
      "model": "ISL-Keras",
      "num_hands": 1,
      "stability": "80%",
      "gesture": "A"
    }
    ```
  - `text`: character to display (or space for word separation).  
  - `gesture`: internal label (e.g. `"C"` when `text` is space).

---

## 9. Configuration & Paths

### 9.1 Model Paths (app.py)

- **Skeleton:** `SKELETON_MODEL_PATH = './model/model.p'`.
- **Keras:** `KERAS_MODEL_PATHS` — list of paths tried in order; first existing file is loaded. Typical entries:
  - Absolute path to `indian_sign_model.h5` on your machine.
  - `./model/indian_sign_model.h5`.
  - `./checkpoints/best_model_*.h5`.

### 9.2 Labels

- **Skeleton (model.p):** Uses `labels_dict` in `app.py` (0→’A’ … 25→’Z’) for the **old** 26-class mapping; the **trained** skeleton model in `train_indian_model.py` uses 35 classes and stores its own `class_names` in the pickle. The app’s `labels_dict` is still used for decoding the predicted index when the pickle format matches.
- **Keras:** `keras_alphabet = ['1','2',…,'9'] + list(string.ascii_uppercase)` (35 classes). Index 0–8 → digits, 9–34 → A–Z.

### 9.3 Special Gestures

- **Left-hand open palm (“C”)** → emitted as **space** in `text` for word separation.
- **Right-hand open palm** → digit **’5’**.

### 9.4 Tunables in the Frame Loop

- `prediction_skip_frames`: run prediction every N-th frame (e.g. 3).
- `HISTORY_SIZE`: number of recent predictions for stability (e.g. 5).
- `CONSISTENCY_THRESHOLD`: fraction that must agree (e.g. 0.6).
- `MIN_CONFIDENCE`: minimum confidence to consider (e.g. 0.75).
- Keras weight in `get_combined_prediction`: 1.2.

---

## 10. Docker Deployment

For easy deployment on any system, SignSpeak can be run using Docker. This eliminates the need to install Python dependencies manually.

### Quick Start

1. **Place model file** in `./model/indian_sign_model.h5`

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Access:** http://localhost:5000

### Files Created

- **Dockerfile**: Multi-stage build with system dependencies for OpenCV and MediaPipe
- **docker-compose.yml**: Easy orchestration with camera access and volume mounts
- **.dockerignore**: Excludes unnecessary files from Docker build context
- **DOCKER.md**: Complete Docker deployment guide

### Key Features

- **Camera access**: Configured for Linux (`/dev/video0`), Windows, and macOS
- **Model mounting**: Models mounted as volumes (no rebuild needed to update)
- **Health checks**: Built-in container health monitoring
- **Resource limits**: Configurable CPU/memory limits
- **Production-ready**: Optimized for deployment

See **DOCKER.md** for:
- Detailed setup instructions for each platform
- Camera access troubleshooting
- Production deployment recommendations
- Advanced configuration options

---

## 11. Troubleshooting

| Issue | What to check |
|-------|----------------|
| **Camera not opening** | `open_camera()` uses `CAP_DSHOW` on Windows; try different camera index (0, 1, 2). Ensure no other app is using the camera. |
| **“No Models” / model not loaded** | Ensure `model/model.p` exists and `model/indian_sign_model.h5` (or one of `KERAS_MODEL_PATHS`) exists. Check console at startup for “[OK] … loaded” or “[WARNING] … not found”. |
| **TensorFlow/Keras errors** | Match Python and TensorFlow versions to `requirements.txt`. If loading `.h5` fails (e.g. DepthwiseConv2D), the app tries a custom `CompatibleDepthwiseConv2D`; ensure you’re not using an old CNN `.h5` that doesn’t match the current Dense landmark model. |
| **Predictions not updating in UI** | Confirm WebSocket is connected (browser dev tools → Network/WS). Check that confidence and stability thresholds are met; try signing more clearly and holding the gesture longer. |
| **Wrong character** | Lighting, hand size, and distance to camera affect MediaPipe and thus features. Ensure hand is fully in frame and matches training conditions. Retrain with more data if needed. |
| **Slow FPS** | Reduce resolution in `open_camera()`, or increase `prediction_skip_frames` in `generate_frames()`. Use GPU-backed TensorFlow if available. Run `profile_models.py` to compare skeleton vs Keras inference time. |

---

## Summary

- **Model:** Only the Keras Dense model (84→256→128→64→35) is used for prediction, on 84-D hand-landmark features (preprocessed, two-hand support via padding/concatenation).
- **Architecture:** Flask + SocketIO server runs a single frame generator that does camera → MediaPipe → features → Keras model → temporal smoothing → WebSocket + MJPEG.
- **Output:** Real-time characters (and space) in the web UI and via WebSocket `prediction` events; optional TTS with “Speak All”.
- **Training:** `train_indian_model.py` builds both the Keras and skeleton models from an `Indian/` image dataset using MediaPipe-derived 84-D features and saves them plus metadata for use by `app.py`.

For quick start and user-facing features, see **README.md**. For implementation details, model layers, and APIs, this **DOCUMENTATION.md** is the reference.
