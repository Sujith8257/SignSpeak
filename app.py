# python -m venv venv
# venv\Scripts\activate  (Windows)
# pip install -r requirements.txt

import os
import sys
import warnings
import logging
import platform
import copy
import itertools
import string
import json
import time
from collections import Counter

# Fix Windows console encoding for Unicode characters
if platform.system() == 'Windows':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Fallback for older Python versions
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import cv2
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow import keras

from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit

# -----------------------------
#  SUPPRESS WARNINGS
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

logging.getLogger('absl').setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# TensorFlow availability flag (needed because keras comes from TF)
try:
    import tensorflow as tf  # noqa: F401
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] TensorFlow not installed. Keras model (.h5) will not be available.")

# -----------------------------
#   FLASK + SOCKET
# -----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
#   MODEL PATHS
# -----------------------------
SKELETON_MODEL_PATH = './model/model.p'
MODEL_METADATA_PATH = './model/model_metadata.json'
# Try multiple possible Keras model file names (prioritize new trained model)
KERAS_MODEL_PATHS = [
    r'D:\sign-to-text-and-speech\model\indian_sign_model.h5',  # New trained model (absolute path)
    './model/indian_sign_model.h5',  # New trained model (relative path)
    './checkpoints/best_model_20251228_173824.h5',  # Latest checkpoint
    r'D:\sign-to-text-and-speech\model\indian_sign_weights.h5',  # Legacy weights (fallback)
    './model/indian_sign_weights.h5'  # Legacy weights (relative fallback)
]

# -----------------------------
#   LOAD SKELETON MODEL (model.p)
# -----------------------------
skeleton_model = None
try:
    model_dict = pickle.load(open(SKELETON_MODEL_PATH, 'rb'))
    skeleton_model = model_dict['model']
    print("[OK] Skeleton model (model.p) loaded successfully")
except FileNotFoundError:
    print(f"[WARNING] ISL Skeleton model not found at {SKELETON_MODEL_PATH}")
except Exception as e:
        print(f"[ERROR] Error loading ISL skeleton model: {e}")

# -----------------------------
#   LOAD KERAS LANDMARK MODEL (.h5)
# -----------------------------
keras_model = None
keras_model_path_used = None

if TENSORFLOW_AVAILABLE:
    # Try to find and load a Keras model from available paths
    for model_path in KERAS_MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                # Try loading with compile=False first to avoid compilation issues
                try:
                    keras_model = keras.models.load_model(model_path, compile=False)
                    keras_model_path_used = model_path
                    print(f"[OK] ISL Keras landmark model ({os.path.basename(model_path)}) loaded successfully")
                    break
                except Exception as load_error:
                    # Handle version compatibility issues with DepthwiseConv2D groups parameter
                    if 'groups' in str(load_error) or 'DepthwiseConv2D' in str(load_error):
                        try:
                            # Create a custom DepthwiseConv2D that ignores the groups parameter
                            from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D
                            
                            class CompatibleDepthwiseConv2D(BaseDepthwiseConv2D):
                                def __init__(self, *args, **kwargs):
                                    # Remove 'groups' parameter if present (not supported in older TF versions)
                                    kwargs.pop('groups', None)
                                    super().__init__(*args, **kwargs)
                            
                            # Try loading with the custom object
                            keras_model = keras.models.load_model(
                                model_path, 
                                compile=False,
                                custom_objects={'DepthwiseConv2D': CompatibleDepthwiseConv2D}
                            )
                            keras_model_path_used = model_path
                            print(f"[OK] ISL Keras landmark model ({os.path.basename(model_path)}) loaded successfully (with compatibility fix)")
                            break
                        except Exception as e2:
                            print(f"[WARNING] Failed to load {model_path}: {e2}")
                            continue
                    else:
                        print(f"[WARNING] Failed to load {model_path}: {load_error}")
                        continue
            except Exception as e:
                print(f"[WARNING] Error loading {model_path}: {e}")
                continue
    
    if keras_model is None:
        print("[WARNING] No ISL Keras model found. Tried the following paths:")
        for path in KERAS_MODEL_PATHS:
            exists = "[OK]" if os.path.exists(path) else "[NOT FOUND]"
            print(f"   {exists} {path}")
        print("   Continuing without ISL Keras model - ISL skeleton model will be used.")
else:
        print("[WARNING] TensorFlow not available. ISL Keras model cannot be loaded.")

# -----------------------------
#   LABELS / ALPHABET
# -----------------------------
# Labels for ISL skeleton model (scikit-learn) - Indian Sign Language letters (A-Z)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Alphabet for ISL Keras landmark model (digits 1-9 + letters A–Z)
keras_alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)

# Special gesture mappings:
# - 'C' (detected from left hand open palm) → SPACE (for word separation)
# - Right hand open palm → '5' (normal digit)
# Note: Left vs right hand produces different landmark patterns, 
#       creating this natural distinction for special functions


def verify_model_metadata_matches_alphabet():
    """Load model metadata and compare class_names to keras_alphabet for quick sanity check."""
    if not os.path.exists(MODEL_METADATA_PATH):
        print(f"[WARNING] Metadata not found at {MODEL_METADATA_PATH}")
        return

    try:
        with open(MODEL_METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        meta_classes = metadata.get('class_names', [])

        if meta_classes == keras_alphabet:
            print(f"[OK] Metadata classes match keras_alphabet ({len(meta_classes)} classes)")
            return

        # Length mismatch or ordering mismatch
        if len(meta_classes) != len(keras_alphabet):
            print(f"[WARNING] Metadata class count ({len(meta_classes)}) differs from keras_alphabet ({len(keras_alphabet)})")
        else:
            # Same length but maybe different order/content
            diff_meta = [c for c in meta_classes if c not in keras_alphabet]
            diff_alphabet = [c for c in keras_alphabet if c not in meta_classes]
            if diff_meta or diff_alphabet:
                print(f"[WARNING] Metadata/keras_alphabet mismatch. Extra in metadata: {diff_meta}. Missing in metadata: {diff_alphabet}")
            else:
                print("[WARNING] Metadata classes order differs from keras_alphabet")
    except Exception as e:
        print(f"[WARNING] Could not verify metadata vs keras_alphabet: {e}")


# Run the verification once on startup
verify_model_metadata_matches_alphabet()


@app.route('/')
def lander():
    """Public landing page with marketing content; links through to /app."""
    return render_template('lander.html')


@app.route('/app')
def index():
    """Main SignSpeak application UI."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Return model status for frontend badge."""
    return jsonify({
        'skeleton_model': skeleton_model is not None,
        'keras_model': keras_model is not None,
        'tensorflow': TENSORFLOW_AVAILABLE
    })


@socketio.on('connect')
def handle_connect():
    print("Client connected")


# ----------------------------------------------------
#   FIXED CAMERA OPENING (WINDOWS → CAP_DSHOW)
# ----------------------------------------------------
def open_camera():
    is_windows = platform.system().lower() == "windows"
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    
    # Try multiple backends in order of preference
    backends_to_try = []
    
    if is_windows and not is_docker:
        # Native Windows: try DirectShow first
        try:
            backends_to_try.append(cv2.CAP_DSHOW)
        except AttributeError:
            pass
        try:
            backends_to_try.append(cv2.CAP_MSMF)
        except AttributeError:
            pass
    elif is_docker:
        # Docker: try V4L2 (Linux) or MSMF if available
        try:
            backends_to_try.append(cv2.CAP_V4L2)
        except AttributeError:
            pass
        try:
            backends_to_try.append(cv2.CAP_MSMF)
        except AttributeError:
            pass
    
    # Always try default backend as fallback
    backends_to_try.append(None)
    
    # Try each backend until one works
    for backend in backends_to_try:
        backend_name = "CAP_DSHOW" if backend == cv2.CAP_DSHOW else \
                      "CAP_MSMF" if backend == cv2.CAP_MSMF else \
                      "CAP_V4L2" if backend == cv2.CAP_V4L2 else \
                      "default"
        print(f"Trying to open camera at index 0 using backend: {backend_name}")
        
        if backend is not None:
            cap = cv2.VideoCapture(0, backend)
        else:
            cap = cv2.VideoCapture(0)

    if cap.isOpened():
        # Set camera properties for better compatibility
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Try reading a few test frames to ensure camera is working
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[OK] Camera test frame {i+1}/5 successful")
                time.sleep(0.1)
            else:
                print(f"[WARNING] Camera test frame {i+1}/5 failed")
                time.sleep(0.2)
        
        # Final validation with latest frame
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print("[OK] Camera opened successfully at index 0")
            return cap
        else:
            print("[WARNING] Camera opened but frames are invalid")
            cap.release()
            return None
    else:
        print("[ERROR] Could not open camera at index 0")
        return None


# ----------------------------------------------------
#   LANDMARK HELPERS (FROM REFERENCE SCRIPT)
# ----------------------------------------------------
def calc_landmark_list(image, landmarks):
    """Convert MediaPipe landmarks to pixel coordinates list."""
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for lm in landmarks.landmark:
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """Make landmarks relative to first point and normalize (reference logic)."""
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list))) or 1.0

    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


# ----------------------------------------------------
#   PREDICTION FUNCTIONS
# ----------------------------------------------------
def predict_with_skeleton_model(data_aux):
    """Make prediction using ISL skeleton (MediaPipe landmarks) model"""
    if skeleton_model is None:
        return None, 0.0
    
    try:
        prediction = skeleton_model.predict([np.asarray(data_aux)])
        prediction_proba = skeleton_model.predict_proba([np.asarray(data_aux)])
        confidence = max(prediction_proba[0])
        predicted_label = labels_dict.get(int(prediction[0]), '?')
        return predicted_label, float(confidence)
    except Exception as e:
        print(f"ISL Skeleton prediction error: {e}")
        return None, 0.0


def predict_with_cnn_model(frame, hand_region=None):
    """(Deprecated stub) kept for backward compatibility."""
    return None, 0.0


def predict_with_keras_landmark_model(pre_processed_landmark_list):
    """Make prediction using ISL Keras landmark model."""
    if keras_model is None:
        return None, 0.0

    try:
        # Optimize: Use numpy array directly instead of creating DataFrame
        input_array = np.array(pre_processed_landmark_list, dtype=np.float32).reshape(1, -1)
        predictions = keras_model.predict(input_array, verbose=0)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        if 0 <= predicted_class < len(keras_alphabet):
            label = keras_alphabet[predicted_class]
        else:
            label = '?'

        return label, confidence
    except Exception as e:
        print(f"ISL Keras landmark model prediction error: {e}")
        return None, 0.0


def get_combined_prediction(skeleton_pred, skeleton_conf, keras_pred, keras_conf):
    """
    Combine predictions from both ISL models.
    Prioritizes Keras model (trained on Indian Sign Language dataset) when available.
    """
    # If only one model has prediction, use that
    if skeleton_pred is None and keras_pred is not None:
        return keras_pred, keras_conf, "ISL-Keras"
    if keras_pred is None and skeleton_pred is not None:
        return skeleton_pred, skeleton_conf, "ISL-Skeleton"
    if skeleton_pred is None and keras_pred is None:
        return None, 0.0, None
    
    # Both models have predictions - PRIORITIZE KERAS (it's trained on your dataset)
    # Keras model is more reliable for Indian Sign Language
    keras_weight = 1.2  # 20% boost for keras model (prioritize it)
    
    weighted_keras_conf = keras_conf * keras_weight
    
    if weighted_keras_conf >= skeleton_conf:
        return keras_pred, keras_conf, "ISL-Keras"
    else:
        return skeleton_pred, skeleton_conf, "ISL-Skeleton"


# ----------------------------------------------------
#   MAIN FRAME GENERATOR
# ----------------------------------------------------
def generate_frames():
    cap = open_camera()

    # If camera not found → show placeholder forever
    if cap is None:
        print("Showing error placeholder frame...")

        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Not Available", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', error_frame)
        error_bytes = buffer.tobytes()

        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   error_bytes + b'\r\n')

    # --------------------------
    #  MEDIAPIPE INIT
    # --------------------------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    prediction_skip_frames = 3  # Process prediction every 3rd frame (reduce TensorFlow overhead)
    last_prediction = None
    last_confidence = 0.0
    
    # Temporal smoothing for stable predictions
    prediction_history = []  # Store last N predictions
    HISTORY_SIZE = 5  # Number of frames to check for consistency
    CONSISTENCY_THRESHOLD = 0.6  # 60% of frames must agree (3 out of 5)
    MIN_CONFIDENCE = 0.75  # Minimum confidence to consider a prediction
    last_emitted_prediction = None  # Last prediction sent to frontend

    # ----------------------------------------------------
    #   FRAME LOOP
    # ----------------------------------------------------
    while True:
        try:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("[WARNING] Failed to read frame from camera")
                time.sleep(0.1)  # Longer delay to allow camera to recover
                continue
            
            # Validate frame shape and data
            if frame.size == 0 or len(frame.shape) != 3:
                print("[WARNING] Invalid frame shape received")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Track all predictions for this frame
            all_predictions = []
            
            # OPTIMIZATION: Only run predictions every N frames to reduce TensorFlow overhead
            should_predict = (frame_count % prediction_skip_frames == 0)
            
            if results.multi_hand_landmarks:
                num_hands_detected = len(results.multi_hand_landmarks)
                if num_hands_detected > 1 and frame_count % 50 == 0:  # Log less frequently
                    print(f"[OK] Detected {num_hands_detected} hands in frame")
                
                # Get hand labels (Left/Right) if available
                hand_labels = []
                if results.multi_handedness:
                    for hand_info in results.multi_handedness:
                        hand_labels.append(hand_info.classification[0].label)
                else:
                    # If hand labels not available, assign based on position
                    hand_labels = ['Hand'] * len(results.multi_hand_landmarks)

                # Collect features from ALL hands first (for two-hand model)
                skeleton_features_list = []
                keras_features_list = []
                hand_bboxes = []
                
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Initialize data structures for THIS hand only
                    data_aux = []
                    x_ = []
                    y_ = []
                    
                    # Get hand chirality (Left or Right)
                    is_left_hand = False
                    if hand_idx < len(hand_labels):
                        is_left_hand = (hand_labels[hand_idx] == 'Left')

                    # Draw landmarks on frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # --------------------------
                    #  SKELETON MODEL FEATURES (with chirality correction)
                    # --------------------------
                    # Extract x and y coordinates for this hand
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    # Normalize coordinates relative to this hand's bounding box
                    if len(x_) > 0 and len(y_) > 0:
                        min_x = min(x_)
                        max_x = max(x_)
                        min_y = min(y_)
                        
                        # For left hands, mirror X coordinates so they look like right hands
                        for lm in hand_landmarks.landmark:
                            if is_left_hand:
                                # Mirror: flip x around center of hand bounding box
                                mirrored_x = max_x - (lm.x - min_x)
                                data_aux.append(mirrored_x - min_x)
                            else:
                                data_aux.append(lm.x - min_x)
                            data_aux.append(lm.y - min_y)
                        
                        skeleton_features_list.append(data_aux)

                    # --------------------------
                    #  BBOX FOR VISUALIZATION
                    # --------------------------
                    if len(x_) > 0 and len(y_) > 0:
                        x1 = int(min(x_) * W) - 10
                        y1 = int(min(y_) * H) - 10
                        x2 = int(max(x_) * W) + 10
                        y2 = int(max(y_) * H) + 10

                        # Ensure bbox is within frame bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(W, x2)
                        y2 = min(H, y2)
                        
                        hand_bboxes.append((x1, y1, x2, y2))

                        # --------------------------
                        #  KERAS LANDMARK FEATURES
                        # --------------------------
                        landmark_list = calc_landmark_list(frame, hand_landmarks)
                        pre_processed_landmarks = pre_process_landmark(landmark_list)
                        keras_features_list.append(pre_processed_landmarks)

                # Combine features for two-hand model (pad with zeros if only one hand)
                def combine_two_hands(per_hand_list):
                    """Combine features from multiple hands, padding with zeros if only one hand."""
                    if not per_hand_list:
                        return None
                    if len(per_hand_list) == 1:
                        # Pad with zeros to make 84 features (42 per hand)
                        return per_hand_list[0] + [0.0] * len(per_hand_list[0])
                    elif len(per_hand_list) >= 2:
                        # Combine first two hands
                        return per_hand_list[0] + per_hand_list[1]
                    return per_hand_list[0]

                combined_skeleton_features = combine_two_hands(skeleton_features_list)
                combined_keras_features = combine_two_hands(keras_features_list)

                # -------------------------------------
                #  GET PREDICTIONS FROM BOTH MODELS (using combined features)
                # Run on prediction frames for balance between speed and accuracy
                # Prioritizes Keras model (trained on your Indian Sign Language dataset)
                # -------------------------------------
                if should_predict and combined_skeleton_features and combined_keras_features:
                    skeleton_pred, skeleton_conf = predict_with_skeleton_model(combined_skeleton_features)
                    keras_pred, keras_conf = predict_with_keras_landmark_model(combined_keras_features)

                    # Combine predictions (ensemble with preference for Keras model)
                    predicted_character, confidence, model_used = get_combined_prediction(
                        skeleton_pred, skeleton_conf, keras_pred, keras_conf
                    )

                    # Store prediction (combined for all hands)
                    if predicted_character is not None:
                        last_prediction = predicted_character
                        last_confidence = confidence
                        all_predictions.append({
                            'character': predicted_character,
                            'confidence': confidence,
                            'model': model_used,
                            'hand_label': 'Combined' if num_hands_detected > 1 else hand_labels[0] if hand_labels else 'Hand',
                            'bbox': hand_bboxes[0] if hand_bboxes else (0, 0, 0, 0)
                        })
                        if num_hands_detected > 1 and frame_count % 30 == 0:  # Log less frequently
                            print(f"  Combined prediction from {num_hands_detected} hands: {predicted_character} (conf: {confidence:.2f})")

                    # Store prediction (combined for all hands)
                    if predicted_character is not None:
                        last_prediction = predicted_character
                        last_confidence = confidence
                        all_predictions.append({
                            'character': predicted_character,
                            'confidence': confidence,
                            'model': model_used,
                            'hand_label': 'Combined' if num_hands_detected > 1 else hand_labels[0] if hand_labels else 'Hand',
                            'bbox': hand_bboxes[0] if hand_bboxes else (0, 0, 0, 0)
                        })
                        if num_hands_detected > 1 and frame_count % 30 == 0:  # Log less frequently
                            print(f"  Combined prediction from {num_hands_detected} hands: {predicted_character} (conf: {confidence:.2f})")
                elif last_prediction is not None:
                    # Between prediction frames, use the last prediction
                    all_predictions.append({
                        'character': last_prediction,
                        'confidence': last_confidence,
                        'model': 'ISL-Keras',
                        'hand_label': 'Combined' if num_hands_detected > 1 else hand_labels[0] if hand_labels else 'Hand',
                        'bbox': hand_bboxes[0] if hand_bboxes else (0, 0, 0, 0)
                    })
                    
                    # Draw bounding boxes and predictions for each hand
                    for hand_idx, (x1, y1, x2, y2) in enumerate(hand_bboxes):
                        # Use different colors for left/right hands
                        if hand_idx == 0:
                            color = (0, 255, 0) if last_confidence > 0.7 else (0, 165, 255) if last_confidence > 0.5 else (0, 0, 255)
                        else:
                            color = (255, 0, 0) if last_confidence > 0.7 else (255, 165, 0) if last_confidence > 0.5 else (255, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Label with hand identifier if multiple hands
                        if last_prediction is not None:
                            label_text = f"{last_prediction} ({last_confidence*100:.1f}%)"
                            if num_hands_detected > 1:
                                hand_label = hand_labels[hand_idx] if hand_idx < len(hand_labels) else f'Hand {hand_idx + 1}'
                                label_text = f"[{hand_label}] {label_text}"
                            
                            cv2.putText(
                                frame,
                                label_text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2
                            )
                    
                # Draw bounding boxes and predictions for prediction frames
                if should_predict and all_predictions:
                    for hand_idx, (x1, y1, x2, y2) in enumerate(hand_bboxes):
                        # Use different colors for left/right hands
                        if hand_idx == 0:
                            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.5 else (0, 0, 255)
                        else:
                            color = (255, 0, 0) if confidence > 0.7 else (255, 165, 0) if confidence > 0.5 else (255, 0, 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Label with hand identifier if multiple hands
                        if predicted_character is not None:
                            label_text = f"{predicted_character} ({confidence*100:.1f}%)"
                            if num_hands_detected > 1:
                                hand_label = hand_labels[hand_idx] if hand_idx < len(hand_labels) else f'Hand {hand_idx + 1}'
                                label_text = f"[{hand_label}] {label_text}"
                            
                            cv2.putText(
                                frame,
                                label_text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2
                            )

            # Send predictions via WebSocket
            # Since we combine features from all hands, we have one prediction
            if all_predictions and len(all_predictions) > 0:
                single_prediction = all_predictions[0]
                predicted_char = single_prediction['character']
                pred_confidence = single_prediction['confidence']
                
                # Only consider predictions with high enough confidence
                if pred_confidence >= MIN_CONFIDENCE:
                    # Add to prediction history
                    prediction_history.append(predicted_char)
                    
                    # Keep only last N predictions
                    if len(prediction_history) > HISTORY_SIZE:
                        prediction_history.pop(0)
                    
                    # Check if we have enough history to make a decision
                    if len(prediction_history) >= HISTORY_SIZE:
                        # Count occurrences of each prediction
                        counts = Counter(prediction_history)
                        most_common_pred, count = counts.most_common(1)[0]
                        
                        # Check if the most common prediction meets consistency threshold
                        consistency_ratio = count / len(prediction_history)
                        
                        if consistency_ratio >= CONSISTENCY_THRESHOLD:
                            # Only emit if it's different from last emitted (avoid spam)
                            if most_common_pred != last_emitted_prediction:
                                # Convert 'C' (left hand open palm) to SPACE for word separation
                                display_char = '   ' if most_common_pred == 'C' else most_common_pred
                                
                                hands_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                                try:
                                    socketio.emit(
                                        'prediction',
                                        {
                                            'text': display_char, 
                                            'confidence': float(pred_confidence),
                                            'model': single_prediction['model'],
                                            'num_hands': hands_count,
                                            'stability': f'{consistency_ratio*100:.0f}%',
                                            'gesture': most_common_pred  # Keep original for debugging
                                        }
                                    )
                                    last_emitted_prediction = most_common_pred
                                    gesture_label = 'SPACE' if most_common_pred == 'C' else most_common_pred
                                    print(f"[STABLE] Emitted: {gesture_label} (stability: {consistency_ratio*100:.0f}%)")
                                except Exception as ws_error:
                                    print(f"[WARNING] WebSocket emit error: {ws_error}")
                else:
                    # Low confidence - likely transitioning, clear history
                    if len(prediction_history) > 0:
                        print(f"[TRANSITION] Low confidence ({pred_confidence:.2f}), clearing history")
                        prediction_history.clear()
                        last_emitted_prediction = None  # Reset so same gesture can be detected next time
            else:
                # No hands detected - clear history
                if len(prediction_history) > 0:
                    prediction_history.clear()
                    last_emitted_prediction = None  # Reset so same gesture can be detected next time
            
            # Visual feedback for stability
            if len(prediction_history) > 0:
                stability = len(set(prediction_history)) == 1  # All predictions same = stable
                stability_text = f"Stability: {len(prediction_history)}/{HISTORY_SIZE}"
                color = (0, 255, 0) if stability else (0, 165, 255)
                cv2.putText(frame, stability_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Encode frame
            ret_encode, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret_encode or buffer is None:
                print("[WARNING] Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

            frame_count += 1
            if frame_count % 300 == 0:  # Log less frequently (every 300 frames instead of 100)
                print(f"Processed {frame_count} frames (FPS optimized with frame skipping)")

        except Exception as e:
            print("Error in generate_frames:", e)
            break

    cap.release()
    hands.close()


@app.route('/video_feed')
def video_feed():
    print("Video feed accessed from", request.remote_addr)
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    print("\n" + "="*50)
    print("   SignSpeak - Sign Language Detection")
    print("="*50)
    print(f"ISL Skeleton Model: {'[OK] Loaded' if skeleton_model else '[ERROR] Not Available'}")
    print(f"ISL Keras Landmark Model: {'[OK] Loaded' if keras_model else '[ERROR] Not Available'}")
    print(f"TensorFlow/Keras: {'[OK] Available' if TENSORFLOW_AVAILABLE else '[ERROR] Not Installed'}")
    print("="*50 + "\n")
    
    # Disable debug mode in Docker for security
    is_docker = os.environ.get('DOCKER_CONTAINER', 'false').lower() == 'true'
    debug_mode = not is_docker  # Debug only when NOT in Docker
    
    print(f"Running in Docker: {is_docker}")
    print(f"Debug mode: {debug_mode}\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=debug_mode, allow_unsafe_werkzeug=True)