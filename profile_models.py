"""
Profile both ISL models to measure inference time.
This helps determine if the Keras model is worth the computational overhead.
"""

import os
import sys
import warnings
import pickle
import time
import numpy as np
import string
import platform

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

from tensorflow import keras

# Fix Windows console encoding
if platform.system() == 'Windows':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Model paths
SKELETON_MODEL_PATH = './model/model.p'
KERAS_MODEL_PATHS = [
    r'D:\sign-to-text-and-speech\model\indian_sign_model.h5',
    './model/indian_sign_model.h5',
    './checkpoints/best_model_20251228_173824.h5',
]

# Labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}
keras_alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)

print("="*60)
print("   ISL MODEL PROFILING - Inference Time Comparison")
print("="*60)

# Load Skeleton Model
print("\n[1/3] Loading Skeleton Model (model.p)...")
skeleton_model = None
try:
    model_dict = pickle.load(open(SKELETON_MODEL_PATH, 'rb'))
    skeleton_model = model_dict['model']
    print("✓ Skeleton model loaded successfully")
except Exception as e:
    print(f"✗ Error loading skeleton model: {e}")

# Load Keras Model
print("[2/3] Loading Keras Landmark Model...")
keras_model = None
keras_model_path_used = None

for model_path in KERAS_MODEL_PATHS:
    if os.path.exists(model_path):
        try:
            keras_model = keras.models.load_model(model_path, compile=False)
            keras_model_path_used = model_path
            print(f"✓ Keras model loaded from: {os.path.basename(model_path)}")
            break
        except Exception as e:
            if 'groups' in str(e) or 'DepthwiseConv2D' in str(e):
                try:
                    from tensorflow.keras.layers import DepthwiseConv2D as BaseDepthwiseConv2D
                    
                    class CompatibleDepthwiseConv2D(BaseDepthwiseConv2D):
                        def __init__(self, *args, **kwargs):
                            kwargs.pop('groups', None)
                            super().__init__(*args, **kwargs)
                    
                    keras_model = keras.models.load_model(
                        model_path, 
                        compile=False,
                        custom_objects={'DepthwiseConv2D': CompatibleDepthwiseConv2D}
                    )
                    keras_model_path_used = model_path
                    print(f"✓ Keras model loaded from: {os.path.basename(model_path)} (with compatibility fix)")
                    break
                except Exception as e2:
                    continue
            continue

if keras_model is None:
    print("✗ Keras model not found at any path")

print("[3/3] Generating synthetic test data...")

# Generate test data (84 features for skeleton = 2 hands, 42 for keras)
num_tests = 100
skeleton_test_data = np.random.rand(num_tests, 84).astype(np.float32)  # 84 = 2 hands combined
keras_test_data = np.random.rand(num_tests, 84).astype(np.float32)      # 42 = pre-processed landmarks

print(f"✓ Generated {num_tests} test samples (84 features for skeleton, 42 for keras)")

print("\n" + "="*60)
print("   PROFILING RESULTS")
print("="*60)

# Profile Skeleton Model
if skeleton_model is not None:
    print("\n[SKELETON MODEL]")
    print("Running 1000 predictions...")
    
    start_time = time.time()
    for i in range(num_tests):
        prediction = skeleton_model.predict([skeleton_test_data[i]])
        pred_proba = skeleton_model.predict_proba([skeleton_test_data[i]])
    end_time = time.time()
    
    skeleton_total_time = end_time - start_time
    skeleton_avg_time = skeleton_total_time / num_tests
    skeleton_fps = 1.0 / skeleton_avg_time
    
    print(f"  Total time:     {skeleton_total_time:.3f} seconds")
    print(f"  Avg per prediction: {skeleton_avg_time*1000:.3f} ms")
    print(f"  Throughput:     {skeleton_fps:.1f} predictions/second")
else:
    print("\n[SKELETON MODEL] Not loaded, skipping profiling")
    skeleton_total_time = None
    skeleton_avg_time = None

# Profile Keras Model
if keras_model is not None:
    print("\n[KERAS MODEL]")
    print("Running 1000 predictions...")
    
    # Warm up GPU/TensorFlow
    _ = keras_model.predict(keras_test_data[0:1], verbose=0)
    
    start_time = time.time()
    for i in range(num_tests):
        predictions = keras_model.predict(keras_test_data[i:i+1], verbose=0)
    end_time = time.time()
    
    keras_total_time = end_time - start_time
    keras_avg_time = keras_total_time / num_tests
    keras_fps = 1.0 / keras_avg_time
    
    print(f"  Total time:     {keras_total_time:.3f} seconds")
    print(f"  Avg per prediction: {keras_avg_time*1000:.3f} ms")
    print(f"  Throughput:     {keras_fps:.1f} predictions/second")
else:
    print("\n[KERAS MODEL] Not loaded, skipping profiling")
    keras_total_time = None
    keras_avg_time = None

# Comparison
print("\n" + "="*60)
print("   COMPARISON & ANALYSIS")
print("="*60)

if skeleton_avg_time is not None and keras_avg_time is not None:
    slowdown_factor = keras_avg_time / skeleton_avg_time
    time_diff = (keras_avg_time - skeleton_avg_time) * 1000
    
    print(f"\nKeras is {slowdown_factor:.1f}x SLOWER than Skeleton model")
    print(f"Difference: {time_diff:.3f} ms per prediction")
    
    if slowdown_factor > 2:
        print("\n⚠️  KERAS MODEL IS SIGNIFICANTLY SLOWER")
        print("   Recommendation: Consider removing Keras model if accuracy difference is minimal")
    elif slowdown_factor > 1.5:
        print("\n⚠️  KERAS MODEL IS MODERATELY SLOWER")
        print("   Recommendation: Profile accuracy difference to justify overhead")
    else:
        print("\n✓ Performance difference is acceptable")
        print("   Recommendation: Keep both models if accuracy trade-off is worth it")
    
    # Frame processing impact at 30 FPS
    print(f"\n--- At 30 FPS (33.3ms per frame) ---")
    print(f"Skeleton model: {skeleton_avg_time*1000:.1f}ms ({(skeleton_avg_time*1000/33.3)*100:.1f}% of frame time)")
    print(f"Keras model:   {keras_avg_time*1000:.1f}ms ({(keras_avg_time*1000/33.3)*100:.1f}% of frame time)")

elif skeleton_avg_time is not None:
    print(f"\nSkeleton model: {skeleton_avg_time*1000:.3f} ms per prediction")
    print("Keras model: NOT AVAILABLE")

elif keras_avg_time is not None:
    print(f"Skeleton model: NOT AVAILABLE")
    print(f"Keras model:   {keras_avg_time*1000:.3f} ms per prediction")

print("\n" + "="*60)
