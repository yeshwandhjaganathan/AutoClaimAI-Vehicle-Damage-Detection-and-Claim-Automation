# -*- coding: utf-8 -*-
"""
AutoClaimAI: Vehicle Damage Detection and Claim Automation
This script integrates:
- YOLOv5 for damage detection
- DenseNet for severity classification
- ImageHash for duplicate/fraud detection
- SHAP for explainability
- Supports images and video input
"""

# Install packages if running in Colab
!pip install torch torchvision ultralytics tensorflow pillow numpy imagehash shap opencv-python-headless matplotlib scikit-learn

# --- Imports
import os
import io
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import shap
import imagehash
from PIL import Image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files

# --- Load YOLOv5 Model
print("Loading YOLOv5s model...")
yolo_model = YOLO('yolov5s.pt')
print("YOLOv5s loaded.")

# --- Create DenseNet Model
NUM_SEVERITY_CLASSES = 3
SEVERITY_LABELS = ['Minor', 'Moderate', 'Severe']

def create_densenet_model(num_classes, input_shape=(224, 224, 3)):
    base = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    pred = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=pred)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

densenet_model = create_densenet_model(NUM_SEVERITY_CLASSES)

# Dummy background data for SHAP
dummy_background_data = np.random.rand(5, 224, 224, 3) * 255.0

# --- Duplicate Check Function
def check_duplicate_image(image_path_or_bytes, stored_hashes, hash_size=8, threshold=5):
    try:
        if isinstance(image_path_or_bytes, str):
            img = Image.open(image_path_or_bytes).convert("L")
        elif isinstance(image_path_or_bytes, bytes):
            img = Image.open(io.BytesIO(image_path_or_bytes)).convert("L")
        else:
            raise ValueError("Invalid input.")
        curr_hash = imagehash.average_hash(img, hash_size=hash_size)
        for stored in stored_hashes:
            if abs(curr_hash - stored) <= threshold:
                return True, stored
        return False, None
    except Exception as e:
        print(f"Duplicate check error: {e}")
        return False, None

# --- SHAP Explanation Function
def generate_shap_explanation(model, image, target_class_idx, background_data=None):
    try:
        img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        img_tensor = tf.image.resize(img_tensor, (224,224))
        img_tensor = tf.expand_dims(img_tensor, 0) / 255.0
        def f(x):
            x = tf.image.resize(tf.convert_to_tensor(x, dtype=tf.float32), (224,224)) / 255.0
            return model.predict(x)
        if background_data is None:
            background_data = np.zeros((1,224,224,3))
        else:
            background_data = tf.image.resize(tf.convert_to_tensor(background_data, dtype=tf.float32), (224,224)).numpy() / 255.0
        explainer = shap.KernelExplainer(f, background_data)
        shap_values = explainer.shap_values(img_tensor[0].numpy(), nsamples=100)
        return shap_values[target_class_idx], img_tensor[0].numpy()
    except Exception as e:
        print(f"SHAP error: {e}")
        return None, None

# --- Video Frame Extraction
def extract_frames_from_video(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
        count +=1
    cap.release()
    return frames

# --- Upload Files
uploaded = files.upload()
uploaded_files = list(uploaded.keys())

# --- Processing Loop
stored_claim_hashes = []
all_results = []

for file in uploaded_files:
    print(f"\nProcessing: {file}")
    if file.lower().endswith((".mp4",".avi",".mov",".mkv")):
        frames = extract_frames_from_video(file, frame_interval=30)
        frame_paths = []
        for idx, frame in enumerate(frames):
            fname = f"frame_{idx}_{file}.jpg"
            frame.save(fname)
            frame_paths.append(fname)
    else:
        frame_paths = [file]

    for img_path in frame_paths:
        # YOLO Detection
        results = yolo_model(img_path)
        detected_objs = []
        for r in results:
            for box in r.boxes:
                detected_objs.append({
                    'class': yolo_model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'box': [int(c) for c in box.xyxy[0]]
                })

        # Severity Classification
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        resized = cv2.resize(arr, (224,224))[None]/255.0
        pred = densenet_model.predict(resized)
        class_idx = np.argmax(pred[0])
        severity = SEVERITY_LABELS[class_idx]
        conf = float(pred[0][class_idx])

        # Duplicate Check
        is_dup, matched_hash = check_duplicate_image(img_path, stored_claim_hashes)
        if not is_dup:
            stored_claim_hashes.append(imagehash.average_hash(img.convert("L")))

        # SHAP Explanation
        shap_vals, _ = generate_shap_explanation(densenet_model, cv2.resize(arr,(224,224)), class_idx, dummy_background_data)

        # Result Summary
        res = {
            "Image": img_path,
            "Objects Detected": ", ".join([o['class'] for o in detected_objs]) if detected_objs else "None",
            "Detection Confidences": ", ".join([f"{o['confidence']:.2f}" for o in detected_objs]) if detected_objs else "None",
            "Predicted Severity": severity,
            "Severity Confidence": f"{conf:.2f}",
            "Duplicate Status": "Duplicate" if is_dup else "Unique"
        }
        all_results.append(res)

# --- CSV Export
df = pd.DataFrame(all_results)
csv_path = "AutoClaimAI_Results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults exported to: {csv_path}")

# Display
df
