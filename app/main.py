import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

Path("app").mkdir(exist_ok=True)

import cv2
import mediapipe as mp
import numpy as np
from src.inference.predict import GesturePredictor
from src.data.extractor import LandmarkExtractor

predictor = GesturePredictor()
extractor = LandmarkExtractor()

cap = cv2.VideoCapture(0)
print("Press Q to quit")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(frame, landmarks_normalized):
    h, w = frame.shape[:2]
    points = [
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks_normalized
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 128, 255), -1)
    cv2.circle(frame, points[0], 6, (255, 255, 255), -1)

# Smoothing buffer — average last N predictions
from collections import deque
pred_buffer = deque(maxlen=5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Extract directly from frame — no disk I/O
    features, result = extractor.extract_from_frame(frame)

    # Draw landmarks
    if result.hand_landmarks:
        draw_landmarks(frame, result.hand_landmarks[0])

    # Predict
    if features is not None:
        label, confidence = predictor.predict(features)

        # Add to smoothing buffer
        if confidence > 0.5:
            pred_buffer.append(label)

        # Pick most common label in buffer
        if pred_buffer:
            from collections import Counter
            smooth_label = Counter(pred_buffer).most_common(1)[0][0]
        else:
            smooth_label = label

        color = (0, 200, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.rectangle(frame, (0, 0), (320, 65), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {smooth_label}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        pred_buffer.clear()
        cv2.rectangle(frame, (0, 0), (260, 40), (0, 0, 0), -1)
        cv2.putText(frame, "No hand detected",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()