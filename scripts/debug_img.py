# scripts/debug_image.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import LeapGestRecogLoader
from PIL import Image
import numpy as np
import cv2

loader = LeapGestRecogLoader(r"C:\Users\PRASHANTH\OneDrive\文档\Desktop\SCT_TASK_04\data\leapGestRecog")
samples = loader.scan()
img_path, label = samples[0]
print(f"Path: {img_path}")

# Try PIL instead of cv2
img = Image.open(img_path)
print(f"PIL mode: {img.mode}")        # L = grayscale, RGB, RGBA etc.
print(f"PIL size: {img.size}")        # (width, height)

# Convert to numpy
arr = np.array(img)
print(f"Array shape: {arr.shape}")
print(f"Pixel range: {arr.min()} - {arr.max()}")

# Try saving as a test jpg to visually inspect
img.save("debug_sample.jpg")
print("Saved debug_sample.jpg — open it to see what the image looks like")