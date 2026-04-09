# SCT_ML_04
# ✋ Hand Gesture Recognition using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Hand Gesture Recognition system that detects and classifies hand gestures using landmark-based feature extraction.

It uses computer vision and ML models to recognize gestures from images or live video input.

---

## 🚀 Features

* Extracts hand landmarks from images
* Trains ML models for gesture classification
* Supports real-time gesture prediction
* Modular pipeline (data → processing → training → prediction)
* Easy to extend with new gestures

---

## 🧠 Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy, Pandas
* Scikit-learn
* (Optional) TensorFlow / PyTorch

---

## 📂 Project Structure

hand-gesture-recognition/

│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── data/
│   │   ├── extractor.py
│   │   ├── loader.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│
├── scripts/
│   ├── build_dataset.py
│   ├── train_model.py
│
├── models/
│   ├── gesture_model.pkl
│
├── app.py
├── requirements.txt
├── README.md

---

## ⚙️ Installation

1. Clone the repository:
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition

2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

---

## 📊 Dataset Preparation

Place raw gesture images inside:
data/raw/

Run:
python scripts/build_dataset.py

Output:
data/processed/landmarks.csv

---

## 🏋️ Model Training

python scripts/train_model.py

Output:
models/gesture_model.pkl

---

## 🎯 Prediction

python src/models/predict.py

OR run app:
streamlit run app.py

---

## 📌 Future Improvements

* Add deep learning models (CNN/LSTM)
* Improve dataset quality
* Deploy as web/mobile app
* Multi-hand detection

---

## 👨‍💻 Author

Prashanth B

---

