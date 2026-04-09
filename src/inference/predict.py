import torch
import pickle
import numpy as np
from pathlib import Path
from src.model.classifier import MLP

class GesturePredictor:
    def __init__(self, model_dir="models"):
        # Load label encoder and classes
        with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
        with open(f"{model_dir}/classes.pkl", "rb") as f:
            self.classes = pickle.load(f)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = MLP(input_dim=63, num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(
            f"{model_dir}/gesture_model.pt",
            map_location=self.device
        ))
        self.model.eval()
        print(f"Model loaded | Classes: {list(self.classes)}")

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        """
        Takes a (63,) landmark array, returns (label, confidence).
        """
        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        label = self.label_encoder.inverse_transform([idx.item()])[0]
        return label, round(conf.item(), 3)