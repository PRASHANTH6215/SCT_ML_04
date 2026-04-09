import numpy as np
from pathlib import Path
import yaml

class LandmarkExtractor:
    def __init__(self, config_path=r"C:\Users\PRASHANTH\OneDrive\文档\Desktop\SCT_TASK_04\configs\config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        lm_cfg = self.cfg["landmark_extraction"]

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=lm_cfg["max_num_hands"],
            min_hand_detection_confidence=lm_cfg["min_detection_confidence"],
            min_tracking_confidence=lm_cfg["min_tracking_confidence"],
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _get_model_path(self):
        model_path = Path("models/hand_landmarker.task")
        if not model_path.exists():
            print("Downloading hand landmarker model...")
            import urllib.request
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                model_path
            )
            print("Downloaded → models/hand_landmarker.task")
        return str(model_path)

    def extract_from_frame(self, frame: np.ndarray):
        """
        Accepts a BGR numpy array directly from cv2 (no disk I/O).
        Returns (features, raw_result) tuple.
        features is (63,) array or None if no hand detected.
        """
        import mediapipe as mp
        import cv2
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
        )
        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return None, result
        landmarks = result.hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        coords -= coords[0]
        scale = np.abs(coords).max()
        if scale > 0:
            coords /= scale
            return coords.flatten(), result

    def extract_dataset(self, raw_dir: str, output_path: str):
        from tqdm import tqdm
        import pandas as pd

        raw_path = Path(raw_dir)
        records = []

        for gesture_dir in sorted(raw_path.iterdir()):
            if not gesture_dir.is_dir():
                continue
            label = gesture_dir.name
            image_files = list(gesture_dir.glob("*.jpg")) + \
                          list(gesture_dir.glob("*.png"))

            for img_path in tqdm(image_files, desc=label):
                features = self.extract_from_image(str(img_path))
                if features is not None:
                    records.append([label] + features.tolist())

        cols = ["label"] + [
            f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]
        ]
        df = pd.DataFrame(records, columns=cols)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} samples → {output_path}")
        return df