from pathlib import Path
import re

class LeapGestRecogLoader:
    """
    Walks the LeapGestRecog folder structure:
      <root>/<subject_id>/<gesture_folder>/<frame>.png
    Returns a list of (image_path, gesture_label) tuples.
    """

    # Maps folder prefix → clean label
    GESTURE_MAP = {
        "01": "palm",
        "02": "l",
        "03": "fist",
        "04": "fist_moved",
        "05": "thumb",
        "06": "index",
        "07": "ok",
        "08": "palm_moved",
        "09": "c",
        "10": "down",
    }

    def __init__(self, root_dir: str):
        self.root = Path(root_dir)

    def scan(self) -> list[tuple[str, str]]:
        """Returns [(image_path, label), ...]"""
        samples = []

        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue

            for gesture_dir in sorted(subject_dir.iterdir()):
                if not gesture_dir.is_dir():
                    continue

                # Extract the numeric prefix, e.g. "03" from "03_fist"
                match = re.match(r"^(\d+)_", gesture_dir.name)
                if not match:
                    continue

                prefix = match.group(1).zfill(2)
                label = self.GESTURE_MAP.get(prefix)
                if label is None:
                    continue

                for img_path in sorted(gesture_dir.glob("*.png")):
                    samples.append((str(img_path), label))

        print(f"Found {len(samples)} images across "
              f"{len(self.GESTURE_MAP)} gestures")
        return samples

    def subject_split(self, test_subjects: list[str] = None):
        """
        Returns train/test split by subject ID (avoids data leakage).
        Default: subjects 08 & 09 held out for test.
        """
        if test_subjects is None:
            test_subjects = ["08", "09"]

        train, test = [], []
        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue
            is_test = subject_dir.name in test_subjects

            for gesture_dir in sorted(subject_dir.iterdir()):
                if not gesture_dir.is_dir():
                    continue
                match = re.match(r"^(\d+)_", gesture_dir.name)
                if not match:
                    continue
                prefix = match.group(1).zfill(2)
                label = self.GESTURE_MAP.get(prefix)
                if label is None:
                    continue
                for img_path in sorted(gesture_dir.glob("*.png")):
                    entry = (str(img_path), label)
                    (test if is_test else train).append(entry)

        print(f"Train: {len(train)} | Test: {len(test)}")
        return train, test