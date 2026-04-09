import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.loader import LeapGestRecogLoader
from src.data.extractor import LandmarkExtractor
import pandas as pd
from tqdm import tqdm

def extract_split(samples, output_path, extractor):
    records = []
    failed = 0

    for img_path, label in tqdm(samples, desc=f"Extracting → {Path(output_path).name}"):
        features = extractor.extract_from_image(img_path)
        if features is not None:
            records.append([label] + features.tolist())
        else:
            failed += 1

    cols = ["label"] + [
        f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]
    ]
    df = pd.DataFrame(records, columns=cols)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples | Skipped {failed} (no hand detected) → {output_path}")
    return df

# 1. Scan dataset
loader = LeapGestRecogLoader(r"C:\Users\PRASHANTH\OneDrive\文档\Desktop\SCT_TASK_04\data\leapGestRecog")
train_samples, test_samples = loader.subject_split(test_subjects=["08", "09"])

# 2. Init extractor
extractor = LandmarkExtractor()

# 3. Extract both splits
train_df = extract_split(train_samples, "data/processed/train.csv", extractor)
test_df  = extract_split(test_samples,  "data/processed/test.csv",  extractor)

print(f"\nDone! Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Classes: {train_df['label'].unique()}")