import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model.classifier import GestureClassifier

clf = GestureClassifier()

(X_train, y_train), (X_test, y_test) = clf.load_data(
    train_csv="data/processed/train.csv",
    test_csv="data/processed/test.csv",
)

acc = clf.train(
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=32,
)

clf.save()