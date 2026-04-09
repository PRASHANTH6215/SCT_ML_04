import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import yaml

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class GestureClassifier:
    def __init__(self, config_path=r"C:\Users\PRASHANTH\OneDrive\文档\Desktop\SCT_TASK_04\configs\config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self, train_csv: str, test_csv: str):
        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
        all_labels = pd.concat([train_df["label"], test_df["label"]])
        self.label_encoder.fit(all_labels)
        self.classes = self.label_encoder.classes_
        print(f"Classes: {list(self.classes)}")
        X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
        y_train = self.label_encoder.transform(train_df["label"])
        X_test  = test_df.drop(columns=["label"]).values.astype(np.float32)
        y_test  = self.label_encoder.transform(test_df["label"])
        print(f"Train: {X_train.shape} | Test: {X_test.shape}")
        return (X_train, y_train), (X_test, y_test)
    
    def train(self, X_train, y_train, X_test, y_test,
              epochs=50, batch_size=32):

        num_classes = len(self.classes)
        input_dim   = X_train.shape[1]

        # Build model
        self.model = MLP(input_dim, num_classes).to(self.device)

        # Split off validation set (10%)
        val_size    = int(len(X_train) * 0.1)
        X_val       = X_train[:val_size]
        y_val       = y_train[:val_size]
        X_tr        = X_train[val_size:]
        y_tr        = y_train[val_size:]

        # DataLoaders
        def make_loader(X, y, shuffle=True):
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            return DataLoader(TensorDataset(X_t, y_t),
                              batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(X_tr, y_tr)
        val_loader   = make_loader(X_val, y_val, shuffle=False)

        optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion  = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=4, factor=0.5)

        best_val_loss  = float("inf")
        patience_count = 0
        best_weights   = None

        for epoch in range(1, epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_b), y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    val_loss += criterion(self.model(X_b), y_b).item()

            train_loss /= len(train_loader)
            val_loss   /= len(val_loader)
            scheduler.step(val_loss)

            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = self.model.state_dict().copy()
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 8:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        self.model.load_state_dict(best_weights)

        # Test accuracy
        self.model.eval()
        with torch.no_grad():
            X_t   = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            preds = self.model(X_t).argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        print(f"\nTest accuracy: {acc*100:.2f}%")
        return acc

    def save(self, output_dir="models"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f"{output_dir}/gesture_model.pt")
        with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        # Save class list for inference
        with open(f"{output_dir}/classes.pkl", "wb") as f:
            pickle.dump(self.classes, f)
        print(f"Model saved → {output_dir}/")