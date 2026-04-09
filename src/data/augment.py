import numpy as np

def augment_landmarks(features: np.ndarray, n_augments: int = 5) -> np.ndarray:
    """
    Generates augmented versions of a single landmark sample.
    Applies small rotations, scale jitter, and Gaussian noise.
    Input shape: (63,) — returns shape: (n_augments, 63)
    """
    coords = features.reshape(21, 3)
    augmented = []

    for _ in range(n_augments):
        c = coords.copy()

        # Scale jitter ±10%
        c *= np.random.uniform(0.9, 1.1)

        # Rotation in xy-plane (simulate wrist rotation)
        angle = np.random.uniform(-15, 15) * np.pi / 180
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle),  np.cos(angle), 0],
                         [0,              0,             1]])
        c = (rot @ c.T).T

        # Gaussian noise
        c += np.random.normal(0, 0.01, c.shape)

        augmented.append(c.flatten())

    return np.array(augmented)