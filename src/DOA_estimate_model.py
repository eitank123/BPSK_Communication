import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg

# -----------------------------
# SIGNAL MODEL
# -----------------------------
def generate_sample(theta, d=5e-2, wavelength=cfg.wavelength, snr=10):
    phi = 2*np.pi*d*np.sin(theta)/wavelength

    s = np.random.randn() + 1j*np.random.randn()

    x1 = s
    x2 = s * np.exp(-1j*phi)

    noise_power = 10**(-snr/10)

    x1 += np.sqrt(noise_power/2)*(np.random.randn()+1j*np.random.randn())
    x2 += np.sqrt(noise_power/2)*(np.random.randn()+1j*np.random.randn())

    return np.array([x1, x2])


# -----------------------------
# FEATURES (covariance)
# -----------------------------
def covariance_features(x):
    R = np.outer(x, np.conj(x))

    return np.array([
        np.real(R[0,0]),
        np.real(R[1,1]),
        np.real(R[0,1]),
        np.imag(R[0,1])
    ])


# -----------------------------
# MODEL
# -----------------------------
class DOA_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# DATASET BUILDER (vectorized)
# -----------------------------
def build_dataset(N=50000):

    X = []
    y = []

    for _ in range(N):

        theta = np.random.uniform(-np.pi/2, np.pi/2)
        x = generate_sample(theta)

        X.append(covariance_features(x))
        y.append(theta)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1,1)

    return X, y


# -----------------------------
# TRAINING (MINI-BATCH)
# -----------------------------
def train_model(batch_size=256, epochs=60, lr=2e-3):

    X, y = build_dataset()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DOA_Net()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        total_loss = 0

        for xb, yb in loader:

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss/len(loader):.4f}")

    return model


# -----------------------------
# INFERENCE
# -----------------------------
def predict(model, x):

    f = covariance_features(x)
    f = torch.tensor(f, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        return model(f).item()


# -----------------------------
# TEST
# -----------------------------
def test(model):

    theta_true = np.pi / 4
    x = generate_sample(theta_true)

    theta_hat = predict(model, x)

    print("True (deg):", np.degrees(theta_true))
    print("Pred (deg):", np.degrees(theta_hat))
    print("Error (deg):", np.degrees(theta_hat - theta_true))