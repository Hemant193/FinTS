import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, n_heads, n_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = x[:, -1, :]
        return self.fc(x)

def prepare_data(data, input_dim, sequence_length=10):

    feature_scaler = StandardScaler()
    data_scaled = feature_scaler.fit_transform(data.values[:, :input_dim])
    target = data.values[:, 0].reshape(-1, 1)
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(target)

    X = []
    y = []

    for i in range(len(data) - sequence_length):
        X.append(data_scaled[i:i+sequence_length, :input_dim])
        y.append(target_scaled[i+sequence_length])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y, feature_scaler, target_scaler

def train_model(model, data, input_dim, model_path, epochs=10, device="mps"):
    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    X, y, feature_scaler, target_scaler = prepare_data(data, input_dim)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    model.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    torch.save({
        "model_state": model.state_dict(),
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler
    }, model_path)

def probabilistic_forecast(model, data, input_dim, feature_scaler, target_scaler, device="mps"):
    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    X, _, _, _ = prepare_data(data, input_dim)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    model.to(device)
    model.eval()
    
    def pyro_model(input_data):
        mean = model(input_data).squeeze()
        with pyro.plate("data", len(mean)):
            pyro.sample("obs", dist.Normal(mean, 0.1), obs=None)
    
    guide = pyro.infer.autoguide.AutoNormal(pyro_model)
    svi = pyro.infer.SVI(pyro_model, guide, pyro.optim.Adam({"lr": 0.01}), loss=pyro.infer.Trace_ELBO())
    for _ in range(1000):
        svi.step(X)
    predictive = pyro.infer.Predictive(pyro_model, guide=guide, num_samples=1000)
    samples = predictive(X)["obs"].detach().cpu().numpy()
    mean = target_scaler.inverse_transform(samples.mean(axis=0).reshape(-1, 1)).flatten()
    std = samples.std(axis=0) * target_scaler.scale_[0]
    return mean, std


if __name__ == "__main__":
    daily_data = pd.read_parquet("data/processed/BMW.DE.parquet")
    tick_data = pd.read_parquet("data/processed/EURUSDT_ticks.parquet")
    model_daily = TimeSformer(input_dim=5)
    model_tick = TimeSformer(input_dim=2)
    train_model(model_daily, daily_data[["Open", "High", "Low", "Close", "Volume"]], 5, "models/timeSformer_daily.pth")
    train_model(model_tick, tick_data[["price", "qty"]], 2, "models/timeSformer_tick.pth")