import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
import pyro
import pyro.distributions as dist

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

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.values[:, :input_dim])
    target = data.values[:, 0]
    X = []
    y = []

    data_array = data.values
    for i in range(len(data) - sequence_length):
        X.append(data_array[i:i+sequence_length, :input_dim])
        y.append(data_array[i+sequence_length, 0])

    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y, scaler

def train_model(model, data, input_dim, model_path, epochs=10, device="mps"):

    X, y, scaler = prepare_data(data, input_dim)
    X = torch.tensor(X).to(device)
    y = torch.tensor(y).to(device).reshape(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    model.to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    torch.save({"model_state": model.state_dict(), "scaler": scaler}, model_path)


def probabilistic_forecast(model, data, input_dim, scaler, device="mps"):
    X, _, _ = prepare_data(data, input_dim)
    X = torch.tensor(X).to(device)
    
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
    mean = scaler.inverse_transform(samples.mean(axis=0).reshape(-1, 1)).flatten()
    std = samples.std(axis=0) * scaler.scale_[0]
    return mean, std

# def train_prophet(data):
#     df = pd.DataFrame({"ds": data.index, "y": data["Close"]})
#     model = Prophet()
#     model.fit(df)
#     return model

if __name__ == "__main__":

    daily_data = pd.read_parquet("data/processed/BMW.DE.parquet")
    tick_data = pd.read_parquet("data/processed/EURUSDT_ticks.parquet")

    model_daily = TimeSformer(input_dim = 5)
    model_tick = TimeSformer(input_dim = 2)

    train_model(model_daily, daily_data[["Open", "High", "Low", "Close", "Volume"]], 5, "models/timesSformer_daily.pth", 20)
    train_model(model_tick, tick_data[["price", "qty"]], 2, "models/timeSformer_tick.pth", 20)