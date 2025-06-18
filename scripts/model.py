import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
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




if __name__ == "__main__":
    
    daily_data = pd.read_parquet("data/processed/BMW.DE.parquet")

    model_daily = TimeSformer(input_dim = 5)
    train_model(model_daily, daily_data[["Open", "High", "Low", "Close", "Volume"]], 5, "models/timesSformer_daily.pth", 20)