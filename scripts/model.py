import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet

class TimeSformer(nn.Module):
    def __init__(self, input_dim, d_model = 64, n_heads = 4, n_layers = 2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model, n_heads, n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc(x)
    
def prepare_data(data, input_dim, sequence_len = 10):
    X = []
    y = []

    for i in range(len(data) - sequence_len):
        X.append(data[i : i + sequence_len, : input_dim])
        y.append(data[i + sequence_len, 0])
    
    return np.array(X, dtype = np.float32), np.array(y, dtype = np.float32)