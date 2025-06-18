import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from scripts.model import TimeSformer, prepare_data

def compute_shap_values(model, data, input_dim, feature_scaler, device="mps", sequence_length=10, n_samples=100):

    device = torch.device(device if torch.backends.mps.is_available() else "cpu")
    X, _, _, _ = prepare_data(data, input_dim, sequence_length)
    model.to(device)
    model.eval()
    
    background = X[np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)]
    background_np = background
    
    # Define model prediction function
    def model_predict(inputs):
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(inputs_tensor).cpu().numpy()
        return outputs
    
    explainer = shap.KernelExplainer(model_predict, background_np)
    shap_values = explainer.shap_values(X[:min(100, X.shape[0])], nsamples=100)
    return shap_values, X

def plot_shap_values(shap_values, symbol, feature_names, output_path, sequence_length=10):

    shap_mean = np.mean(np.abs(shap_values), axis=(0, 1))
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, shap_mean, color='skyblue')
    plt.title(f"SHAP Feature Importance for {symbol}")
    plt.xlabel("Features")
    plt.ylabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # BMW.DE (Daily)
    daily_data = pd.read_parquet("data/processed/BMW.DE.parquet")
    model_daily = TimeSformer(input_dim=5)
    checkpoint_daily = torch.load("models/timeSformer_daily.pth", map_location=device)
    model_daily.load_state_dict(checkpoint_daily["model_state"])
    feature_scaler_daily = checkpoint_daily["feature_scaler"]
    shap_values_daily, X_daily = compute_shap_values(
        model_daily,
        daily_data[["Open", "High", "Low", "Close", "Volume"]],
        input_dim=5,
        feature_scaler=feature_scaler_daily,
        device=device
    )
    plot_shap_values(
        shap_values_daily,
        "BMW.DE",
        feature_names=["Open", "High", "Low", "Close", "Volume"],
        output_path="images/shap_daily.png"
    )
    
    # EURUSDT (Tick)
    tick_data = pd.read_parquet("data/processed/EURUSDT_ticks.parquet")
    model_tick = TimeSformer(input_dim=2)
    checkpoint_tick = torch.load("models/timeSformer_tick.pth", map_location=device)
    model_tick.load_state_dict(checkpoint_tick["model_state"])
    feature_scaler_tick = checkpoint_tick["feature_scaler"]
    shap_values_tick, X_tick = compute_shap_values(
        model_tick,
        tick_data[["price", "qty"]],
        input_dim=2,
        feature_scaler=feature_scaler_tick,
        device=device
    )
    plot_shap_values(
        shap_values_tick,
        "EURUSDT",
        feature_names=["price", "qty"],
        output_path="images/shap_tick.png"
    )
    print("SHAP visualizations saved to images/shap_daily.png and images/shap_tick.png")