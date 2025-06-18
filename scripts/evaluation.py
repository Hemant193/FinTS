from scripts.model import prepare_data, TimeSformer
import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, data, input_dim, feature_scaler, target_scaler, device="mps"):
    X, y, _, _ = prepare_data(data, input_dim)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy().flatten()
    y_unscaled = target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    pred_unscaled = target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_unscaled, pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_unscaled, pred_unscaled))
    crps = np.mean(np.abs(pred_unscaled - y_unscaled) - 0.5 * np.abs(pred_unscaled - y_unscaled))
    return {"mae": mae, "rmse": rmse, "crps": crps}

def log_metrics(metrics, model_name):
    with mlflow.start_run():
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_param("model", model_name)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    daily_data = pd.read_parquet("data/processed/BMW.DE.parquet")
    tick_data = pd.read_parquet("data/processed/EURUSDT_ticks.parquet")

    model_daily = TimeSformer(input_dim=5)
    model_tick = TimeSformer(input_dim=2)

    checkpoint_daily = torch.load("models/timeSformer_daily.pth", map_location=device)
    checkpoint_tick = torch.load("models/timeSformer_tick.pth", map_location=device)

    model_daily.load_state_dict(checkpoint_daily["model_state"])
    model_tick.load_state_dict(checkpoint_tick["model_state"])

    feature_scaler_daily = checkpoint_daily["feature_scaler"]
    target_scaler_daily = checkpoint_daily["target_scaler"]

    feature_scaler_tick = checkpoint_tick["feature_scaler"]
    target_scaler_tick = checkpoint_tick["target_scaler"]

    metrics_daily = evaluate_model(model_daily, daily_data[["Open", "High", "Low", "Close", "Volume"]], 5, feature_scaler_daily, target_scaler_daily, device)
    metrics_tick = evaluate_model(model_tick, tick_data[["price", "qty"]], 2, feature_scaler_tick, target_scaler_tick, device)

    log_metrics(metrics_daily, "timeSformer_daily")
    log_metrics(metrics_tick, "timeSformer_tick")

    print("TimeSformer Daily Metrics:", metrics_daily)
    print("TimeSformer Tick Metrics:", metrics_tick)