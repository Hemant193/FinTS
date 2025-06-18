# FinTS: Financial Time-Series Forecasting Platform

FinTS is a machine learning platform for forecasting financial markets, built to predict stock and forex prices with high accuracy. It uses a **TimeSformer** model to forecast daily prices for German stocks like BMW.DE and tick-level forex pairs like EURUSDT. With interactive visualizations and explainability features, FinTS is ideal for traders, quants, and data scientists interested in financial modeling.

Developed as part of my Data Science Master’s, FinTS targets firms like Deutsche Bank for stock forecasting and HFT companies like Optiver for tick-level predictions. It’s optimized for Apple M2 (MPS backend)

## Features
- **TimeSformer Model**: Transformer-based model for time-series forecasting.
- **Data Support**: Daily data (BMW.DE: Open, High, Low, Close, Volume) and tick-level data (EURUSDT: price, qty).
- **Interactive UI**: TODO
- **API**: FastAPI endpoints for predictions and visualizations.
- **MLflow**: Tracks model metrics and parameters.
- **MPS Optimization**: Runs on Apple M2 with PyTorch MPS backend.


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/finTS.git
   cd finTS
   ```
2. Create and activate a virtual environment (Python 3.9):
   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure MPS support (for Apple M2):
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   Output should be `True`.

## Usage
1. **Fetch and Process Data**:
   ```bash
   python scripts/ingestion.py
   ```
   Generates `data/processed/BMW.DE.parquet` and `EURUSDT_ticks.parquet`.

2. **Train Models**:
   ```bash
   python scripts/model.py
   ```
   Saves `timeSformer_daily.pth` and `timeSformer_tick.pth` in `models/`.

3. **Evaluate Models**:
   ```bash
   python scripts/evaluation.py
   ```
   Outputs metrics (e.g., MAE < 2% for BMW.DE, < 0.01% for EURUSDT).

4. **Run API**:
   ```bash
   uvicorn scripts.api:app --host 0.0.0.0 --port 8000
   ```
   Access endpoints like `/predict/daily` and `/explain/tick`.



## Contributing
Contributions are welcome! Please open issues or submit pull requests for bug fixes, new features, or improvements. Focus areas:
- Adding more financial instruments.
- Enhancing model performance.
- Adding UI interactivity.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or collaboration, reach out via GitHub Issues or email (hemishra555@gmail.com).

---
Built with ❤️ by [Hemant] for financial forecasting enthusiasts.
