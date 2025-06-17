import logging
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_daily_data(symbol, start_date="2020-01-01", end_date="2025-06-17"):
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")
        # Ensure timezone is CET
    data.index = data.index.tz_convert('Europe/Berlin') if data.index.tz else data.index.tz_localize('Europe/Berlin')
    data.reset_index().to_csv(f"data/raw/{symbol}_daily.csv", index=False)
    logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
    return data


if __name__ == "__main__":
    fetch_daily_data("BMW.DE")