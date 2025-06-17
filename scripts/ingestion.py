import logging
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_daily_data(symbol, start_date = "2020-01-01", end_date = "2025-06-17"):
    logging.info("Fetching")

    ticker = yf.Ticker(symbol)
    data = ticker.history(start = start_date, end = end_date, auto_adjust = True)
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")
        # Ensure timezone is CET
    data.index = data.index.tz_convert('Europe/Berlin') if data.index.tz else data.index.tz_localize('Europe/Berlin')
    data.reset_index().to_csv(f"data/raw/{symbol}_daily.csv", index = False)
    logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
    return data

def fetch_real_time_data(symbol, api_key):
    logger.info(f"Fetchin 5 min data using Alpha Vintage for {symbol}")

    ts = TimeSeries(key = api_key, output_format = "pandas")
    data, _ = ts.get_intraday(symbol = symbol, interval = "5min")
    data.index = data.index.tz_localize('Europe/Berlin')
    data.to_csv(f"data/raw/{symbol}_5min.csv")
    logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
    logger.info("Success")
    return data

if __name__ == "__main__":

    api_key = os.getenv("API_KEY")

    # fetch_daily_data("BMW.DE")
    fetch_real_time_data("EURUSD", api_key)
    