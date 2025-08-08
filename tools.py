# tools.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime

def finance_tool(ticker: str,  period: str = "1mo", interval: str = "1d"):
    """
    A single finance tool that:
    - Gets stock data for a ticker


    Returns: dict with keys 'text', 'image_path' (optional)
    """
    ticker = ticker.strip().upper()
    result = {"text": "", "image_path": None}

    # 1. Get data
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            result["text"] = f"No data found for {ticker}."
            return result
        hist = hist.reset_index()
    except Exception as e:
        result["text"] = f"Error fetching data for {ticker}: {e}"
        return result

    # # 2. Summarize if needed
    # if action in ("summarize", "both"):
    #     closes = hist["Close"].astype(float)
    #     pct_change = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
    #     trend = "up" if pct_change > 0 else "down" if pct_change < 0 else "flat"
    #     result["text"] += f"{ticker} moved {pct_change:.2f}% over the last {len(closes)} points ({trend}).\n"

    # 3. Plot if needed
    # if action in ("plot"):
    fig, ax = plt.subplots()
    ax.plot(hist["Date"], hist["Close"], marker="o")
    ax.set_title(f"{ticker} Closing Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    fig.autofmt_xdate()

    tmpdir = tempfile.gettempdir()
    fname = f"{ticker}_plot_{int(datetime.utcnow().timestamp())}.png"
    path = os.path.join(tmpdir, fname)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

    result["image_path"] = path

    # # 4. Always include last rows if just "data"
    # if action == "data":
    #     last_rows = hist.tail(5)[["Date", "Open", "High", "Low", "Close", "Volume"]]
    #     result["text"] = f"Last 5 rows for {ticker}:\n{last_rows.to_string(index=False)}"

    return result
