# tools.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from datetime import datetime


def moving_average_tool(
    ticker: str,
    period: str = "1mo",
    interval: str = "1d",
    windows=(20,),
):
    """
    Fetch OHLCV data for `ticker` and plot Close with one or more
    simple moving averages (SMA). Saves a PNG to a temp folder.

    Args:
        ticker (str): e.g., "AAPL"
        period (str): e.g., "1mo", "3mo", "1y"
        interval (str): e.g., "1d", "1h", "15m"
        windows (Iterable[int]): moving-average window sizes in periods

    Returns:
        dict: {'text': str, 'image_path': str | None}
    """
    ticker = ticker.strip().upper()
    result = {"text": "", "image_path": None}

    # Validate windows
    try:
        windows = [int(w) for w in windows if int(w) > 1]
        if not windows:
            result["text"] = "No valid moving-average windows provided (need integers > 1)."
            return result
    except Exception:
        result["text"] = "Invalid `windows` argument; provide integers > 1."
        return result

    # 1) Fetch data
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

    # 2) Compute SMAs (skip if window longer than dataset)
    for w in windows:
        if len(hist) >= w:
            hist[f"SMA_{w}"] = hist["Close"].rolling(window=w).mean()

    # 3) Plot
    fig, ax = plt.subplots()
    ax.plot(hist["Date"], hist["Close"], marker="o", label="Close")

    plotted_any_ma = False
    for w in windows:
        col = f"SMA_{w}"
        if col in hist:
            ax.plot(hist["Date"], hist[col], linewidth=2, label=f"SMA {w}")
            plotted_any_ma = True

    ax.set_title(f"{ticker} Close with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best")
    fig.autofmt_xdate()

    # 4) Save
    tmpdir = tempfile.gettempdir()
    fname = f"{ticker}_ma_{int(datetime.utcnow().timestamp())}.png"
    path = os.path.join(tmpdir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

    result["image_path"] = path
    if not plotted_any_ma:
        result["text"] = (
            "Plot created, but no SMA lines were added because all windows "
            "are longer than the number of data points."
        )
    else:
        result["text"] = (
            f"Plotted {ticker} Close with SMAs: {', '.join(str(w) for w in windows)}."
        )
    return result

def past_history_tool(ticker: str,  period: str = "1mo", interval: str = "1d"):
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
    
    # 2. Plot if needed
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


    return result
