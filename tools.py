# tools.py
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import re
from datetime import datetime, timezone, timedelta
from pandas.tseries.offsets import DateOffset
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional

def general_search_tool(
    query: str,
    max_results: int = 5,
    backend: str = "duckduckgo",   # "duckduckgo" (no key), "tavily", or "serpapi"
    api_key: Optional[str] = None,
    locale: str = "en-US",
    region: str = "us"
) -> Dict:
    """
    Perform a general-purpose web search and return top results.

    Returns:
        dict: {
          "text": str,
          "results": [{"title": str, "url": str, "snippet": str}],
          "image_path": None
        }
    """
    query = (query or "").strip()
    if not query:
        return {"text": "Please provide a non-empty search query.", "results": [], "image_path": None}

    backend = backend.lower()
    try:
        if backend == "tavily":
            if not api_key:
                return {"text": "Tavily requires api_key.", "results": [], "image_path": None}
            # Docs: https://docs.tavily.com/
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
                timeout=12,
            )
            resp.raise_for_status()
            data = resp.json()
            out = []
            for r in (data.get("results") or [])[:max_results]:
                out.append({"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "").strip()})
            text = f"Found {len(out)} result(s) with Tavily."
            return {"text": text, "results": out, "image_path": None}

        elif backend == "serpapi":
            if not api_key:
                return {"text": "SerpAPI requires api_key.", "results": [], "image_path": None}
            # Docs: https://serpapi.com/
            params = {
                "engine": "google",
                "q": query,
                "num": max_results,
                "hl": locale.split("-")[0],
                "gl": region.upper(),
                "api_key": api_key,
            }
            resp = requests.get("https://serpapi.com/search", params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic_results", [])[:max_results]
            out = []
            for r in organic:
                out.append({"title": r.get("title"), "url": r.get("link"), "snippet": (r.get("snippet") or "").strip()})
            text = f"Found {len(out)} result(s) with SerpAPI."
            return {"text": text, "results": out, "image_path": None}

        else:
            # ---- Default: DuckDuckGo HTML (no API key) ----
            # We use their HTML endpoint to avoid JS; structure is reasonably stable.
            # Be polite with headers & timeout.
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; GeneralSearchTool/1.0; +https://example.com/bot)"
            }
            # DDG “html” endpoint prefers POST with form data
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": region, "k1": "-1"},  # k1=-1: no safe search tweak; adjust as you like
                headers=headers,
                timeout=12,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            results = []
            # Results live under .result; link in a.result__a; snippet in .result__snippet
            for res in soup.select("div.result"):
                a = res.select_one("a.result__a")
                if not a:
                    continue
                title = a.get_text(" ", strip=True)
                url = a.get("href")
                snippet_el = res.select_one(".result__snippet") or res.select_one(".result__snippet.js-result-snippet")
                snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
                if title and url:
                    results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= max_results:
                    break

            return {
                "text": f"Found {len(results)} result(s) with DuckDuckGo.",
                "results": results,
                "image_path": None,
            }

    except requests.RequestException as e:
        return {"text": f"Search failed: network error: {e}", "results": [], "image_path": None}
    except Exception as e:
        return {"text": f"Search failed: {e}", "results": [], "image_path": None}


def _parse_period_to_offset(period: str) -> DateOffset:
    """
    Convert a yfinance-style period string ('1mo','3mo','1y','5d','2wk') to a pandas DateOffset.
    """
    if period == "max":
        # Fallback: ~30 years. Adjust if you want.
        return DateOffset(years=30)
    m = re.fullmatch(r"(\d+)([a-zA-Z]+)", period.strip())
    if not m:
        raise ValueError(f"Unsupported period format: {period}")
    n, unit = int(m.group(1)), m.group(2).lower()
    if unit in ("d", "day", "days"):
        return DateOffset(days=n)
    if unit in ("wk", "w", "week", "weeks"):
        return DateOffset(weeks=n)
    if unit in ("mo", "month", "months"):
        return DateOffset(months=n)
    if unit in ("y", "yr", "year", "years"):
        return DateOffset(years=n)
    raise ValueError(f"Unsupported period unit: {unit}")

def _interval_to_timedelta(interval: str) -> timedelta:
    """
    Approximate bar duration for common yfinance intervals.
    """
    interval = interval.lower().strip()
    if interval.endswith("m"):  # minutes
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):  # hours
        return timedelta(hours=int(interval[:-1]))
    if interval == "1d":
        return timedelta(days=1)
    if interval == "5d":
        return timedelta(days=5)
    if interval == "1wk":
        return timedelta(weeks=1)
    if interval == "1mo":
        # Use ~30 days for approximation
        return timedelta(days=30)
    if interval == "3mo":
        return timedelta(days=90)
    # Default fallback
    return timedelta(days=1)

def moving_average_tool(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    windows=(20,),
):
    """
    Fetch OHLCV for `ticker`, compute SMAs with proper lookback so the
    SMA starts at the beginning of the requested period, and plot.

    - Extra data from further back is used ONLY to compute the SMA.
    - The visible Close line is clipped to the requested period.
    - The SMA line is shown starting at the first timestamp of the requested period.
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

    max_win = max(windows)

    # Compute visible window [visible_start, visible_end]
    now = datetime.now(timezone.utc)
    try:
        period_offset = _parse_period_to_offset(period)
    except Exception as e:
        result["text"] = f"Invalid period '{period}': {e}"
        return result

    visible_end = now
    visible_start = (now - period_offset)

    # Compute lookback needed for SMA calculation
    bar_delta = _interval_to_timedelta(interval)
    # Add a small buffer (e.g., 20%) to cover non-trading days / sparse bars
    extra_seconds = int((max_win - 1) * bar_delta.total_seconds() * 1.2)
    lookback_start = visible_start - timedelta(seconds=extra_seconds)

    # 1) Fetch extended data using start/end so we can control lookback
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=lookback_start, end=visible_end, interval=interval, auto_adjust=False)
        if hist.empty:
            result["text"] = f"No data found for {ticker}."
            return result
        hist = hist.reset_index()  # ensures a 'Date' column
    except Exception as e:
        result["text"] = f"Error fetching data for {ticker}: {e}"
        return result

    # 2) Compute SMAs on the EXTENDED data so first visible point has a valid SMA
    for w in windows:
        if len(hist) >= w:
            hist[f"SMA_{w}"] = hist["Close"].rolling(window=w).mean()

    # 3) Build masks for plotting
    # Visible range mask: we only *display* Close inside requested period
    mask_visible = hist["Date"] >= pd.Timestamp(visible_start)

    # 4) Plot
    fig, ax = plt.subplots()

    # Plot Close ONLY in visible window
    vis = hist.loc[mask_visible]
    ax.plot(vis["Date"], vis["Close"], marker="o", label="Close")

    # Plot SMA lines starting at the first visible timestamp
    for w in windows:
        col = f"SMA_{w}"
        if col in hist.columns:
            sma_vis = hist.loc[mask_visible, ["Date", col]].dropna()
            # If we fetched enough lookback, we should have a value at the first visible point.
            if not sma_vis.empty:
                ax.plot(sma_vis["Date"], sma_vis[col], linewidth=2, label=f"SMA {w}")

    ax.set_title(f"{ticker} Close with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best")
    fig.autofmt_xdate()

    # 5) Save
    tmpdir = tempfile.gettempdir()
    fname = f"{ticker}_ma_{int(datetime.now(timezone.utc).timestamp())}.png"
    path = os.path.join(tmpdir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

    result["image_path"] = path
    result["text"] = (
        f"Plotted {ticker} Close ({period}, {interval}) with SMAs: "
        + ", ".join(str(w) for w in windows)
        + ". Extra lookback fetched to start SMA at the beginning of the visible window."
    )
    return result

# def moving_average_tool(
#     ticker: str,
#     period: str = "1mo",
#     interval: str = "1d",
#     windows=(20,),
# ):
#     """
#     Fetch OHLCV data for `ticker` and plot Close with one or more
#     simple moving averages (SMA). Saves a PNG to a temp folder.

#     Args:
#         ticker (str): e.g., "AAPL"
#         period (str): e.g., "1mo", "3mo", "1y"
#         interval (str): e.g., "1d", "1h", "15m"
#         windows (Iterable[int]): moving-average window sizes in periods

#     Returns:
#         dict: {'text': str, 'image_path': str | None}
#     """
#     ticker = ticker.strip().upper()
#     result = {"text": "", "image_path": None}

#     # Validate windows
#     try:
#         windows = [int(w) for w in windows if int(w) > 1]
#         if not windows:
#             result["text"] = "No valid moving-average windows provided (need integers > 1)."
#             return result
#     except Exception:
#         result["text"] = "Invalid `windows` argument; provide integers > 1."
#         return result

#     # 1) Fetch data
#     try:
#         t = yf.Ticker(ticker)
#         hist = t.history(period=period, interval=interval)
#         if hist.empty:
#             result["text"] = f"No data found for {ticker}."
#             return result
#         hist = hist.reset_index()
#     except Exception as e:
#         result["text"] = f"Error fetching data for {ticker}: {e}"
#         return result

#     # 2) Compute SMAs (skip if window longer than dataset)
#     for w in windows:
#         if len(hist) >= w:
#             hist[f"SMA_{w}"] = hist["Close"].rolling(window=w).mean()

#     # 3) Plot
#     fig, ax = plt.subplots()
#     ax.plot(hist["Date"], hist["Close"], marker="o", label="Close")

#     plotted_any_ma = False
#     for w in windows:
#         col = f"SMA_{w}"
#         if col in hist:
#             ax.plot(hist["Date"], hist[col], linewidth=2, label=f"SMA {w}")
#             plotted_any_ma = True

#     ax.set_title(f"{ticker} Close with Moving Averages")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Price (USD)")
#     ax.legend(loc="best")
#     fig.autofmt_xdate()

#     # 4) Save
#     tmpdir = tempfile.gettempdir()
#     fname = f"{ticker}_ma_{int(datetime.utcnow().timestamp())}.png"
#     path = os.path.join(tmpdir, fname)
#     plt.savefig(path, bbox_inches="tight")
#     plt.close(fig)

#     result["image_path"] = path
#     if not plotted_any_ma:
#         result["text"] = (
#             "Plot created, but no SMA lines were added because all windows "
#             "are longer than the number of data points."
#         )
#     else:
#         result["text"] = (
#             f"Plotted {ticker} Close with SMAs: {', '.join(str(w) for w in windows)}."
#         )
#     return result

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
