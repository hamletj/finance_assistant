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
from typing import List, Dict, Optional
import time, html
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from bs4 import BeautifulSoup
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    # OpenAI Python SDK v1 style
    from openai import OpenAI
except ImportError:
    OpenAI = None



def _make_session() -> requests.Session:
    s = requests.Session()
    # Robust retry policy (handles 429 + common 5xx)
    retry = Retry(
        total=6,
        connect=3,
        read=3,
        backoff_factor=0.7,             # exponential backoff: 0.7, 1.4, 2.8, ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True, # honor Retry-After
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://finance.yahoo.com/",
    })
    return s

def _yahoo_news_search(query: str, count: int = 5, region: str = "US", lang: str = "en-US"):
    """
    Resilient Yahoo Finance news search with retries, backoff, and host fallback.
    """
    session = _make_session()
    params = {
        "q": query,
        "quotesCount": 0,
        "newsCount": max(1, int(count)),
        "region": region,
        "lang": lang,
    }

    # Try both hosts before giving up
    hosts = [
        "https://query1.finance.yahoo.com/v1/news/search",
        "https://query2.finance.yahoo.com/v1/news/search",
    ]

    last_err = None
    for host in hosts:
        try:
            # jitter to avoid thundering herd
            time.sleep(random.uniform(0.2, 0.6))
            r = session.get(host, params=params, timeout=12)
            # If 429 and Retry didn’t already backoff enough, respect Retry-After manually
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                        time.sleep(wait + random.uniform(0.2, 0.8))
                        # one more attempt after honoring Retry-After
                        r = session.get(host, params=params, timeout=12)
                    except Exception:
                        pass
            r.raise_for_status()
            data = r.json()
            news = data.get("news", []) or []
            return [{
                "title": n.get("title"),
                "publisher": n.get("publisher"),
                "link": n.get("link"),
                "time": n.get("providerPublishTime"),
            } for n in news[:count]]
        except requests.RequestException as e:
            last_err = e
            # short pause before switching host / failing
            time.sleep(random.uniform(0.5, 1.2))

    # If both hosts failed:
    raise RuntimeError(f"Yahoo Finance news search failed after retries: {last_err}")


def finance_news_digest_tool(
    query: str,
    top_n: int = 5,
    max_articles_chars: int = 18000,
) -> Dict:
    """
    Self-contained: searches Yahoo Finance news for `query`, fetches top articles,
    extracts text, and summarizes with GPT. Requires OPENAI_API_KEY in env.
    Returns: {'text': str, 'sources': List[...], 'image_path': None}
    """

    def _clean_url(u: str) -> str:
        try:
            p = urlparse(u)
            q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_")]
            return urlunparse(p._replace(query=urlencode(q)))
        except Exception:
            return u

    def _extract_article_text(url: str, timeout: int = 12) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceNewsDigest/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Try common article containers
        selectors = [
            "article",
            "[itemprop='articleBody']",
            ".article-content", ".post-content", ".story-body", ".entry-content",
            ".meteredContent", ".paywall"
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                ps = el.find_all(["p","h2","li"])
                text = "\n".join(p.get_text(" ", strip=True) for p in ps) or el.get_text(" ", strip=True)
                if len(text) > 400:
                    return text.strip()
        # Fallback: all <p>
        ps = soup.find_all("p")
        return "\n".join(p.get_text(" ", strip=True) for p in ps).strip()

    # 1) Search Yahoo Finance news
    q = (query or "").strip()
    if not q:
        return {"text": "Please provide a non-empty query.", "sources": [], "image_path": None}

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": q, "quotesCount": 0, "newsCount": max(1, int(top_n)), "region": "US", "lang": "en-US"}
    try:
        hits = _yahoo_news_search(query, count=top_n)
    except Exception as e:
        return {"text": f"Yahoo Finance news search failed: {e}", "sources": [], "image_path": None}

    # 2) Fetch & extract article text
    articles, total_chars = [], 0
    for h in hits:
        link = h["link"]
        if not link:
            continue
        try:
            txt = _extract_article_text(link)
        except Exception:
            txt = ""
        if len(txt) < 500:
            continue
        if total_chars + len(txt) > max_articles_chars:
            break
        articles.append({"title": h["title"], "publisher": h["publisher"], "url": link, "text": txt})
        total_chars += len(txt)
        time.sleep(0.35)  # polite pause

    if not articles:
        return {"text": f"No readable article bodies fetched for '{query}'.", "sources": hits, "image_path": None}

    # 3) Summarize with GPT (self-contained client)
    if OpenAI is None:
        return {"text": "OpenAI SDK not installed. `pip install openai`.", "sources": hits, "image_path": None}
    if not os.getenv("OPENAI_API_KEY"):
        return {"text": "Missing OPENAI_API_KEY in environment.", "sources": hits, "image_path": None}

    client = OpenAI()
    system = (
        "You are a financial news analyst. Produce a concise, neutral brief that integrates multiple sources. "
        "Quantify key facts and cite sources inline with [n] where n is the article index."
    )
    sources_block = "\n\n".join(
        f"[{i+1}] {a['title']} — {a['publisher']}\nURL: {a['url']}\n---\n{a['text'][:6000]}"
        for i, a in enumerate(articles)
    )
    user = (
        f"Topic: {query}\n\nSources (truncated content below):\n{sources_block}\n\n"
        "Write:\n"
        "- 5–8 bullet executive summary\n"
        "- What’s new vs background\n"
        "- Market impact (tickers/sectors) with key numbers\n"
        "- Risks/unknowns\n"
        "- One-sentence takeaway\n"
        "Keep ~300 words. Use [n] to cite."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.3,
    )
    summary = (resp.choices[0].message.content or "").strip()

    return {
        "text": summary,
        "sources": [{"index": i+1, "title": a["title"], "publisher": a["publisher"], "url": a["url"]} for i, a in enumerate(articles)],
        "image_path": None
    }


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
