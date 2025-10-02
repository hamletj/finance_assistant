# tools.py
import os
import re
import time
import random
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ssl
import time
import random
import json
import certifi
import pandas as pd
import urllib.error
from urllib.request import urlopen, Request
from typing import List, Union, Optional

from datetime import datetime, timezone, timedelta
from pandas.tseries.offsets import DateOffset
from typing import List, Optional

from openai import OpenAI

# ----------------------
# FMP API helpers
# ----------------------

FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    raise RuntimeError("FinancialModelingPrep API key required: pass `api_key=` or set env FMP_API_KEY")
    
def _fetch_json(url: str, retries: int = 3, retry_backoff: float = 0.8, verbose: bool = True) -> dict:
    """Fetch JSON with retries and basic exponential backoff + jitter."""
    # SSL context using certifi
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cafile=certifi.where())
    headers = {"User-Agent": "Mozilla/5.0"}
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, context=ctx, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as he:
            # 4xx likely permanent -> raise immediately for the caller to handle
            if 400 <= getattr(he, "code", 0) < 500:
                raise
            last_exc = he
        except Exception as e:
            last_exc = e
        # backoff
        sleep_for = retry_backoff * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
        if verbose:
            print(f"[obtain_api_data_tool] fetch failed (attempt {attempt}/{retries}) -> sleeping {sleep_for:.2f}s")
        time.sleep(sleep_for)
    raise RuntimeError(f"Failed to fetch URL after {retries} attempts. Last error: {last_exc}")

# ----------------------
# Obtain API data when the data file is not present
# ----------------------
def obtain_api_data_tool(
    tickers: Union[str, List[str]],
    path_to_save: Optional[str] = 'data',
    from_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch quarterly data for one or more tickers from FinancialModelingPrep (FMP) API,
    combine income statement + key metrics + earnings calendar, save per-ticker CSVs
    to `path_to_save` (if provided), and return a concatenated DataFrame.

    Parameters
    ----------
    tickers : str or list[str]
        Single ticker or list of tickers (e.g., "AAPL" or ["AAPL","MSFT"])
    path_to_save : str | None
        Directory to save per-ticker CSVs. If None, no files are written.
    from_date : str | None
        Optional 'from' date parameter forwarded to FMP endpoints (format: YYYY-MM-DD).
    api_key : str | None
        FMP API key. If None, env var FMP_API_KEY is used.
    max_retries : int
        Number of retries for network requests.
    retry_backoff : float
        Base backoff seconds (exponential backoff applied).
    verbose : bool
        Print progress messages if True.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all fetched tickers (empty DataFrame if none succeeded).
    """

    # normalize tickers to list
    if isinstance(tickers, str):
        ticker_list = [tickers.strip().upper()]
    else:
        ticker_list = [t.strip().upper() for t in tickers]

    # ensure output dir exists if saving
    if path_to_save:
        os.makedirs(path_to_save, exist_ok=True)

    def _get_calendar(tk: str):
        if from_date:
            url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{tk}?from={from_date}&apikey={FMP_API_KEY}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{tk}?apikey={FMP_API_KEY}"
        return _fetch_json(url)

    def _get_income_statements(tk: str):
        # endpoint supports '?period=quarter' and optional limit/from
        if from_date:
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{tk}?from={from_date}&period=quarter&apikey={FMP_API_KEY}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{tk}?period=quarter&apikey={FMP_API_KEY}"
        return _fetch_json(url)

    def _get_keymetrics(tk: str):
        if from_date:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics/{tk}?from={from_date}&period=quarter&apikey={FMP_API_KEY}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics/{tk}?period=quarter&apikey={FMP_API_KEY}"
        return _fetch_json(url)

    def _combine_raw_data(tk: str, incomes_json, keymetrics_json, calendar_json) -> pd.DataFrame:
        """
        incomes_json, keymetrics_json, calendar_json are the raw JSON responses
        (likely lists of dicts). Convert to DataFrames, merge sensibly.
        """
        # turn into DataFrames (safe fallback to empty DF)
        df_inc = pd.DataFrame(incomes_json) if incomes_json else pd.DataFrame()
        df_key = pd.DataFrame(keymetrics_json) if keymetrics_json else pd.DataFrame()
        df_cal = pd.DataFrame(calendar_json) if calendar_json else pd.DataFrame()

        # Clean
        df_cal.drop(columns=['eps', 'revenue'], axis=1, inplace=True)
        df_cal.rename(columns={'date': 'report_date'}, inplace=True)
        # Merge
        df = df_inc.merge(df_key, on=['date', 'symbol', 'calendarYear', 'period'], how='left')
        df = df.merge(df_cal, left_on=['date', 'symbol'], right_on=['fiscalDateEnding', 'symbol'], how='left')
        # Use the date in statement data as report date when report date is null
        df.loc[df.report_date.isnull(), 'report_date'] = df.date
        # Changae data type
        df['report_date'] = pd.to_datetime(df['report_date'])

        return df

    # main loop for tickers
    all_results = []
    for tk in ticker_list:
        try:
            if verbose:
                print(f"[obtain_api_data_tool] Fetching {tk} ...")
            cal = _get_calendar(tk)
            inc = _get_income_statements(tk)
            km = _get_keymetrics(tk)

            # validate shapes - API often returns dict with 'Error message' or empty lists
            if not inc or (isinstance(inc, dict) and len(inc) == 0):
                if verbose:
                    print(f"[obtain_api_data_tool] income-statement returned empty for {tk}, skipping.")
                continue

            df_tk = _combine_raw_data(tk, inc, km, cal)

            # Save per-ticker CSV if requested
            if path_to_save:
                out_fn = os.path.join(path_to_save, f"{tk}.csv")
                try:
                    df_tk.to_csv(out_fn, index=False)
                    if verbose:
                        print(f"[obtain_api_data_tool] saved {tk} -> {out_fn}")
                except Exception as e:
                    if verbose:
                        print(f"[obtain_api_data_tool] Warning: could not save CSV for {tk}: {e}")

            all_results.append(df_tk)
        except Exception as e:
            if verbose:
                print(f"[obtain_api_data_tool] Error for {tk}: {e}. Skipping ticker.")

    if not all_results:
        return pd.DataFrame()
    combined = pd.concat(all_results, ignore_index=True, sort=False)
    return combined


# ----------------------
# Shared parsing & formatting helpers
# ----------------------
def _parse_number_str(x):
    """Robust parse helper: '1.66B','532M','45%','$1,234','(123)' -> float or NaN"""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return np.nan
    s = str(x).strip()
    if s == "" or s in {"-", "–", "—", "N/A", "NA", "null", "None"}:
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # remove thousands separators and spaces
    s = s.replace(",", "").replace(" ", "")
    # remove leading currency or stray characters
    s = re.sub(r"^[^\d\.\-\+]+", "", s)
    # strip trailing stray chars except letters and %
    s = re.sub(r"[^\d\.\-\+a-zA-Z%]+$", "", s)

    m = re.match(r"^([\-+]?\d*\.?\d+)([a-zA-Z%]*)$", s)
    if not m:
        return np.nan
    num = float(m.group(1))
    unit = m.group(2).lower()

    if unit in {"b", "bn", "bill", "billion"}:
        num *= 1e9
    elif unit in {"t", "tn", "trn", "trillion"}:
        num *= 1e12
    elif unit in {"m", "mm", "million"}:
        num *= 1e6
    elif unit in {"k", "thousand"}:
        num *= 1e3
    elif unit == "%":
        num /= 100.0

    return -num if neg else num


def _human_money(x):
    """Format numbers with T/B/M/K; keep NaN as '-'"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        x = float(x)
    except Exception:
        return "-"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e12:
        return f"{sign}{a/1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}{a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}{a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}{a/1e3:.2f}K"
    return f"{sign}{a:.2f}"


def _human_pct(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return f"{x*100:.{decimals}f}%"
    except Exception:
        return "-"


def _human_plain(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return f"{x:.{decimals}f}"
    except Exception:
        return "-"


def _cagr(start, end, years):
    if start is None or end is None:
        return np.nan
    try:
        start = float(start)
        end = float(end)
        if start <= 0 or years <= 0:
            return np.nan
        return (end / start) ** (1.0 / years) - 1.0
    except Exception:
        return np.nan


# ----------------------
# Existing finance tools (kept largely intact)
# ----------------------
def _parse_period_to_offset(period: str) -> DateOffset:
    if period == "max":
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
        return timedelta(days=30)
    if interval == "3mo":
        return timedelta(days=90)
    return timedelta(days=1)


def moving_average_tool(ticker: str, period: str = "1y", interval: str = "1d", windows=(20,)):
    ticker = ticker.strip().upper()
    result = {"text": "", "image_path": None}

    try:
        windows = [int(w) for w in windows if int(w) > 1]
        if not windows:
            result["text"] = "No valid moving-average windows provided (need integers > 1)."
            return result
    except Exception:
        result["text"] = "Invalid `windows` argument; provide integers > 1."
        return result

    max_win = max(windows)
    now = datetime.now(timezone.utc)
    try:
        period_offset = _parse_period_to_offset(period)
    except Exception as e:
        result["text"] = f"Invalid period '{period}': {e}"
        return result

    visible_end = now
    visible_start = (now - period_offset)
    bar_delta = _interval_to_timedelta(interval)
    extra_seconds = int((max_win - 1) * bar_delta.total_seconds() * 1.2)
    lookback_start = visible_start - timedelta(seconds=extra_seconds)

    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(start=lookback_start, end=visible_end, interval=interval, auto_adjust=False)
        if hist.empty:
            result["text"] = f"No data found for {ticker}."
            return result
        hist = hist.reset_index()
    except Exception as e:
        result["text"] = f"Error fetching data for {ticker}: {e}"
        return result

    for w in windows:
        if len(hist) >= w:
            hist[f"SMA_{w}"] = hist["Close"].rolling(window=w).mean()

    mask_visible = hist["Date"] >= pd.Timestamp(visible_start)

    fig, ax = plt.subplots()
    vis = hist.loc[mask_visible]
    ax.plot(vis["Date"], vis["Close"], marker="o", label="Close")
    for w in windows:
        col = f"SMA_{w}"
        if col in hist.columns:
            sma_vis = hist.loc[mask_visible, ["Date", col]].dropna()
            if not sma_vis.empty:
                ax.plot(sma_vis["Date"], sma_vis[col], linewidth=2, label=f"SMA {w}")

    ax.set_title(f"{ticker} Close with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best")
    fig.autofmt_xdate()

    tmpdir = tempfile.gettempdir()
    fname = f"{ticker}_ma_{int(datetime.now(timezone.utc).timestamp())}.png"
    path = os.path.join(tmpdir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

    result["image_path"] = path
    result["text"] = (
        f"Plotted {ticker} Close ({period}, {interval}) with SMAs: " + ", ".join(str(w) for w in windows)
    )
    return result


def past_history_tool(ticker: str, period: str = "1mo", interval: str = "1d"):
    ticker = ticker.strip().upper()
    result = {"text": "", "image_path": None}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            result["text"] = f"No data found for {ticker}."
            return result
        hist = hist.reset_index()
    except Exception as e:
        result["text"] = f"Error fetching data for {ticker}: {e}"
        return result

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


# ----------------------
# Original generate_financial_summary_tool
# ----------------------
def generate_financial_summary_tool(ticker: str, num_quarters: int = 5):
    """
    Generate a Tesla-style financial summary for a given ticker.
    Reads data/{TICKER}.csv, parses numbers, builds pivot, and formats output.

    Prioritization for valuation metrics:
    - If CSV contains a column matching peratio (e.g., 'peratio','peRatio','pe_ratio'), use it as P/E.
    - If CSV contains a column matching priceToSalesRatio (e.g., 'priceToSalesRatio','price_to_sales','ps_ratio'), use it as P/S.
    - Otherwise fall back to computing from price, sharesOutstanding, EPS.
    - Forward P/E computed using: forward_p/e = p/e * (eps / epsEstimated) when epsEstimated exists.
    """
    import os, re, numpy as np, pandas as pd

    # reuse the shared parser from tools.py
    def parse_number(x):
        return _parse_number_str(x)

    # formatting: keep 'x' for valuation ratios
    def fmt_value(val, metric_name):
        if pd.isna(val):
            return "-"
        if metric_name in ["P/E", "Forward P/E", "P/S"]:
            try:
                return f"{float(val):.1f}x"
            except Exception:
                return "-"
        if "Margin" in metric_name:
            return f"{val * 100:.1f}%"
        if metric_name == "EPS Diluted (GAAP)":
            return f"{val:.2f}"
        try:
            return f"{val / 1e9:.2f}B"
        except Exception:
            return str(val)

    # Helper: choose best column from df using candidate regexes (case-insensitive)
    def find_col(cols, patterns):
        lc_map = {c: c.lower() for c in cols}
        for pat in patterns:
            r = re.compile(pat, flags=re.IGNORECASE)
            for orig, lc in lc_map.items():
                if r.search(lc):
                    return orig
        return None


    path = os.path.join("data", f"{ticker.upper()}.csv")
    if not os.path.exists(path):
        # Try to obtain data using obtain_api_data_tool
        try:
            obtain_api_data_tool(ticker, path_to_save="data")
        except Exception as e:
            raise FileNotFoundError(f"{path} (and failed to fetch via obtain_api_data_tool: {e})")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} (fetch attempted but file still missing)")

    df = pd.read_csv(path)
    # filter by symbol column if present
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df = df[df["symbol"] == ticker.upper()]

    if df.empty:
        raise ValueError(f"No rows for ticker {ticker}")

    # --- detect which columns need parsing & coerce numeric-like columns ---
    protected = {"date", "symbol"}
    cols_to_parse = []
    coerced_to_numeric = []

    for c in df.columns:
        if c in protected:
            continue
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            coerced_to_numeric.append(c)
            continue
        sample = col.dropna().astype(str).head(200).tolist()
        if not sample:
            continue
        needs_parse = False
        for s in sample:
            s_strip = s.strip()
            if not re.search(r"\d", s_strip):
                continue
            if re.search(r"[BMK%(),\$\€\£\¥]", s_strip, flags=re.IGNORECASE) or re.search(r"\d,\d", s_strip):
                needs_parse = True
                break
            if re.search(r"^\s*[\-+]?\d*\.?\d+\s*[A-Za-z%]+$", s_strip):
                needs_parse = True
                break
            try:
                float(s_strip)
            except Exception:
                if re.search(r"\d", s_strip):
                    needs_parse = True
                    break
        if needs_parse:
            cols_to_parse.append(c)
        else:
            coerced = pd.to_numeric(col, errors="coerce")
            non_na = coerced.notna().sum()
            if non_na / max(1, len(col.dropna())) >= 0.9:
                df[c] = coerced
                coerced_to_numeric.append(c)

    # apply parsing for flagged columns
    for c in cols_to_parse:
        df[c] = df[c].apply(parse_number)

    # --- map raw → pretty where possible (existing behavior) ---
    raw_to_pretty = {
        "revenue": "Total Revenues",
        "grossProfit": "Gross Profit",
        "grossProfitRatio": "Gross Margin",
        "operatingIncome": "Income from Operations",
        "operatingIncomeRatio": "Operating Margin",
        "netIncome": "GAAP Net Income",
        "epsdiluted": "EPS Diluted (GAAP)",
        "ebitda": "Adjusted EBITDA",
        "ebitdaratio": "Adjusted EBITDA Margin",
        "capitalExpenditure": "Capital Expenditures",
        "freeCashFlow": "Free Cash Flow",
    }
    pretty_set = set(raw_to_pretty.values())

    # --- discover candidate columns (do not assume standardized names) ---
    cols = list(df.columns)

    # price candidates
    price_col = find_col(cols, [
        r"\bclose\b", r"\badjclose\b", r"\bcloseprice\b", r"\bprice\b", r"\blastprice\b", r"\blast\b"
    ])

    # shares outstanding candidates
    shares_col = find_col(cols, [
        r"sharesoutstanding", r"shares_outstanding", r"commonstocksharesoutstanding",
        r"weightedaverageshares", r"weighted_average_shares", r"\bshares\b"
    ])

    # eps (diluted) candidates
    eps_col = find_col(cols, [
        r"epsdilut", r"eps_dilut", r"\beps\b", r"basic_eps", r"diluted_eps", r"epsbasic", r"epsDiluted"
    ])

    # epsEstimated / forward EPS candidates (user prefers epsEstimated)
    eps_est_col = find_col(cols, [
        r"epsestimated", r"eps_estimated", r"epsestimate", r"eps_estimate", r"epsforward", r"forwardeps", r"eps_est"
    ])

    # forwardPE candidate in CSV (some CSVs may already include forwardPE)
    forwardpe_col = find_col(cols, [r"forwardpe", r"forward_pe", r"fwdpe"])

    # PRIORITY: peratio column for P/E
    peratio_col = find_col(cols, [r"\bperatio\b", r"\bpe_ratio\b", r"\bpe\b", r"\bpeRatio\b", r"\bpeRatio\b"])

    # PRIORITY: priceToSalesRatio column for P/S
    pstosales_col = find_col(cols, [r"pricetosalesratio", r"price_to_sales", r"ps_ratio", r"priceToSalesRatio", r"p_to_s"])

    # include discovered columns to keep list (so they survive copying)
    keep_cols = ["date"]
    keep_cols += [c for c in raw_to_pretty.keys() if c in df.columns]
    keep_cols += [c for c in pretty_set if c in df.columns]
    for extra in (price_col, shares_col, eps_col, eps_est_col, forwardpe_col, peratio_col, pstosales_col):
        if extra and extra not in keep_cols:
            keep_cols.append(extra)

    # dedupe preserve order
    keep_cols = [c for i, c in enumerate(keep_cols) if c not in keep_cols[:i]]

    # create working dataframe with only relevant cols
    work = df[keep_cols].copy()

    # rename known raw columns to pretty names where applicable
    rename_map = {raw: pretty for raw, pretty in raw_to_pretty.items() if raw in work.columns}
    work.rename(columns=rename_map, inplace=True)

    # Normalize important columns into known keys if they exist under other names
    # EPS Diluted
    if "EPS Diluted (GAAP)" not in work.columns and eps_col and eps_col in work.columns:
        work["EPS Diluted (GAAP)"] = work[eps_col]

    # forwardPE
    if "forwardPE" not in work.columns and forwardpe_col and forwardpe_col in work.columns:
        work["forwardPE"] = work[forwardpe_col]

    # epsEstimated
    if "epsEstimated" not in work.columns and eps_est_col and eps_est_col in work.columns:
        work["epsEstimated"] = work[eps_est_col]

    # close price
    if "close" not in work.columns and price_col and price_col in work.columns:
        work["close"] = work[price_col]

    # sharesOutstanding
    if "sharesOutstanding" not in work.columns and shares_col and shares_col in work.columns:
        work["sharesOutstanding"] = work[shares_col]

    # peratio and priceToSalesRatio normalization (keep original names, but also create normalized keys)
    if "peratio" not in work.columns and peratio_col and peratio_col in work.columns:
        work["peratio"] = work[peratio_col]
    if "priceToSalesRatio" not in work.columns and pstosales_col and pstosales_col in work.columns:
        work["priceToSalesRatio"] = work[pstosales_col]

    # now metrics order including valuation metrics
    metrics_order = [
        "Total Revenues",
        "Gross Profit",
        "Gross Margin",
        "Income from Operations",
        "Operating Margin",
        "GAAP Net Income",
        "EPS Diluted (GAAP)",
        "Adjusted EBITDA",
        "Adjusted EBITDA Margin",
        "Capital Expenditures",
        "Free Cash Flow",
        "P/E",
        "Forward P/E",
        "P/S",
    ]

    # Filter to dates, sort and select last N quarters
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    lastN = work.tail(num_quarters).copy()
    if lastN.empty:
        raise ValueError("No valid date rows for selected ticker/quarters.")

    # --- Compute valuation metrics (prefer CSV peratio/priceToSalesRatio when present) ---
    # Ensure numeric dtype for required computed columns
    def safe_get_series(w, col):
        return w[col] if col in w.columns else pd.Series([np.nan] * len(w), index=w.index)

    # series for computation
    price_ser = safe_get_series(lastN, "close").astype(float, errors="ignore")
    shares_ser = safe_get_series(lastN, "sharesOutstanding").astype(float, errors="ignore")
    eps_ser = safe_get_series(lastN, "EPS Diluted (GAAP)").astype(float, errors="ignore")
    eps_est_ser = safe_get_series(lastN, "epsEstimated").astype(float, errors="ignore")

    # 1) P/E: prefer peratio column if present
    lastN["P/E"] = np.nan
    if "peratio" in lastN.columns and lastN["peratio"].notna().any():
        # parse peratio if it contains 'x' or strings
        lastN.loc[lastN["peratio"].notna(), "P/E"] = lastN.loc[lastN["peratio"].notna(), "peratio"].apply(lambda x: parse_number(x) if not pd.api.types.is_numeric_dtype(type(x)) else x)
    else:
        # fallback: price / eps (latest quarterly EPS)
        mask_pe = price_ser.notna() & eps_ser.notna() & (eps_ser != 0)
        if mask_pe.any():
            lastN.loc[mask_pe, "P/E"] = price_ser.loc[mask_pe] / eps_ser.loc[mask_pe]

    # 2) P/S: prefer priceToSalesRatio column if present
    lastN["P/S"] = np.nan
    if "priceToSalesRatio" in lastN.columns and lastN["priceToSalesRatio"].notna().any():
        lastN.loc[lastN["priceToSalesRatio"].notna(), "P/S"] = lastN.loc[lastN["priceToSalesRatio"].notna(), "priceToSalesRatio"].apply(lambda x: parse_number(x) if not pd.api.types.is_numeric_dtype(type(x)) else x)
    else:
        # fallback compute marketcap / revenue (price * shares / revenue)
        mask_ps = price_ser.notna() & shares_ser.notna() & ("Total Revenues" in lastN.columns) & (lastN["Total Revenues"] != 0)
        if mask_ps.any():
            lastN.loc[mask_ps, "P/S"] = (price_ser.loc[mask_ps] * shares_ser.loc[mask_ps]) / lastN.loc[mask_ps, "Total Revenues"]

    # 3) Forward P/E: if forwardPE present in CSV use it; otherwise compute using formula:
    #    forward_p/e = p/e * (eps / epsEstimated)
    lastN["Forward P/E"] = np.nan
    if "forwardPE" in lastN.columns and lastN["forwardPE"].notna().any():
        lastN.loc[lastN["forwardPE"].notna(), "Forward P/E"] = lastN.loc[lastN["forwardPE"].notna(), "forwardPE"].apply(lambda x: parse_number(x) if not pd.api.types.is_numeric_dtype(type(x)) else x)
    else:
        # compute where possible
        mask_fpe_compute = lastN["P/E"].notna() & eps_ser.notna() & eps_est_ser.notna() & (eps_est_ser != 0)
        if mask_fpe_compute.any():
            p_over_e = lastN.loc[mask_fpe_compute, "P/E"].astype(float)
            eps_cur = eps_ser.loc[mask_fpe_compute].astype(float)
            eps_est = eps_est_ser.loc[mask_fpe_compute].astype(float)
            lastN.loc[mask_fpe_compute, "Forward P/E"] = p_over_e * (eps_cur / eps_est)

    # Rebuild metrics list dynamically from metrics_order but only include those present in lastN
    metrics = [m for m in metrics_order if m in lastN.columns]

    # Build pivot (metrics as rows, last N quarters ascending as columns)
    pivot = lastN.set_index("date")[metrics].T

    # --- Compute YoY for metrics where possible (only for latest quarter) ---
    latest_dt = lastN["date"].max()
    prev_year_dt = latest_dt - pd.DateOffset(years=1)
    prev_rows = work[work["date"] == prev_year_dt]
    if prev_rows.empty:
        prev_rows = work[
            (work["date"] >= prev_year_dt - pd.Timedelta(days=10))
            & (work["date"] <= prev_year_dt + pd.Timedelta(days=10))
        ]

    yoy_vals = []
    if not prev_rows.empty:
        for m in pivot.index:
            cur = work.loc[work["date"] == latest_dt, m] if m in work.columns else pd.Series([np.nan])
            prv = work.loc[work["date"].isin(prev_rows["date"]), m] if m in work.columns else pd.Series([np.nan])
            cur_val = cur.iloc[0] if not cur.empty else np.nan
            prv_val = prv.iloc[0] if not prv.empty else np.nan
            if pd.notna(cur_val) and pd.notna(prv_val) and prv_val != 0:
                yoy_vals.append((cur_val - prv_val) / prv_val * 100.0)
            else:
                yoy_vals.append(np.nan)
        pivot["YoY %"] = yoy_vals

    # Friendly column names for pivot
    pivot.columns = [c.strftime("%Y-%m-%d") if isinstance(c, (pd.Timestamp, np.datetime64)) else c for c in pivot.columns]

    # Build display-ready (formatted) pivot
    pivot_display = pivot.copy()
    for col in pivot_display.columns:
        if col == "YoY %":
            continue
        dt = pd.to_datetime(col)
        numeric_series = lastN.set_index("date")[metrics].loc[dt]
        pivot_display[col] = [fmt_value(v, m) for m, v in zip(pivot_display.index, numeric_series.values)]

    if "YoY %" in pivot_display.columns:
        pivot_display["YoY %"] = pivot["YoY %"].apply(lambda x: "-" if pd.isna(x) else f"{x:.1f}%")

    return {
        "pivot_numeric": pivot,
        "pivot_display": pivot_display,
        "parsed_columns": cols_to_parse,
        "coerced_numeric_columns": coerced_to_numeric,
    }


# ----------------------
# Comparison tools (Growth & Profitability)
# ----------------------
def compare_growth_tool(tickers: List[str]):
    """
    Compare growth metrics across multiple tickers.
    Returns: {"tables": [("Growth", df_display)]}
    """
    metrics = [
        "Revenue Growth (YoY)",
        "Revenue Growth (FWD TTM YoY)",
        "Revenue 3 Year (CAGR)",
        "Revenue 5 Year (CAGR)",
        "EBITDA Growth (YoY)",
        "EBITDA Growth (FWD TTM YoY)",
        "EBITDA 3 Year (CAGR)",
        "EBIT 3 Year (CAGR)",
        "Net Income 3 Year (CAGR)",
        "EPS Growth Diluted (YoY)",
        "EPS Growth Diluted (FWD TTM YoY)",
        "EPS Diluted 3 Year (CAGR)",
        "Tang Book Value 3 Year (CAGR)",
        "Total Assets 3 Year (CAGR)",
        "Levered FCF 3 Year (CAGR)"
    ]

    results = {m: {} for m in metrics}
    tickers_upper = [t.strip().upper() for t in tickers]

    # Ensure all required data files exist, fetch if missing
    missing = [t for t in tickers_upper if not os.path.exists(os.path.join("data", f"{t}.csv"))]
    if missing:
        try:
            obtain_api_data_tool(missing, path_to_save="data")
        except Exception as e:
            pass  # If fetch fails, will skip those tickers below

    for ticker in tickers_upper:
        path = os.path.join("data", f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str)
        if "date" not in df.columns:
            continue
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper() == ticker]
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        def get_col(name):
            return df[name].apply(_parse_number_str) if name in df.columns else pd.Series([np.nan] * len(df))

        revenue_s = get_col("revenue")
        ebitda_s = get_col("ebitda")
        ebit_s = get_col("operatingIncome")
        net_s = get_col("netIncome")
        eps_s = get_col("epsdiluted")
        fcf_s = get_col("freeCashFlow")
        opcf_s = get_col("operatingCashFlow")
        capex_s = get_col("capitalExpenditure")
        assets_s = get_col("totalAssets") if "totalAssets" in df.columns else get_col("assets")
        equity_s = get_col("totalStockholdersEquity") if "totalStockholdersEquity" in df.columns else get_col("totalEquity")
        tang_book_s = get_col("tangibleBookValue") if "tangibleBookValue" in df.columns else (equity_s - get_col("intangibleAssets") if "intangibleAssets" in df.columns else pd.Series([np.nan] * len(df)))

        latest_idx = df["date"].idxmax()
        latest_dt = df.loc[latest_idx, "date"]
        prev_year_dt = latest_dt - pd.DateOffset(years=1)

        # TTM sums (last 4 rows assumed quarterly)
        ttm_n = 4
        if len(df) >= ttm_n:
            ttm_revenue = revenue_s.tail(ttm_n).sum()
            ttm_revenue_prev = revenue_s.head(len(df)-ttm_n).tail(ttm_n).sum() if len(df) >= ttm_n*2 else np.nan
            ttm_ebitda = ebitda_s.tail(ttm_n).sum()
            ttm_ebitda_prev = ebitda_s.head(len(df)-ttm_n).tail(ttm_n).sum() if len(df) >= ttm_n*2 else np.nan
            ttm_eps = eps_s.tail(ttm_n).sum()
            ttm_eps_prev = eps_s.head(len(df)-ttm_n).tail(ttm_n).sum() if len(df) >= ttm_n*2 else np.nan
            ttm_fcf = fcf_s.tail(ttm_n).sum()
            ttm_fcf_prev = fcf_s.head(len(df)-ttm_n).tail(ttm_n).sum() if len(df) >= ttm_n*2 else np.nan
        else:
            ttm_revenue = ttm_revenue_prev = ttm_ebitda = ttm_ebitda_prev = ttm_eps = ttm_eps_prev = ttm_fcf = ttm_fcf_prev = np.nan

        # helper to get same quarter prev year value (or ±10 day)
        same_prev_mask = df["date"] == prev_year_dt
        if same_prev_mask.any():
            prev_revenue = revenue_s[same_prev_mask].iloc[0]
            prev_ebitda = ebitda_s[same_prev_mask].iloc[0] if not ebitda_s.isna().all() else np.nan
            prev_eps = eps_s[same_prev_mask].iloc[0] if not eps_s.isna().all() else np.nan
        else:
            window = df[(df["date"] >= prev_year_dt - pd.Timedelta(days=10)) & (df["date"] <= prev_year_dt + pd.Timedelta(days=10))]
            prev_revenue = revenue_s.loc[window.index[0]] if not window.empty else np.nan
            prev_ebitda = ebitda_s.loc[window.index[0]] if not window.empty else np.nan
            prev_eps = eps_s.loc[window.index[0]] if not window.empty else np.nan

        cur_revenue = revenue_s.iloc[latest_idx] if not revenue_s.isna().all() else np.nan
        revenue_yoy = (cur_revenue - prev_revenue)/prev_revenue if pd.notna(cur_revenue) and pd.notna(prev_revenue) and prev_revenue != 0 else np.nan
        revenue_fwd_yoy = (ttm_revenue - ttm_revenue_prev)/ttm_revenue_prev if pd.notna(ttm_revenue) and pd.notna(ttm_revenue_prev) and ttm_revenue_prev != 0 else np.nan

        # CAGR calculations
        def value_at_years_ago(series, dates, years):
            target = dates.max() - pd.DateOffset(years=years)
            match = series[dates == target]
            if not match.empty:
                return match.iloc[0]
            window = series[(dates >= target - pd.Timedelta(days=30)) & (dates <= target + pd.Timedelta(days=30))]
            if not window.empty:
                return window.iloc[0]
            return np.nan

        rev_3_start = value_at_years_ago(revenue_s, df["date"], 3)
        rev_3_cagr = _cagr(rev_3_start, cur_revenue, 3)
        rev_5_start = value_at_years_ago(revenue_s, df["date"], 5)
        rev_5_cagr = _cagr(rev_5_start, cur_revenue, 5)

        # EBITDA YoY & CAGR
        cur_ebitda = ebitda_s.iloc[latest_idx] if not ebitda_s.isna().all() else np.nan
        ebitda_yoy = (cur_ebitda - prev_ebitda)/prev_ebitda if pd.notna(cur_ebitda) and pd.notna(prev_ebitda) and prev_ebitda != 0 else np.nan
        ebitda_fwd_yoy = (ttm_ebitda - ttm_ebitda_prev)/ttm_ebitda_prev if pd.notna(ttm_ebitda) and pd.notna(ttm_ebitda_prev) and ttm_ebitda_prev != 0 else np.nan
        ebitda_3_cagr = _cagr(value_at_years_ago(ebitda_s, df["date"], 3), cur_ebitda, 3)

        ebit_3_cagr = _cagr(value_at_years_ago(ebit_s, df["date"], 3), ebit_s.iloc[latest_idx] if not ebit_s.isna().all() else np.nan, 3)
        net_3_cagr = _cagr(value_at_years_ago(net_s, df["date"], 3), net_s.iloc[latest_idx] if not net_s.isna().all() else np.nan, 3)

        # EPS
        cur_eps = eps_s.iloc[latest_idx] if not eps_s.isna().all() else np.nan
        eps_yoy = (cur_eps - prev_eps)/prev_eps if pd.notna(cur_eps) and pd.notna(prev_eps) and prev_eps != 0 else np.nan
        eps_fwd_yoy = (ttm_eps - ttm_eps_prev)/ttm_eps_prev if pd.notna(ttm_eps) and pd.notna(ttm_eps_prev) and ttm_eps_prev != 0 else np.nan
        eps_3_cagr = _cagr(value_at_years_ago(eps_s, df["date"], 3), cur_eps, 3)

        # Tangible book value 3y
        tang_3_cagr = _cagr(value_at_years_ago(tang_book_s, df["date"], 3), tang_book_s.iloc[latest_idx] if not tang_book_s.isna().all() else np.nan, 3)

        assets_3_cagr = _cagr(value_at_years_ago(assets_s, df["date"], 3), assets_s.iloc[latest_idx] if not assets_s.isna().all() else np.nan, 3)

        # Levered FCF 3y
        if fcf_s.notna().any():
            fcf_3_cagr = _cagr(value_at_years_ago(fcf_s, df["date"], 3), fcf_s.iloc[latest_idx] if not fcf_s.isna().all() else np.nan, 3)
        else:
            levered = opcf_s - capex_s if (opcf_s.notna().any() and capex_s.notna().any()) else pd.Series([np.nan]*len(df))
            fcf_3_cagr = _cagr(value_at_years_ago(levered, df["date"], 3), levered.iloc[latest_idx] if levered.notna().any() else np.nan, 3)

        # store results
        results["Revenue Growth (YoY)"][ticker] = revenue_yoy
        results["Revenue Growth (FWD TTM YoY)"][ticker] = revenue_fwd_yoy
        results["Revenue 3 Year (CAGR)"][ticker] = rev_3_cagr
        results["Revenue 5 Year (CAGR)"][ticker] = rev_5_cagr
        results["EBITDA Growth (YoY)"][ticker] = ebitda_yoy
        results["EBITDA Growth (FWD TTM YoY)"][ticker] = ebitda_fwd_yoy
        results["EBITDA 3 Year (CAGR)"][ticker] = ebitda_3_cagr
        results["EBIT 3 Year (CAGR)"][ticker] = ebit_3_cagr
        results["Net Income 3 Year (CAGR)"][ticker] = net_3_cagr
        results["EPS Growth Diluted (YoY)"][ticker] = eps_yoy
        results["EPS Growth Diluted (FWD TTM YoY)"][ticker] = eps_fwd_yoy
        results["EPS Diluted 3 Year (CAGR)"][ticker] = eps_3_cagr
        results["Tang Book Value 3 Year (CAGR)"][ticker] = tang_3_cagr
        results["Total Assets 3 Year (CAGR)"][ticker] = assets_3_cagr
        results["Levered FCF 3 Year (CAGR)"][ticker] = fcf_3_cagr

    # Build DataFrame: metrics rows, tickers columns
    tickers_present = sorted({t.strip().upper() for t in tickers if os.path.exists(os.path.join("data", f"{t.strip().upper()}.csv"))})
    if not tickers_present:
        return {"tables": [("Growth", pd.DataFrame())]}

    df_out = pd.DataFrame({t: {m: results[m].get(t, np.nan) for m in metrics} for t in tickers_present})

    # Format display DataFrame
    df_display = pd.DataFrame(index=metrics)
    for col in df_out.columns:
        formatted = []
        for m, v in df_out[col].items():
            if "Growth" in m or "CAGR" in m or "EPS" in m:
                formatted.append(_human_pct(v))
            else:
                formatted.append(_human_money(v))
        df_display[col] = formatted
    df_display.index = metrics

    return {"tables": [("Growth", df_display)]}


def compare_profitability_tool(tickers: List[str]):
    """
    Compare profitability metrics across multiple tickers.
    Returns: {"tables":[("Profitability", df_display)]}
    """
    metrics = [
        "Gross Profit Margin",
        "EBIT Margin",
        "EBITDA Margin",
        "Net Income Margin",
        "Levered FCF Margin",
        "Return on Equity",
        "Return on Assets",
        "Return on Total Capital",
        "Cash From Operations (TTM)",
        "Revenue Per Employee",
        "Net Income Per Employee",
        "Asset Turnover"
    ]

    results = {m: {} for m in metrics}
    tickers_upper = [t.strip().upper() for t in tickers]

    # Ensure all required data files exist, fetch if missing
    missing = [t for t in tickers_upper if not os.path.exists(os.path.join("data", f"{t}.csv"))]
    if missing:
        try:
            obtain_api_data_tool(missing, path_to_save="data")
        except Exception as e:
            pass  # If fetch fails, will skip those tickers below

    for ticker in tickers_upper:
        path = os.path.join("data", f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str)
        if "date" not in df.columns:
            continue
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper() == ticker]
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        def get_col(name):
            return df[name].apply(_parse_number_str) if name in df.columns else pd.Series([np.nan]*len(df))

        revenue_s = get_col("revenue")
        gross_s = get_col("grossProfit")
        ebitda_s = get_col("ebitda")
        ebit_s = get_col("operatingIncome")
        net_s = get_col("netIncome")
        fcf_s = get_col("freeCashFlow")
        opcf_s = get_col("operatingCashFlow")
        capex_s = get_col("capitalExpenditure")
        assets_s = get_col("totalAssets") if "totalAssets" in df.columns else get_col("assets")
        equity_s = get_col("totalStockholdersEquity") if "totalStockholdersEquity" in df.columns else get_col("totalEquity")
        employees_s = get_col("fullTimeEmployees") if "fullTimeEmployees" in df.columns else get_col("employees")

        latest_idx = df["date"].idxmax()
        # latest values
        rev_cur = revenue_s.iloc[latest_idx] if not revenue_s.isna().all() else np.nan
        gross_cur = gross_s.iloc[latest_idx] if not gross_s.isna().all() else np.nan
        ebitda_cur = ebitda_s.iloc[latest_idx] if not ebitda_s.isna().all() else np.nan
        ebit_cur = ebit_s.iloc[latest_idx] if not ebit_s.isna().all() else np.nan
        net_cur = net_s.iloc[latest_idx] if not net_s.isna().all() else np.nan
        fcf_cur = fcf_s.iloc[latest_idx] if not fcf_s.isna().all() else np.nan

        gross_margin = gross_cur / rev_cur if pd.notna(gross_cur) and pd.notna(rev_cur) and rev_cur != 0 else np.nan
        ebit_margin = ebit_cur / rev_cur if pd.notna(ebit_cur) and pd.notna(rev_cur) and rev_cur != 0 else np.nan
        ebitda_margin = ebitda_cur / rev_cur if pd.notna(ebitda_cur) and pd.notna(rev_cur) and rev_cur != 0 else np.nan
        net_margin = net_cur / rev_cur if pd.notna(net_cur) and pd.notna(rev_cur) and rev_cur != 0 else np.nan

        # Levered FCF margin
        if not fcf_s.isna().all():
            levered_fcf = fcf_cur
        elif not opcf_s.isna().all() and not capex_s.isna().all():
            levered_fcf = opcf_s.iloc[latest_idx] - capex_s.iloc[latest_idx]
        else:
            levered_fcf = np.nan
        levered_fcf_margin = levered_fcf / rev_cur if pd.notna(levered_fcf) and pd.notna(rev_cur) and rev_cur != 0 else np.nan

        roe = net_cur / equity_s.iloc[latest_idx] if pd.notna(net_cur) and pd.notna(equity_s.iloc[latest_idx]) and equity_s.iloc[latest_idx] != 0 else np.nan
        roa = net_cur / assets_s.iloc[latest_idx] if pd.notna(net_cur) and pd.notna(assets_s.iloc[latest_idx]) and assets_s.iloc[latest_idx] != 0 else np.nan

        total_capital = None
        debt_s = get_col("totalDebt") if "totalDebt" in df.columns else (get_col("longTermDebt") + get_col("shortTermDebt") if ("longTermDebt" in df.columns or "shortTermDebt" in df.columns) else pd.Series([np.nan]*len(df)))
        if pd.notna(equity_s.iloc[latest_idx]) and pd.notna(debt_s.iloc[latest_idx]):
            total_capital = equity_s.iloc[latest_idx] + debt_s.iloc[latest_idx]
        elif pd.notna(assets_s.iloc[latest_idx]):
            total_capital = assets_s.iloc[latest_idx]
        rotc = net_cur / total_capital if pd.notna(net_cur) and pd.notna(total_capital) and total_capital != 0 else np.nan

        # Cash from operations (TTM)
        if len(df) >= 4:
            cfo_ttm = opcf_s.tail(4).sum()
        else:
            cfo_ttm = opcf_s.iloc[latest_idx] if not opcf_s.isna().all() else np.nan

        emp_cur = employees_s.iloc[latest_idx] if not employees_s.isna().all() else np.nan
        rev_per_emp = rev_cur / emp_cur if pd.notna(rev_cur) and pd.notna(emp_cur) and emp_cur != 0 else np.nan
        net_per_emp = net_cur / emp_cur if pd.notna(net_cur) and pd.notna(emp_cur) and emp_cur != 0 else np.nan

        asset_turnover = rev_cur / assets_s.iloc[latest_idx] if pd.notna(rev_cur) and pd.notna(assets_s.iloc[latest_idx]) and assets_s.iloc[latest_idx] != 0 else np.nan

        results["Gross Profit Margin"][ticker] = gross_margin
        results["EBIT Margin"][ticker] = ebit_margin
        results["EBITDA Margin"][ticker] = ebitda_margin
        results["Net Income Margin"][ticker] = net_margin
        results["Levered FCF Margin"][ticker] = levered_fcf_margin
        results["Return on Equity"][ticker] = roe
        results["Return on Assets"][ticker] = roa
        results["Return on Total Capital"][ticker] = rotc
        results["Cash From Operations (TTM)"][ticker] = cfo_ttm
        results["Revenue Per Employee"][ticker] = rev_per_emp
        results["Net Income Per Employee"][ticker] = net_per_emp
        results["Asset Turnover"][ticker] = asset_turnover

    tickers_present = sorted({t.strip().upper() for t in tickers if os.path.exists(os.path.join("data", f"{t.strip().upper()}.csv"))})
    if not tickers_present:
        return {"tables": [("Profitability", pd.DataFrame())]}

    df_out = pd.DataFrame({t: {m: results[m].get(t, np.nan) for m in metrics} for t in tickers_present})

    df_display = pd.DataFrame(index=metrics)
    for col in df_out.columns:
        formatted = []
        for m, v in df_out[col].items():
            if "Margin" in m or "Return" in m:
                formatted.append(_human_pct(v))
            elif "Cash From Operations" in m:
                formatted.append(_human_money(v))
            elif "Per Employee" in m:
                formatted.append(_human_money(v))
            elif "Asset Turnover" in m:
                formatted.append(_human_plain(v, decimals=2))
            else:
                formatted.append(_human_money(v))
        df_display[col] = formatted
    df_display.index = metrics

    return {"tables": [("Profitability", df_display)]}

def compare_valuation_tool(tickers: List[str]):
    """
    Generate a valuation comparison table across multiple tickers.
    Reads data/<TICKER>.csv for each ticker, auto-detects fields,
    computes TTM and Forward valuation ratios, and returns a display table.

    Returns: {"tables":[("Valuation", df_display)]}
    """
    metrics = [
        "P/E GAAP (TTM)",
        "P/E GAAP (FWD)",
        "P/S (TTM)",
        "EV/Sales (TTM)",
        "EV/Sales (FWD)",
        "EV/EBITDA (TTM)",
        "EV/EBITDA (FWD)",
        "Price/Book (TTM)",
        "Price/Cash Flow (TTM)",
        "PEG (TTM)",
        "PEG (FWD)",
    ]

    results = {m: {} for m in metrics}
    tickers_upper = [t.strip().upper() for t in tickers]

    # Ensure all required data files exist, fetch if missing
    missing = [t for t in tickers_upper if not os.path.exists(os.path.join("data", f"{t}.csv"))]
    if missing:
        try:
            obtain_api_data_tool(missing, path_to_save="data")
        except Exception as e:
            pass  # If fetch fails, will skip those tickers below

    for ticker in tickers_upper:
        path = os.path.join("data", f"{ticker}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, dtype=str)
        if "date" not in df.columns:
            # try lowercase date column
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            else:
                continue

        # filter by symbol column if present
        if "symbol" in df.columns:
            df = df[df["symbol"].astype(str).str.upper() == ticker]

        if df.empty:
            continue

        # parse dates and sort
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if df.empty:
            continue

        def get_col(name):
            """
            Case-insensitive column lookup + numeric parsing.
            Returns a pd.Series aligned with df.index containing parsed numbers or NaN.
            """
            name_lower = name.lower()
            match = None
            for col in df.columns:
                if col.lower() == name_lower:
                    match = col
                    break

            if match is None:
                return pd.Series([np.nan] * len(df), index=df.index)
            return df[match].apply(_parse_number_str)

        # possible column names (common)
        ev_s = get_col("enterpriseValue")
        eps_s = get_col("eps")
        eps_est_s = get_col("epsEstimated") if "epsEstimated" in df.columns else get_col("epsEstimate")
        rev_s = get_col("revenue")
        rev_est_s = get_col("revenueEstimated") if "revenueEstimated" in df.columns else get_col("revenueEstimated")
        ebitda_s = get_col("ebitda")
        ebitda_est_s = get_col("ebitdaEstimated") if "ebitdaEstimated" in df.columns else get_col("ebitdaEstimate")
        pe_ttm_s = get_col("peratio")
        ps_ttm_s = get_col("priceToSalesRatio")
        pb_ttm_s = get_col("pbRatio")
        mkt_cap_s = get_col("marketCap")
        ocf_s = get_col("operatingCashFlow")

        latest_idx = df["date"].idxmax()

        ev = ev_s.iloc[latest_idx] if not ev_s.isna().all() else np.nan
        eps = eps_s.iloc[latest_idx] if not eps_s.isna().all() else np.nan
        eps_est = eps_est_s.iloc[latest_idx] if not eps_est_s.isna().all() else np.nan
        rev = rev_s.iloc[latest_idx] if not rev_s.isna().all() else np.nan
        rev_est = rev_est_s.iloc[latest_idx] if not rev_est_s.isna().all() else np.nan
        ebitda = ebitda_s.iloc[latest_idx] if not ebitda_s.isna().all() else np.nan
        ebitda_est = ebitda_est_s.iloc[latest_idx] if not ebitda_est_s.isna().all() else np.nan
        pe_ttm = pe_ttm_s.iloc[latest_idx] if not pe_ttm_s.isna().all() else np.nan
        ps_ttm = ps_ttm_s.iloc[latest_idx] if not ps_ttm_s.isna().all() else np.nan
        pb_ttm = pb_ttm_s.iloc[latest_idx] if not pb_ttm_s.isna().all() else np.nan
        mkt_cap = mkt_cap_s.iloc[latest_idx] if not mkt_cap_s.isna().all() else np.nan
        ocf = ocf_s.iloc[latest_idx] if not ocf_s.isna().all() else np.nan

        # --- Compute metrics (with safety checks) ---
        # P/E TTM
        results["P/E GAAP (TTM)"][ticker] = pe_ttm if not pd.isna(pe_ttm) else np.nan

        # Forward P/E: convert current PE to forward via eps -> eps_est:
        # forward_pe = price / eps_est = (price/eps) * (eps/eps_est) = pe_ttm * (eps / eps_est)
        if pd.notna(pe_ttm) and pd.notna(eps) and pd.notna(eps_est) and eps_est != 0:
            # allow eps to be zero (would be invalid) - check eps != 0
            if eps != 0:
                results["P/E GAAP (FWD)"][ticker] = pe_ttm * (eps / eps_est)
            else:
                # if eps == 0 but we have price info via marketCap & shares we could compute, but keep safe
                results["P/E GAAP (FWD)"][ticker] = np.nan
        else:
            results["P/E GAAP (FWD)"][ticker] = np.nan

        # P/S (TTM)
        results["P/S (TTM)"][ticker] = ps_ttm if not pd.isna(ps_ttm) else np.nan

        # EV / Sales TTM
        if pd.notna(ev) and pd.notna(rev) and rev != 0:
            results["EV/Sales (TTM)"][ticker] = ev / rev
        else:
            results["EV/Sales (TTM)"][ticker] = np.nan

        # EV / Sales FWD (use revenueEstimated)
        if pd.notna(ev) and pd.notna(rev_est) and rev_est != 0:
            results["EV/Sales (FWD)"][ticker] = ev / rev_est
        else:
            results["EV/Sales (FWD)"][ticker] = np.nan

        # EV/EBITDA TTM
        if pd.notna(ev) and pd.notna(ebitda) and ebitda != 0:
            results["EV/EBITDA (TTM)"][ticker] = ev / ebitda
        else:
            results["EV/EBITDA (TTM)"][ticker] = np.nan

        # EV/EBITDA FWD
        if pd.notna(ev) and pd.notna(ebitda_est) and ebitda_est != 0:
            results["EV/EBITDA (FWD)"][ticker] = ev / ebitda_est
        else:
            results["EV/EBITDA (FWD)"][ticker] = np.nan

        # Price / Book
        results["Price/Book (TTM)"][ticker] = pb_ttm if not pd.isna(pb_ttm) else np.nan

        # Price / Cash Flow (TTM) using market cap vs operating cash flow
        if pd.notna(mkt_cap) and pd.notna(ocf) and ocf != 0:
            results["Price/Cash Flow (TTM)"][ticker] = mkt_cap / ocf
        else:
            results["Price/Cash Flow (TTM)"][ticker] = np.nan

        # PEG: not computed here because growth input missing; set NaN
        results["PEG (TTM)"][ticker] = np.nan
        results["PEG (FWD)"][ticker] = np.nan

    # prepare output table
    tickers_present = sorted({t for t in tickers_upper if os.path.exists(os.path.join("data", f"{t}.csv"))})
    if not tickers_present:
        return {"tables": [("Valuation", pd.DataFrame())]}

    df_out = pd.DataFrame({t: {m: results[m].get(t, np.nan) for m in metrics} for t in tickers_present})

    # formatted display dataframe
    df_display = pd.DataFrame(index=metrics)
    for col in df_out.columns:
        formatted = []
        for m, v in df_out[col].items():
            # formatting: numeric ratios -> 2 decimals, missing -> "N/A"
            if pd.isna(v):
                formatted.append("N/A")
            else:
                # ratios like P/E, EV/..., P/S, Price/Book -> plain numbers
                formatted.append(_human_plain(v, decimals=2))
        df_display[col] = formatted
    df_display.index = metrics

    return {"tables": [("Valuation", df_display)]}

# ----------------------
# Resilient Yahoo helper (kept)
# ----------------------
"""
def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
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
    session = _make_session()
    params = {
        "q": query,
        "count": max(1, int(count)),
        "start": 0,
        "region": region,
        "lang": lang,
    }
    hosts = [
        "https://query1.finance.yahoo.com/v1/news/search",
        "https://query2.finance.yahoo.com/v1/news/search",
    ]
    last_err = None
    for host in hosts:
        try:
            time.sleep(random.uniform(0.2, 0.6))
            r = session.get(host, params=params, timeout=12)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                        time.sleep(wait + random.uniform(0.2, 0.8))
                        r = session.get(host, params=params, timeout=12)
                    except Exception:
                        pass
            r.raise_for_status()
            data = r.json()
            raw = data.get("items") or data.get("news") or []
            items = [{
                "title": n.get("title"),
                "publisher": n.get("publisher"),
                "link": n.get("link") or n.get("url"),
                "time": n.get("pubDate") or n.get("providerPublishTime"),
            } for n in raw]
            items.sort(key=lambda x: (x["time"] or 0), reverse=True)
            return items[:count]
        except requests.RequestException as e:
            last_err = e
            time.sleep(random.uniform(0.5, 1.2))
    raise RuntimeError(f"Yahoo Finance news search failed after retries: {last_err}")
"""
# def finance_news_digest_tool(
#     query: str,
#     top_n: int = 5,
#     max_articles_chars: int = 18000,
# ) -> Dict:
#     """
#     Self-contained: searches Yahoo Finance news for `query`, fetches top articles,
#     extracts text, and summarizes with GPT. Requires OPENAI_API_KEY in env.
#     Returns: {'text': str, 'sources': List[...], 'image_path': None}
#     """

#     def _clean_url(u: str) -> str:
#         try:
#             p = urlparse(u)
#             q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
#                  if not k.lower().startswith("utm_")]
#             return urlunparse(p._replace(query=urlencode(q)))
#         except Exception:
#             return u

#     def _extract_article_text(url: str, timeout: int = 12) -> str:
#         headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceNewsDigest/1.0)"}
#         r = requests.get(url, headers=headers, timeout=timeout)
#         r.raise_for_status()
#         soup = BeautifulSoup(r.text, "html.parser")
#         # Try common article containers
#         selectors = [
#             "article",
#             "[itemprop='articleBody']",
#             ".article-content", ".post-content", ".story-body", ".entry-content",
#             ".meteredContent", ".paywall"
#         ]
#         for sel in selectors:
#             el = soup.select_one(sel)
#             if el and el.get_text(strip=True):
#                 ps = el.find_all(["p","h2","li"])
#                 text = "\n".join(p.get_text(" ", strip=True) for p in ps) or el.get_text(" ", strip=True)
#                 if len(text) > 400:
#                     return text.strip()
#         # Fallback: all <p>
#         ps = soup.find_all("p")
#         return "\n".join(p.get_text(" ", strip=True) for p in ps).strip()

#     # 1) Search Yahoo Finance news
#     q = (query or "").strip()
#     if not q:
#         return {"text": "Please provide a non-empty query.", "sources": [], "image_path": None}

#     url = "https://query1.finance.yahoo.com/v1/finance/search"
#     params = {"q": q, "quotesCount": 0, "newsCount": max(1, int(top_n)), "region": "US", "lang": "en-US"}
#     try:
#         hits = _yahoo_news_search(query, count=top_n)
#     except Exception as e:
#         return {"text": f"Yahoo Finance news search failed: {e}", "sources": [], "image_path": None}

#     # 2) Fetch & extract article text
#     articles, total_chars = [], 0
#     for h in hits:
#         link = h["link"]
#         if not link:
#             continue
#         try:
#             txt = _extract_article_text(link)
#         except Exception:
#             txt = ""
#         if len(txt) < 500:
#             continue
#         if total_chars + len(txt) > max_articles_chars:
#             break
#         articles.append({"title": h["title"], "publisher": h["publisher"], "url": link, "text": txt})
#         total_chars += len(txt)
#         time.sleep(0.35)  # polite pause

#     if not articles:
#         return {"text": f"No readable article bodies fetched for '{query}'.", "sources": hits, "image_path": None}

#     # 3) Summarize with GPT (self-contained client)
#     if OpenAI is None:
#         return {"text": "OpenAI SDK not installed. `pip install openai`.", "sources": hits, "image_path": None}
#     if not os.getenv("OPENAI_API_KEY"):
#         return {"text": "Missing OPENAI_API_KEY in environment.", "sources": hits, "image_path": None}

#     client = OpenAI()
#     system = (
#         "You are a financial news analyst. Produce a concise, neutral brief that integrates multiple sources. "
#         "Quantify key facts and cite sources inline with [n] where n is the article index."
#     )
#     sources_block = "\n\n".join(
#         f"[{i+1}] {a['title']} — {a['publisher']}\nURL: {a['url']}\n---\n{a['text'][:6000]}"
#         for i, a in enumerate(articles)
#     )
#     user = (
#         f"Topic: {query}\n\nSources (truncated content below):\n{sources_block}\n\n"
#         "Write:\n"
#         "- 5–8 bullet executive summary\n"
#         "- What’s new vs background\n"
#         "- Market impact (tickers/sectors) with key numbers\n"
#         "- Risks/unknowns\n"
#         "- One-sentence takeaway\n"
#         "Keep ~300 words. Use [n] to cite."
#     )

#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role":"system","content":system},{"role":"user","content":user}],
#         temperature=0.3,
#     )
#     summary = (resp.choices[0].message.content or "").strip()

#     return {
#         "text": summary,
#         "sources": [{"index": i+1, "title": a["title"], "publisher": a["publisher"], "url": a["url"]} for i, a in enumerate(articles)],
#         "image_path": None
#     }


