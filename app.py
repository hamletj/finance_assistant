# app.py
import streamlit as st
import os
import json
import pandas as pd
from openai import OpenAI

# import your tools (including the two new ones)
from tools import (
    moving_average_tool,
    past_history_tool,
    generate_financial_summary_tool,
    compare_growth_tool,
    compare_profitability_tool,
    compare_valuation_tool,
    obtain_api_data_tool,
)

st.set_page_config(page_title="Finance AI Assistant", page_icon="💹", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY as an environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Define the tools metadata for the LLM (add compare_growth and compare_profitability)
FUNCTIONS = [
    {
        "name": "obtain_api_data_tool",
        "description": "Fetches quarterly data for one or more tickers from FinancialModelingPrep (FMP) API, combines income statement, key metrics, and earnings calendar, saves per-ticker CSVs, and returns a concatenated DataFrame.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}, "description": "List of stock tickers, e.g., ['AAPL','MSFT'] or a single ticker as string."},
                "path_to_save": {"type": "string", "description": "Directory to save per-ticker CSVs. If omitted, no files are written.", "default": None},
                "from_date": {"type": "string", "description": "Optional 'from' date parameter (YYYY-MM-DD).", "default": None},
                "api_key": {"type": "string", "description": "FMP API key. If omitted, uses env var FMP_API_KEY.", "default": None},
                "max_retries": {"type": "integer", "description": "Number of retries for network requests.", "default": 3},
                "retry_backoff": {"type": "number", "description": "Base backoff seconds (exponential backoff applied).", "default": 0.8},
                "verbose": {"type": "boolean", "description": "Print progress messages if True.", "default": True},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "generate_financial_summary_tool",
        "description": "For a given ticker / stock, generate a table to display its quarterly key metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g., AAPL"},
                "num_quarters": {"type": "integer", "description": "e.g., 5, 10"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "past_history_tool",
        "description": "Fetch financial data and plot closing prices.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g., AAPL"},
                "period": {"type": "string", "description": "e.g., '1mo', '3mo', '1y'"},
                "interval": {"type": "string", "description": "e.g., '1d', '1wk'"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "moving_average_tool",
        "description": "Fetch data and plot Close with one or more simple moving averages (SMA).",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g., AAPL"},
                "period": {"type": "string", "description": "e.g., '1mo', '3mo', '1y'"},
                "interval": {"type": "string", "description": "e.g., '1d', '1wk', '1h'"},
                "windows": {
                    "type": "array",
                    "description": "SMA window sizes in periods (integers > 1), e.g., [5, 20, 50].",
                    "items": {"type": "integer", "minimum": 2},
                    "minItems": 1,
                    "default": [20],
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_growth_tool",
        "description": "Compare growth metrics across multiple tickers.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock tickers, e.g., ['AAPL','MSFT','GOOG']",
                }
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "compare_profitability_tool",
        "description": "Compare profitability metrics across multiple tickers.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock tickers, e.g., ['AAPL','MSFT','GOOG']",
                }
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "compare_valuation_tool",
        "description": "Compare valuation metrics across multiple tickers. The metrics include P/E, P/S etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock tickers, e.g., ['AAPL','MSFT','GOOG']",
                }
            },
            "required": ["tickers"],
        },
    },
]

SYSTEM_PROMPT = "You are a finance assistant. If it's a general question, you don't need to select a tool. Just answer. If it's a question highly related to the tools, select tools to fulfill requests."

def run_agent(user_input: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    tool_registry = {
        "past_history_tool": past_history_tool,
        "moving_average_tool": moving_average_tool,
        "generate_financial_summary_tool": generate_financial_summary_tool,
        "compare_growth_tool": compare_growth_tool,
        "compare_profitability_tool": compare_profitability_tool,
        "compare_valuation_tool": compare_valuation_tool,
        "obtain_api_data_tool": obtain_api_data_tool,
    }
    # Try to dynamically attach the earnings transcript tool if available (avoid ImportError at module import time)
    try:
        from importlib import import_module
        tools_mod = import_module("tools")
        eet = getattr(tools_mod, "obtain_earnings_transcript_tool", None)
        if eet:
            tool_registry["obtain_earnings_transcript_tool"] = eet
    except Exception:
        # ignore; tool simply won't be available at runtime
        pass

    resp = client.chat.completions.create(
        model="gpt-5-mini",  # keep your chosen model
        messages=messages,
        functions=FUNCTIONS,
        function_call="auto",
    )

    message = resp.choices[0].message

    if getattr(message, "function_call", None):
        name = message.function_call.name
        raw_args = message.function_call.arguments or "{}"

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"text": f"Could not parse tool arguments for {name}: {raw_args}"}

        tool_fn = tool_registry.get(name)
        if tool_fn is None:
            return {"text": f"Unknown tool requested: {name}"}

        try:
            tool_result = tool_fn(**args)
        except TypeError as e:
            return {"text": f"Invalid arguments for {name}: {e}"}
        except Exception as e:
            return {"text": f"Error running {name}: {e}"}

        return tool_result

    return {"text": (message.content or "").strip()}


# ---------------- Streamlit UI ----------------
st.title("💹 Finance AI Assistant — Tool-enabled")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me something like 'summarize A' or 'compare growth of AAPL, MSFT, NVDA'...")

if user_input:
    with st.spinner("Thinking..."):
        result = run_agent(user_input)
    entry = {"user": user_input}
    if isinstance(result, dict):
        entry.update(result)
    else:
        entry["text"] = str(result)
    st.session_state.history.append(entry)

# Render conversation history
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat.get("user", ""))

    with st.chat_message("assistant"):
        # 1) plain text (if any)
        if chat.get("text"):
            st.write(chat["text"])

        # 2) formatted financial summary (existing)
        pivot_display = chat.get("pivot_display")
        if pivot_display is not None:
            try:
                st.subheader("Financial Summary")
                if hasattr(pivot_display, "to_html"):
                    st.markdown(pivot_display.to_html(), unsafe_allow_html=True)
                else:
                    st.dataframe(pivot_display)
            except Exception as e:
                st.error(f"Could not render financial summary: {e}")

        # 3) any tables returned by compare tools (list of (title, DataFrame))
        tables = chat.get("tables")
        if tables:
            for title, table in tables:
                try:
                    st.subheader(title)
                    # if the tool returned a dict or unexpected object, just show it
                    if isinstance(table, dict):
                        st.write(table)
                    else:
                        st.dataframe(table)
                except Exception as e:
                    st.write(f"Could not render table '{title}': {e}")

        # 4) image or plot
        if chat.get("image_path"):
            try:
                st.image(chat["image_path"])
            except Exception as e:
                st.write("Could not display image:", e)

st.markdown("---")
st.caption("This app routes user queries to tools. Use natural language (e.g., 'compare growth of AAPL, MSFT').")
