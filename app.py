# app.py
import streamlit as st
import os
import json
import pandas as pd
from openai import OpenAI

# import your tools (make sure tools.py exports the functions)
from tools import moving_average_tool, past_history_tool, generate_financial_summary_tool

st.set_page_config(page_title="Finance AI Assistant", page_icon="💹", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY as an environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Define the tools (functions metadata) for the LLM
FUNCTIONS = [
    {
        "name": "generate_financial_summary_tool",
        "description": "For a given ticker / stock, generate a table to display its quarterly key metrics.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g., AAPL"},
                "num_quarters": {"type": "integer", "description": "e.g., 5, 10"},
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "past_history_tool",
        "description": "Fetch financial data and plot closing prices.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g., AAPL"},
                "period": {"type": "string", "description": "e.g., '1mo', '3mo', '1y'"},
                "interval": {"type": "string", "description": "e.g., '1d', '1wk'"}
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "moving_average_tool",
        "description": "Fetch data and plot Close with one or more simple moving averages (SMA).",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker":   {"type": "string", "description": "Stock ticker, e.g., AAPL"},
                "period":   {"type": "string", "description": "e.g., '1mo', '3mo', '1y'"},
                "interval": {"type": "string", "description": "e.g., '1d', '1wk', '1h'"},
                "windows":  {
                    "type": "array",
                    "description": "SMA window sizes in periods (integers > 1), e.g., [5, 20, 50].",
                    "items": {"type": "integer", "minimum": 2},
                    "minItems": 1,
                    "default": [20]
                }
            },
            "required": ["ticker"]
        }
    }
]

SYSTEM_PROMPT = "You are a finance assistant. Select tools to fulfill requests."

def run_agent(user_input: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    # Map tool names to functions
    tool_registry = {
        "past_history_tool": past_history_tool,
        "moving_average_tool": moving_average_tool,
        "generate_financial_summary_tool": generate_financial_summary_tool,
    }

    # Ask the model which tool to call (if any)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=FUNCTIONS,
        function_call="auto"
    )

    message = resp.choices[0].message

    # If the model requested a function call, dispatch
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

        # Call the tool and capture result (may be dict, DataFrame, Styler, etc.)
        try:
            tool_result = tool_fn(**args)
        except TypeError as e:
            return {"text": f"Invalid arguments for {name}: {e}"}
        except Exception as e:
            return {"text": f"Error running {name}: {e}"}

        # If the tool returned a string or dict, pass it through
        return tool_result

    # No function call — just plain model text
    return {"text": (message.content or "").strip()}

# ---------------- Streamlit UI ----------------
st.title("💹 Finance AI Assistant — Tool-enabled")

if "history" not in st.session_state:
    st.session_state.history = []

col_left, col_right = st.columns([3, 1])
with col_left:
    user_input = st.chat_input("Ask me something like 'summarize A' or 'plot MSFT'...")
with col_right:
    st.write("Tools")
    st.write("- generate_financial_summary_tool")
    st.write("- past_history_tool")
    st.write("- moving_average_tool")

if user_input:
    with st.spinner("Thinking..."):
        result = run_agent(user_input)
    # store both the user prompt and the full result (could be dict or text)
    entry = {"user": user_input}
    # if result is dict, merge keys so rendering loop can pick them up
    if isinstance(result, dict):
        entry.update(result)
    else:
        entry["text"] = str(result)
    st.session_state.history.append(entry)

# Render the conversation history
for chat in st.session_state.history:
    # user message
    with st.chat_message("user"):
        st.write(chat.get("user", ""))

    # assistant message / tool outputs
    with st.chat_message("assistant"):
        # 1) Plain text reply (if any)
        if chat.get("text"):
            st.write(chat["text"])

        # 2) If tool returned pivot_display (formatted human strings)
        if "pivot_display" in chat and chat["pivot_display"] is not None:
            st.subheader("Financial Summary (formatted)")
            try:
                pivot_display = chat["pivot_display"]
                # If a Styler was returned, render as HTML
                if hasattr(pivot_display, "to_html"):
                    st.markdown(pivot_display.to_html(), unsafe_allow_html=True)
                else:
                    # ensure it's a DataFrame
                    if isinstance(pivot_display, dict):
                        # defensive: some older tools might wrap DataFrame in dict
                        st.write(pivot_display)
                    else:
                        st.dataframe(pivot_display)
            except Exception as e:
                st.error(f"Could not render pivot_display: {e}")

        # 3) Numeric pivot (raw numbers) — helpful for debugging
        if "pivot_numeric" in chat and chat["pivot_numeric"] is not None:
            st.subheader("Financial Summary (numeric)")
            try:
                st.dataframe(chat["pivot_numeric"])
            except Exception as e:
                st.write("Could not render numeric pivot:", e)

        # 4) Parsing debug info
        if "parsed_columns" in chat:
            st.subheader("Parsing debug")
            st.write("Columns parsed (converted):", chat.get("parsed_columns", []))
            st.write("Columns coerced to numeric:", chat.get("coerced_numeric_columns", []))

        # 5) Optional image or plot
        if chat.get("image_path"):
            try:
                st.image(chat["image_path"])
            except Exception as e:
                st.write("Could not display image:", e)

# Footer controls
st.markdown("---")
st.caption("This app routes user queries to tools. The financial summary tool returns both numeric and formatted tables for display.")
