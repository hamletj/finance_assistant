# app.py
import streamlit as st
import os
import json
from openai import OpenAI
from tools import finance_tool

st.set_page_config(page_title="Finance AI Assistant", page_icon="💹")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY as an environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Define the one tool for the LLM
FUNCTIONS = [
    {
        "name": "finance_tool",
        "description": "Fetch financial data, plot, and/or summarize in one call.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g., AAPL"},
                "action": {
                    "type": "string",
                    "enum": ["data", "plot", "summarize", "both"],
                    "description": "What to do: get raw data, plot chart, summarize trend, or both."
                },
                "period": {"type": "string", "description": "e.g., '1mo', '3mo', '1y'"},
                "interval": {"type": "string", "description": "e.g., '1d', '1wk'"}
            },
            "required": ["ticker", "action"]
        }
    }
]

SYSTEM_PROMPT = (
    "You are a finance assistant. Use finance_tool to fulfill requests. "
    "Choose the correct action: 'data' for just data, 'plot' for just a chart, "
    "'summarize' for text trend, 'both' for summary and plot."
)

def run_agent(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    # Step 1 — let the LLM decide
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=FUNCTIONS,
        function_call="auto"
    )

    message = resp.choices[0].message

    if message.function_call:
        args = json.loads(message.function_call.arguments)
        tool_result = finance_tool(**args)
        return tool_result
    else:
        return {"text": message.content, "image_path": None}

# --- Streamlit UI ---
st.title("💹 Finance AI Assistant — One Tool Version")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me something like 'plot AAPL' or 'summarize TSLA'...")

if user_input:
    with st.spinner("Thinking..."):
        result = run_agent(user_input)
    st.session_state.history.append({"user": user_input, **result})

# Render history
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.write(chat["user"])
    with st.chat_message("assistant"):
        if chat.get("text"):
            st.write(chat["text"])
        if chat.get("image_path"):
            st.image(chat["image_path"])