# app.py
import streamlit as st
import os
import json
from openai import OpenAI
from tools import moving_average_tool, past_history_tool, generate_financial_summary_tool

st.set_page_config(page_title="Finance AI Assistant", page_icon="💹")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY as an environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Define the one tool for the LLM
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
    },
    # {
    #     "name": "finance_news_digest_tool",
    #     "description": "Search Yahoo Finance news for a topic, fetch top articles, and summarize with GPT.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {"type":"string","description":"Finance topic/company, e.g., 'Nvidia earnings', 'latest news about tesla'"},
    #             "top_n": {"type":"integer","default":5,"minimum":1},
    #             "max_articles_chars": {"type":"integer","default":18000,"minimum":2000}
    #         },
    #         "required": ["query"]
    #     }
    # }
]

SYSTEM_PROMPT = (
    "You are a finance assistant. Select tools to fulfill requests. "
)

def run_agent(user_input):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    # 1) Register your tools in a simple mapping
    tool_registry = {
        "past_history_tool": past_history_tool,
        "moving_average_tool": moving_average_tool,
        # "finance_news_digest_tool": finance_news_digest_tool,
        "generate_financial_summary_tool": generate_financial_summary_tool,

        # add more tools here later...
    }

    # 2) Call the model and dispatch to the selected function
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=FUNCTIONS,      # your updated FUNCTIONS list
        function_call="auto"
    )

    message = resp.choices[0].message

    if getattr(message, "function_call", None):
        name = message.function_call.name
        raw_args = message.function_call.arguments or "{}"

        # Be defensive about JSON parsing
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"text": f"Could not parse tool arguments for {name}: {raw_args}", "image_path": None}

        # Route to the correct tool
        tool_fn = tool_registry.get(name)
        if tool_fn is None:
            return {"text": f"Unknown tool requested: {name}", "image_path": None}

        try:
            tool_result = tool_fn(**args)
        except TypeError as e:
            # bad / missing args
            return {"text": f"Invalid arguments for {name}: {e}", "image_path": None}
        except Exception as e:
            # tool runtime error
            return {"text": f"Error running {name}: {e}", "image_path": None}

        return tool_result

    # 3) No tool call: just return the model’s text
    return {"text": (message.content or "").strip(), "image_path": None}

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