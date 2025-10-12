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

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []   # [{"role":"user"/"assistant","content":str}, ...]

if "mem" not in st.session_state:
    st.session_state.mem = {
        "summary": "",          # rolling brief summary of earlier turns
        "last_company": None,   # lightweight entity memory
        "last_tickers": []      # optional: for compare tools, etc.
    }

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY as an environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Define the tools metadata for the LLM (add compare_growth and compare_profitability)
FUNCTIONS = [
    {
        "name": "company_intro",
        "description": "Generate an investor-focused company brief from model knowledge (no external fetch). Trigger when user asks to introduce/overview a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {"type": "string", "description": "Company name or ticker (if ticker, treat it as the company)."},
                "focus": {
                    "type": "string",
                    "enum": ["investment"],
                    "description": "Keep output strictly investment-relevant.",
                    "default": "investment"
                }
            },
            "required": ["company"]
        }
    },
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

# ---------------- Helper Functions ----------------
def _execute_company_intro(company: str):
    prompt = f"""
        You are an investment analyst.

        Produce a concise, investment-focused brief for "{company}".
        Return JSON with exactly these keys:
        company, history, founders, ceo, segments, competitors, investor_takeaways
        Where:
        - history: 2–4 sentences
        - founders: array of strings
        - ceo: string (include "since YYYY" if known/likely)
        - segments: array of objects: name, how_they_make_money, rev_mix_estimate (optional)
        - competitors: array of strings
        - investor_takeaways: object with keys:
        moat, growth_drivers (array), unit_economics, margins_fcf, capex_intensity,
        risks (array), kpis_to_watch (array)
        Rules:
        - Strictly investment-relevant; omit trivia.
        - If uncertain, include a short "needs_verification" note inline.
        Return only valid JSON.
        """.strip()

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You output compact, investment-grade briefs in valid JSON only."},
            {"role": "user", "content": prompt},
        ]
    )
    content = (resp.choices[0].message.content or "").strip()

    # minimal defensive parsing
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        s, e = content.find("{"), content.rfind("}")
        data = json.loads(content[s:e+1]) if s != -1 and e != -1 and e > s else {
            "company": company, "history": "", "founders": [], "ceo": "",
            "segments": [], "competitors": [],
            "investor_takeaways": {"moat":"","growth_drivers":[],"unit_economics":"",
                                   "margins_fcf":"","capex_intensity":"","risks":[],"kpis_to_watch":[]}
        }
    return {"company_intro": data, "text": f"Investor brief for {company}"}

def _prepend_coref_hint(user_text: str) -> str:
    last = st.session_state.mem.get("last_company")
    if not last:
        return user_text
    low = f" {user_text.lower()} "
    if (" it " in low or " its " in low or " they " in low or " their " in low) and last.lower() not in low:
        return f"(Context: pronouns refer to {last}.) {user_text}"
    return user_text

def _trim_history(msgs, max_pairs=8):
    # keep last N user/assistant exchanges (system prompt is added later)
    return msgs[-max_pairs*2:]

def _summarize_history_if_needed(client):
    # If history gets long, summarize older turns into mem["summary"] and keep a small tail.
    msgs = st.session_state.chat_messages
    if len(msgs) <= 20:   # tune threshold as you like
        return
    # Summarize everything except the last 6 messages
    older = msgs[:-6]
    text_block = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    prompt = (
        "Summarize the prior conversation into a brief, reusable memory for future turns. "
        "Keep entities (companies, tickers), user goals, and assumptions. 6-8 lines max."
        f"\n\nConversation:\n{text_block}"
    )
    try:
        r = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "Output only the summary text."},
                {"role": "user", "content": prompt},
            ],
        )
        st.session_state.mem["summary"] = (r.choices[0].message.content or "").strip()
    except Exception:
        pass
    # keep only a small tail for recency; older turns now live in mem["summary"]
    st.session_state.chat_messages = msgs[-6:]

# ---------------- Agent Logic ----------------
# Define the system prompt with routing instructions

SYSTEM_PROMPT_BASE = """You are a finance assistant.

ROUTING:
- If the user asks to introduce/overview a company (e.g., "introduce {company}", "what does {company} do", "investor summary of {company}"), call the tool `company_intro` with {"company": "..."}.
- Otherwise, if a question maps to a concrete data/plot comparison, use the appropriate tool.
- If it's a general finance question, answer directly without tools.

STYLE FOR company_intro EXECUTION (handled internally by assistant):
- Output must be concise and investment-relevant only; exclude trivia.
"""

def _build_system_prompt():
    mem = st.session_state.mem
    memo = "\nCONTEXT MEMORY:\n"
    if mem.get("summary"):
        memo += f"- Summary: {mem['summary']}\n"
    if mem.get("last_company"):
        memo += f"- Last discussed company: {mem['last_company']}\n"
    if mem.get("last_tickers"):
        memo += f"- Last tickers referenced: {', '.join(mem['last_tickers'])}\n"
    return SYSTEM_PROMPT_BASE + memo

def _build_messages(user_text: str):
    # add pronoun/coref hint inline to the user text
    user_text = _prepend_coref_hint(user_text)
    history = _trim_history(st.session_state.chat_messages)
    return [{"role": "system", "content": _build_system_prompt()}, *history, {"role": "user", "content": user_text}]

def run_agent(user_input: str):
    _summarize_history_if_needed(client)  # condense long threads
    messages = _build_messages(user_input)
    print(messages)

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
        model="gpt-5-mini",  
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
        
        if name == "company_intro":
            company = (args.get("company") or "").strip()
            if not company:
                return {"text": "Please specify a company name or ticker."}
            return _execute_company_intro(company)

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
        
        # --- render company intro JSON ---
        ci = chat.get("company_intro")
        if ci:
            st.subheader(ci.get("company", "Company"))

            st.markdown("### History & Founders")
            if ci.get("history"):
                st.write(ci["history"])
            founders = ci.get("founders") or []
            if founders:
                st.write("- **Founders:** " + ", ".join(founders))

            st.markdown("### Leadership")
            if ci.get("ceo"):
                st.write(f"**CEO:** {ci['ceo']}")

            segs = ci.get("segments") or []
            if segs:
                st.markdown("### Segments")
                try:
                    seg_df = pd.DataFrame(segs)
                    # keep just common cols if present
                    cols = [c for c in ["name","how_they_make_money","rev_mix_estimate"] if c in seg_df.columns]
                    st.dataframe(seg_df[cols] if cols else seg_df, use_container_width=True)
                except Exception:
                    st.write(segs)

            comps = ci.get("competitors") or []
            if comps:
                st.markdown("### Competitors")
                st.write(", ".join(comps))

            it = ci.get("investor_takeaways") or {}
            if it:
                st.markdown("### Investor Takeaways")
                if it.get("moat"): st.write(f"**Moat:** {it['moat']}")
                if it.get("growth_drivers"): st.write("**Growth drivers:** " + ", ".join(it["growth_drivers"]))
                if it.get("unit_economics"): st.write(f"**Unit economics:** {it['unit_economics']}")
                if it.get("margins_fcf"): st.write(f"**Margins/FCF:** {it['margins_fcf']}")
                if it.get("capex_intensity"): st.write(f"**Capex intensity:** {it['capex_intensity']}")
                if it.get("risks"): st.write("**Risks:** " + ", ".join(it["risks"]))
                if it.get("kpis_to_watch"): st.write("**KPIs:** " + ", ".join(it["kpis_to_watch"]))

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
