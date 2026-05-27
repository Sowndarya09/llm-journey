# """
# Level 3 — test_tool_use.py
# Tests: tool_use block, tool_result handshake, tool_use_id matching
# API: Groq (llama-3.3-70b-versatile supports tool use)

# THE HANDSHAKE (burn this in):
#   Turn 1 → you send messages + tools definition
#   Turn 2 → model replies with role:assistant, content: tool_use block
#   Turn 3 → you execute the tool, append role:user content: tool_result block (with tool_use_id)
#   Turn 4 → model replies with final answer using the result
# """

import os
import json
import pytest
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

# ── Fake tool implementations ──────────────────────────────────────────────────

def fake_get_weather(city: str) -> dict:
    # """Simulates a weather API. Returns hardcoded data."""
    return {"city": city, "temp_c": 22, "condition": "sunny"}


def fake_get_stock(symbol: str) -> dict:
    # """Simulates a stock price API."""
    prices = {"AAPL": 189.50, "GOOG": 141.20, "MSFT": 415.00}
    price = prices.get(symbol.upper(), 0.0)
    return {"symbol": symbol.upper(), "price": price}


# ── Tool definitions (sent to API) ────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Get current stock price by ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker e.g. AAPL"}
                },
                "required": ["symbol"],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    # """Route tool call to the right fake function. Return JSON string."""
    if name == "get_weather":
        result = fake_get_weather(**args)
    elif name == "get_stock":
        result = fake_get_stock(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)


def run_tool_handshake(user_prompt: str) -> tuple[str, str, str]:
    # """
    # Full two-turn tool use handshake.
    # Returns (tool_name_called, tool_use_id, final_answer)
    # """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user",   "content": user_prompt},
    ]

    # Turn 1: model requests tool use
    response1 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
        max_tokens=256,
    )

    assistant_msg = response1.choices[0].message
    assert assistant_msg.tool_calls, "Model did not request a tool call"

    tool_call = assistant_msg.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    tool_use_id = tool_call.id  # ← this must be echoed back

    # Execute the tool locally
    tool_output = execute_tool(tool_name, tool_args)

    # Turn 2: send result back — tool_use_id must match
    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
    messages.append({
        "role": "tool",
        "tool_call_id": tool_use_id,   # ← the critical ID
        "content": tool_output,
    })

    response2 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
    )

    final_answer = response2.choices[0].message.content.strip()
    return tool_name, tool_use_id, final_answer


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_weather_tool_called():
    # """Model must call get_weather when asked about weather."""
    tool_name, tool_id, answer = run_tool_handshake("What's the weather in Vancouver?")
    assert tool_name == "get_weather", f"Wrong tool called: {tool_name}"
    assert tool_id, "tool_use_id was empty"
    assert "vancouver" in answer.lower() or "sunny" in answer.lower(), (
        f"Answer doesn't reference result: '{answer}'"
    )


def test_stock_tool_called():
    # """Model must call get_stock when asked about a stock price."""
    tool_name, tool_id, answer = run_tool_handshake("What is the current price of AAPL?")
    assert tool_name == "get_stock", f"Wrong tool called: {tool_name}"
    # Answer must include the fake price we returned
    assert "189" in answer or "189.50" in answer, (
        f"Answer doesn't include the stock price: '{answer}'"
    )


def test_tool_use_id_present():
    # """tool_use_id must be non-empty — API rejects mismatched or missing IDs."""
    _, tool_id, _ = run_tool_handshake("What's the weather in Toronto?")
    assert tool_id and len(tool_id) > 4, f"tool_use_id looks invalid: '{tool_id}'"


def test_no_tool_for_general_question():
    # """Model should NOT call a tool for a general knowledge question."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools only when needed."},
        {"role": "user",   "content": "What is the capital of France?"},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
        max_tokens=128,
    )
    msg = response.choices[0].message
    assert not msg.tool_calls, (
        f"Model incorrectly called a tool for a general question"
    )
    assert "paris" in msg.content.lower(), f"Wrong answer: '{msg.content}'"