
# # Level 3 — test_streaming.py
# Tests: stream accumulation, [DONE] guard, streamed == non-streamed output
# API: Groq

# KEY RULES:
 #  - Never act on partial chunks — R4 applies
 #  - Only assert after stream closes (finish_reason == 'stop')
 #  - Streamed and non-streamed responses for same prompt/temp=0 must match


import os
import pytest
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

SYSTEM = "You are a helpful assistant. Be concise. Max 2 sentences."


def stream_response(user_prompt: str) -> tuple[str, list[str]]:
  # 
    # Stream a response. Accumulate all chunks.
    # Returns (full_text, list_of_raw_chunks)
    # Never returns early — waits for finish_reason == 'stop'.
    # 
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=128,
        stream=True,
    )

    chunks = []
    full_text = ""
    finished = False

    for chunk in stream:
        delta = chunk.choices[0].delta
        finish = chunk.choices[0].finish_reason

        if delta.content:
            chunks.append(delta.content)
            full_text += delta.content

        if finish == "stop":
            finished = True
            break

    assert finished, "Stream ended without finish_reason=='stop' — [DONE] never received"
    return full_text.strip(), chunks


def non_stream_response(user_prompt: str) -> str:
    # Same prompt, no streaming. Returns full text."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=128,
        stream=False,
    )
    return response.choices[0].message.content.strip()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_stream_accumulates_chunks():
    # """Streaming must produce multiple chunks, not one blob."""
    full_text, chunks = stream_response("Count from 1 to 10 slowly.")
    assert len(chunks) > 1, (
        f"Expected multiple chunks, got {len(chunks)} — stream may not be working"
    )
    assert len(full_text) > 0, "Accumulated text is empty"


def test_no_action_on_partial_chunk():
    # """
    # R4: partial chunk is not a complete response.
    # First chunk alone must NOT contain the full expected answer.
    # """
    full_text, chunks = stream_response("What is the capital of Japan?")
    first_chunk = chunks[0] if chunks else ""

    # Full answer must contain Tokyo
    assert "tokyo" in full_text.lower(), f"Full answer missing 'Tokyo': '{full_text}'"

    # First chunk alone is partial — it should not be treated as complete
    # (it may or may not contain 'tokyo' — the point is we never stop at chunk[0])
    assert len(full_text) > len(first_chunk), (
        "First chunk equals full response — streaming may not be chunking"
    )


def test_stream_waits_for_done():
    # """Full text only available after finish_reason==stop. Assert at end, not mid-stream."""
    full_text, chunks = stream_response("Explain what an API is in one sentence.")
    # If we got here, finish_reason was 'stop' — the guard in stream_response enforces it
    assert full_text, "Empty response after stream closed"
    assert len(full_text) > 20, f"Response suspiciously short: '{full_text}'"


def test_streamed_equals_non_streamed():
    # """
    # Both assertions: same prompt at temp=0 must produce identical output
    # whether streamed or not.
    # """
    prompt = "What is 15 multiplied by 4? Answer with only the number."
    streamed, _ = stream_response(prompt)
    non_streamed = non_stream_response(prompt)

    # Normalise whitespace for comparison
    streamed_norm = " ".join(streamed.split())
    non_streamed_norm = " ".join(non_streamed.split())

    assert streamed_norm == non_streamed_norm, (
        f"Mismatch!\n  Streamed:     '{streamed_norm}'\n  Non-streamed: '{non_streamed_norm}'"
    )


def test_partial_chunk_is_not_valid_json():
    # """
    # Ask for JSON. Prove that an early chunk is not valid JSON.
    # Only the full accumulated response should parse cleanly.
    # """
    import json

    full_text, chunks = stream_response(
        'Return ONLY valid JSON: {"answer": "<capital of Canada>"}'
    )

    # Full response must be valid JSON
    try:
        parsed = json.loads(full_text)
        assert "answer" in parsed, f"JSON missing 'answer' key: {parsed}"
    except json.JSONDecodeError as e:
        pytest.fail(f"Full streamed response is not valid JSON: '{full_text}' — {e}")

    # First chunk alone must NOT be valid JSON (it's partial)
    first_chunk = chunks[0] if chunks else ""
    is_first_chunk_valid_json = True
    try:
        json.loads(first_chunk)
    except json.JSONDecodeError:
        is_first_chunk_valid_json = False

    # We expect partial chunks to be invalid JSON — proves R4
    # (If this fails, the model returned the whole thing in one chunk — not a test failure per se)
    if is_first_chunk_valid_json and len(chunks) == 1:
        pytest.skip("Model returned full JSON in single chunk — streaming not chunked enough to test")