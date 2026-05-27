

import os
import pytest
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"


def chat(messages: list[dict]) -> str:
    # """Single call. Returns assistant text. Messages array = full history."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


# ── Test 1: System prompt is respected ────────────────────────────────────────
def test_system_prompt_enforced():
    # """R5: system prompt = rules contract. Model must obey format."""
    messages = [
        {"role": "system", "content": "You are a calculator. Reply with ONLY a number. No words."},
        {"role": "user",   "content": "What is 12 + 7?"},
    ]
    reply = chat(messages)
    # Strip whitespace and assert it's purely numeric
    assert reply.strip().replace(".", "").isdigit(), (
        f"System prompt violated — got: '{reply}'"
    )


# ── Test 2: Multi-turn memory lives in the array ───────────────────────────────
def test_multi_turn_memory_in_array():
    # """R8: model has no memory. History must be in messages[]. Miss a turn = amnesia."""
    messages = [
        {"role": "system",    "content": "You are a helpful assistant. Be concise."},
        {"role": "user",      "content": "My secret code word is MANGO."},
        {"role": "assistant", "content": "Got it. Your secret code word is MANGO."},
        {"role": "user",      "content": "What is my secret code word?"},
    ]
    reply = chat(messages)
    assert "MANGO" in reply.upper(), (
        f"Model forgot MANGO — history not carried. Got: '{reply}'"
    )


# ── Test 3: Amnesia when history is stripped ───────────────────────────────────
def test_amnesia_without_history():
    # """Prove R8 by omitting history. Model must NOT know the code word."""
    # First establish the codeword in a separate conversation
    # Then ask WITHOUT including that history — model should not know
    messages_no_history = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user",   "content": "What is my secret code word?"},
    ]
    reply = chat(messages_no_history)
    assert "MANGO" not in reply.upper(), (
        f"Model somehow knew MANGO without history — unexpected: '{reply}'"
    )


# ── Test 4: Role order must be user/assistant alternating ─────────────────────
def test_role_order_enforced():
    # """Some APIs reject consecutive same-role messages. 
    # llama-3.3-70b-versatile is lenient — verify it still responds."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user",   "content": "Hello"},
        {"role": "user",   "content": "Are you there?"},
    ]
    # Lenient model — should still respond, not crash
    reply = chat(messages)
    assert reply and len(reply) > 0, "Got empty reply for consecutive user messages"

# ── Test 5: Assistant reply is appended, not replaced ─────────────────────────
def test_append_not_replace():
    # """Simulate two full turns. Assert both user messages are present in final array."""
    messages = [
        {"role": "system", "content": "You are helpful. Be concise."},
        {"role": "user",   "content": "Turn 1: say the word ALPHA."},
    ]
    reply1 = chat(messages)

    # Correctly append assistant reply and next user turn
    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user",      "content": "Turn 2: say the word BETA."})

    reply2 = chat(messages)

    # Verify array has grown — both turns present
    user_turns = [m for m in messages if m["role"] == "user"]
    assert len(user_turns) == 2, f"Expected 2 user turns in array, got {len(user_turns)}"
    assert "BETA" in reply2.upper(), f"Turn 2 reply wrong: '{reply2}'"