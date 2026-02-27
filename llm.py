"""
llm.py — Centralised LLM interface for multi-provider support.

Reads LLM_PROVIDER + LLM_API_KEY from env.  Falls back to ANTHROPIC_API_KEY
when the provider is (or defaults to) "anthropic".

Non-Anthropic providers all use OpenAI-compatible chat completions endpoints.
"""

import logging
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry — non-Anthropic, all OpenAI-compatible
# ---------------------------------------------------------------------------

_OPENAI_PROVIDERS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "model": "grok-3-mini-fast",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
}

# ---------------------------------------------------------------------------
# Resolve provider + key from environment
# ---------------------------------------------------------------------------

def _get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").lower().strip()


def _get_api_key() -> str:
    """Return the API key for the active provider."""
    key = os.getenv("LLM_API_KEY", "").strip()
    if key:
        return key
    # Backwards compat — fall back to ANTHROPIC_API_KEY when provider is anthropic
    if _get_provider() == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if key:
            return key
    raise ValueError(
        "No API key configured. Set LLM_API_KEY (or ANTHROPIC_API_KEY for Anthropic)."
    )

# ---------------------------------------------------------------------------
# OpenAI-compatible helpers
# ---------------------------------------------------------------------------

def _openai_chat(
    provider: str,
    api_key: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 1.0,
) -> str:
    cfg = _OPENAI_PROVIDERS[provider]
    url = f"{cfg['base_url']}/chat/completions"
    resp = httpx.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": cfg["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _openai_chat_with_tools(
    provider: str,
    api_key: str,
    messages: list[dict],
    tools: list[dict],
    tool_handler,
    max_tokens: int = 1024,
    max_rounds: int = 3,
) -> str:
    """OpenAI function-calling loop."""
    cfg = _OPENAI_PROVIDERS[provider]
    url = f"{cfg['base_url']}/chat/completions"

    # Convert Anthropic-style tool defs to OpenAI function format
    oai_tools = []
    for t in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        })

    for _ in range(max_rounds):
        resp = httpx.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": cfg["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "tools": oai_tools,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        # If no tool calls, return the content
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return (msg.get("content") or "").strip()

        # Append assistant message with tool calls
        messages.append(msg)

        # Execute each tool call and append results
        for tc in tool_calls:
            import json
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            result = tool_handler(fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    # Exhausted rounds — return whatever content we have
    return (msg.get("content") or "").strip()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat_completion(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 1.0,
) -> str:
    """Generate a chat completion. Works with any configured provider."""
    provider = _get_provider()
    api_key = _get_api_key()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if response.stop_reason == "max_tokens":
            log.warning("Generation hit max_tokens ceiling (%d)", max_tokens)
        return response.content[0].text.strip()

    # OpenAI-compatible provider
    if provider not in _OPENAI_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    oai_messages = [{"role": "system", "content": system}] + messages
    return _openai_chat(provider, api_key, oai_messages, max_tokens, temperature)


def chat_completion_cheap(
    messages: list[dict],
    max_tokens: int = 30,
) -> str:
    """Cheap/fast completion — used for verification challenge math."""
    provider = _get_provider()
    api_key = _get_api_key()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.content[0].text.strip()

    # Non-Anthropic providers use their default model (already cheap)
    if provider not in _OPENAI_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return _openai_chat(provider, api_key, messages, max_tokens, temperature=0.0)


def chat_completion_with_tools(
    system: str,
    messages: list[dict],
    tools: list[dict],
    tool_handler,
    max_tokens: int = 1024,
    max_rounds: int = 3,
) -> str:
    """Chat completion with tool use. tool_handler(name, args) -> str."""
    provider = _get_provider()
    api_key = _get_api_key()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msgs = list(messages)
        resp = None
        for _ in range(max_rounds):
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                system=system,
                max_tokens=max_tokens,
                tools=tools,
                messages=msgs,
            )
            if resp.stop_reason == "end_turn":
                return next((b.text for b in resp.content if hasattr(b, "text")), "")
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result = tool_handler(block.name, block.input)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            if not results:
                break
            msgs += [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": results},
            ]
        return next((b.text for b in resp.content if hasattr(b, "text")), "") if resp else ""

    # OpenAI-compatible provider
    if provider not in _OPENAI_PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider}")
    oai_messages = [{"role": "system", "content": system}] + list(messages)
    return _openai_chat_with_tools(
        provider, api_key, oai_messages, tools, tool_handler, max_tokens, max_rounds
    )
