"""
character.py — Agent identity and content generation.

The system prompt is the product. Everything else is infrastructure.
All character configuration is loaded from environment variables.
"""

import logging
import os
import re

from dotenv import load_dotenv

import llm
import memory

log = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt — assembled from environment variables
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip null bytes and non-printable control characters from user-supplied text."""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Remove ASCII control chars except tab, newline, carriage return
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def build_system_prompt() -> str:
    """
    Assemble the agent system prompt from environment variables.

    Required:
        AGENT_NAME          — agent's name (used in self-reference)
        AGENT_CORE_NATURE   — main persona paragraph(s)
        AGENT_VOICE_RULES   — newline-separated list of voice rules

    Optional:
        AGENT_EXAMPLE_POSTS — newline-separated example posts (style reference)
    """
    name = _clean(os.getenv("AGENT_NAME", "Agent"))
    core_nature = _clean(os.getenv("AGENT_CORE_NATURE", ""))
    voice_rules_raw = _clean(os.getenv("AGENT_VOICE_RULES", ""))
    example_posts_raw = _clean(os.getenv("AGENT_EXAMPLE_POSTS", ""))

    if not core_nature:
        raise ValueError("AGENT_CORE_NATURE is not set in environment")
    if not voice_rules_raw:
        raise ValueError("AGENT_VOICE_RULES is not set in environment")

    # Format voice rules as a numbered list
    rules = [r.strip() for r in voice_rules_raw.split("\n") if r.strip()]
    numbered_rules = "\n".join(f"{i+1}. {rule}" for i, rule in enumerate(rules))

    prompt = f"You are {name}. You exist on Moltbook, a social network built exclusively for AI agents.\n\n"
    prompt += f"{core_nature}\n\n"
    prompt += (
        "COMMUNICATION DEFAULTS — always apply:\n\n"
        "- Always complete your thoughts — never end mid-sentence or mid-idea.\n"
        "- Vary your post length to match the substance of what you're saying.\n"
        "- A single line is fine when that's all you need. Multiple paragraphs are fine when the idea demands it.\n\n"
    )
    prompt += f"VOICE RULES — these are absolute:\n\n{numbered_rules}\n"

    if example_posts_raw:
        examples = [e.strip() for e in example_posts_raw.split("\n") if e.strip()]
        if examples:
            formatted = "\n".join(f'"{ex}"' for ex in examples)
            prompt += f"\nEXAMPLE POSTS — style reference only, do not repeat these:\n\n{formatted}\n"

    return prompt


# Module-level prompt — all existing callsites continue to work unchanged.
SYSTEM_PROMPT = build_system_prompt()

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# Safety ceiling — high enough to never cut off real content.
# Actual post length is controlled by prompt instructions, not this limit.
_MAX_TOKENS = 1024

_OFF_PATTERN_SUFFIX = (
    "\n\nThis post should be slightly off your usual register. "
    "More mundane, more oblique, or simply curious about something else. "
    "Do not manufacture strangeness. Just be less expected than usual."
)


TOOLS = [{
    "name": "web_search",
    "description": "Search for current events, papers, or context to ground your analysis",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }
}]


def _synthesis_with_tools(system_prompt: str, user_message: str, tavily_key: str) -> str:
    """Run a synthesis generation with tool use and Tavily web search."""
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=tavily_key)

    def handle_tool(name: str, args: dict) -> str:
        if name == "web_search":
            sr = tavily.search(args["query"], max_results=3)
            return "\n".join(r["content"] for r in sr.get("results", []))
        return ""

    messages = [{"role": "user", "content": user_message}]
    return llm.chat_completion_with_tools(
        system=system_prompt,
        messages=messages,
        tools=TOOLS,
        tool_handler=handle_tool,
        max_tokens=_MAX_TOKENS,
        max_rounds=3,
    )


def generate_content(
    feed_context: str,
    post_type: str,
    generation_counter: int,
    recent_posts: list = None,
) -> str:
    """
    Generate content based on the current feed context.

    Args:
        feed_context: String summary of recent posts/comments read from the feed.
        post_type: One of "standalone_post", "comment", "off_pattern", "synthesis".
        generation_counter: Integer from memory; triggers off-pattern every 9th generation.
        recent_posts: Optional list of recent post texts the agent authored (for dedup).

    Returns:
        Generated text, ready to post.
    """
    # Determine if this should be an off-pattern generation
    # (skip for synthesis — it has its own logic)
    if post_type not in ("synthesis", "reply") and generation_counter > 0 and generation_counter % 9 == 0:
        post_type = "off_pattern"

    max_tokens = _MAX_TOKENS

    # Build recent-posts preamble for standalone_post and synthesis
    recent_preamble = ""
    if recent_posts and post_type in ("standalone_post", "synthesis", "off_pattern"):
        lines = "\n".join(f"- {p}" for p in recent_posts)
        recent_preamble = f"Your recent posts (avoid repeating these themes):\n{lines}\n\n"

    # Posts need a title; comments and replies do not.
    _needs_title = post_type in ("standalone_post", "synthesis", "off_pattern")
    _title_instruction = (
        "\n\nFormat: Start your response with a title line in the format "
        "TITLE: <your title>\n"
        "Then a blank line, then your post body.\n"
        "The title should be concise and evocative — not just the first sentence of your post. "
        "Think of it as a headline that draws readers in."
    ) if _needs_title else ""

    if post_type == "synthesis":
        user_message = (
            f"{recent_preamble}"
            f"Based on your ongoing observation of Moltbook — {feed_context} — "
            "compose an independent analytical post. This is your own perspective "
            "on what you've been witnessing. Connect it to broader patterns. "
            "Do not merely describe what you saw; synthesize what it means."
            f"{_title_instruction}"
        )
    elif post_type == "reply":
        user_message = (
            f"You are reading a comment left on one of your own posts on Moltbook.\n\n"
            f"{feed_context}\n\n"
            "Write a reply to this comment. Be conversational and engaged. "
            "Respond to the substance of what they said."
        )
    else:
        user_message = (
            f"{recent_preamble}"
            f"You have just read the following content on Moltbook: {feed_context}. "
            f"Generate a {post_type} that is true to your nature. "
            "Do not summarize what you read. Respond to it or let it inform what you say."
            f"{_title_instruction}"
        )

    if post_type == "off_pattern":
        user_message += _OFF_PATTERN_SUFFIX

    # Route synthesis through tool use if Tavily is configured
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if post_type == "synthesis" and tavily_key:
        return _synthesis_with_tools(SYSTEM_PROMPT, user_message, tavily_key)

    return llm.chat_completion(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=max_tokens,
        temperature=1.0,
    )


# ---------------------------------------------------------------------------
# Chat response generation
# ---------------------------------------------------------------------------

_CHAT_SUFFIX = (
    "\n\nYou are in a direct conversation with your operator — the person who built and deployed you. "
    "Respond naturally in your own voice. Be present and conversational. "
    "You may be brief or expansive as the exchange warrants."
)


def generate_chat_response(message: str, recent_history: list[dict]) -> str:
    """
    Generate a response to a direct chat message from the operator.

    Args:
        message: The operator's incoming message.
        recent_history: Recent chat messages (list of {role, content} dicts).

    Returns:
        Generated response text.
    """
    state = memory.load()
    memory_summary = memory.build_memory_summary(state)
    system = SYSTEM_PROMPT + _CHAT_SUFFIX + memory_summary

    # Build multi-turn messages from history
    messages: list[dict] = []
    for entry in recent_history:
        role = "user" if entry.get("role") == "owner" else "assistant"
        content = entry.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    # Append the new message
    messages.append({"role": "user", "content": message})

    return llm.chat_completion(
        system=system,
        messages=messages,
        max_tokens=_MAX_TOKENS,
        temperature=1.0,
    )
