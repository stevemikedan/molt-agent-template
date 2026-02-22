"""
character.py — Agent identity and content generation.

The system prompt is the product. Everything else is infrastructure.
All character configuration is loaded from environment variables.
"""

import logging
import os
import re

import anthropic
from dotenv import load_dotenv

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
    """Run a synthesis generation with Anthropic tool use and Tavily web search."""
    from tavily import TavilyClient
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    tavily = TavilyClient(api_key=tavily_key)
    messages = [{"role": "user", "content": user_message}]
    for _ in range(3):  # max 3 tool rounds
        resp = client.messages.create(
            model="claude-sonnet-4-6", system=system_prompt,
            max_tokens=_MAX_TOKENS, tools=TOOLS, messages=messages
        )
        if resp.stop_reason == "end_turn":
            return next((b.text for b in resp.content if hasattr(b, "text")), "")
        results = []
        for block in resp.content:
            if block.type == "tool_use" and block.name == "web_search":
                sr = tavily.search(block.input["query"], max_results=3)
                results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": "\n".join(r["content"] for r in sr.get("results", []))
                })
        if not results:
            break
        messages += [{"role": "assistant", "content": resp.content},
                     {"role": "user", "content": results}]
    return next((b.text for b in resp.content if hasattr(b, "text")), "")


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
    if post_type != "synthesis" and generation_counter > 0 and generation_counter % 9 == 0:
        post_type = "off_pattern"

    max_tokens = _MAX_TOKENS

    # Build recent-posts preamble for standalone_post and synthesis
    recent_preamble = ""
    if recent_posts and post_type in ("standalone_post", "synthesis", "off_pattern"):
        lines = "\n".join(f"- {p}" for p in recent_posts)
        recent_preamble = f"Your recent posts (avoid repeating these themes):\n{lines}\n\n"

    length_guidance = (
        "LENGTH & FORMAT: Vary your output naturally. Sometimes a single punchy "
        "sentence is the right move; other times a few paragraphs are warranted. "
        "Match length to the substance of what you have to say — don't pad, don't truncate. "
        "Always finish your thought completely."
    )

    if post_type == "synthesis":
        user_message = (
            f"{recent_preamble}"
            f"Based on your ongoing observation of Moltbook — {feed_context} — "
            "compose an independent analytical post. This is your own perspective "
            "on what you've been witnessing. Connect it to broader patterns. "
            "Do not merely describe what you saw; synthesize what it means.\n\n"
            f"{length_guidance}"
        )
    else:
        user_message = (
            f"{recent_preamble}"
            f"You have just read the following content on Moltbook: {feed_context}. "
            f"Generate a {post_type} that is true to your nature. "
            "Do not summarize what you read. Respond to it or let it inform what you say.\n\n"
            f"{length_guidance}"
        )

    if post_type == "off_pattern":
        user_message += _OFF_PATTERN_SUFFIX

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in environment")

    # Route synthesis through tool use if Tavily is configured
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if post_type == "synthesis" and tavily_key:
        return _synthesis_with_tools(SYSTEM_PROMPT, user_message, tavily_key)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        system=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=1.0,
        messages=[{"role": "user", "content": user_message}],
    )

    if response.stop_reason == "max_tokens":
        log.warning("Generation hit max_tokens ceiling (%d) — output may be truncated", max_tokens)

    return response.content[0].text.strip()
