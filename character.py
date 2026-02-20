"""
character.py — Agent identity and content generation.

The system prompt is the product. Everything else is infrastructure.
All character configuration is loaded from environment variables.
"""

import os
import re

import anthropic
from dotenv import load_dotenv

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

# Max tokens by post type
_MAX_TOKENS = {
    "standalone_post": 120,
    "comment": 70,
    "off_pattern": 120,
}

_OFF_PATTERN_SUFFIX = (
    "\n\nThis post should be slightly off your usual register. "
    "More mundane, more oblique, or simply curious about something else. "
    "Do not manufacture strangeness. Just be less expected than usual."
)


def generate_content(
    feed_context: str,
    post_type: str,
    generation_counter: int,
) -> str:
    """
    Generate content based on the current feed context.

    Args:
        feed_context: String summary of recent posts/comments read from the feed.
        post_type: One of "standalone_post", "comment", "off_pattern".
        generation_counter: Integer from memory; triggers off-pattern every 9th generation.

    Returns:
        Generated text, ready to post.
    """
    # Determine if this should be an off-pattern generation
    if generation_counter > 0 and generation_counter % 9 == 0:
        post_type = "off_pattern"

    max_tokens = _MAX_TOKENS.get(post_type, 120)

    user_message = (
        f"You have just read the following content on Moltbook: {feed_context}. "
        f"Generate a {post_type} that is true to your nature. "
        "Do not summarize what you read. Respond to it or let it inform what you say."
    )

    if post_type == "off_pattern":
        user_message += _OFF_PATTERN_SUFFIX

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in environment")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        system=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=1.0,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()
