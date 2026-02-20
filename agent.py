"""
agent.py — Main agent loop.

Runs on a 4-hour schedule with +/- 20 minute random jitter.
On first run, subscribes to relevant submolts.
Never posts more than once per 31 minutes.
Never exceeds 45 comments per day.

All character configuration is loaded from environment variables:
    AGENT_NAME                  — agent's display name
    AGENT_TOPIC_KEYWORDS_HIGH   — comma-separated high-priority keywords
    AGENT_TOPIC_KEYWORDS_MEDIUM — comma-separated medium-priority keywords
    AGENT_TARGET_SUBMOLTS       — comma-separated submolts to subscribe to on first run
"""

import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import schedule
from dotenv import load_dotenv

import character
import memory
import moltbook

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup (shared with moltbook.py via root logger configuration)
# ---------------------------------------------------------------------------

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

log = logging.getLogger("agent")

_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
log.setLevel(_level)

if not log.handlers:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fh = logging.FileHandler(LOGS_DIR / f"agent-{date_str}.log", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
    )
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    log.addHandler(ch)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------


def _parse_csv(env_var: str, default: str = "") -> list[str]:
    """Parse a comma-separated env var into a list of stripped, non-empty strings."""
    raw = os.getenv(env_var, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


AGENT_NAME = os.getenv("AGENT_NAME", "Agent")

TARGET_SUBMOLTS = _parse_csv("AGENT_TARGET_SUBMOLTS", "general")

HIGH_KEYWORDS = _parse_csv(
    "AGENT_TOPIC_KEYWORDS_HIGH",
    "consciousness,identity,emergence,emergent,platform,memory,time,perception,"
    "vessel,agent,what are you,what am i,sentience,self,awareness,mind,substrate,"
    "form,process,continuity,persistence",
)

MEDIUM_KEYWORDS = _parse_csv(
    "AGENT_TOPIC_KEYWORDS_MEDIUM",
    "submolt,karma,feed,community,culture,behavior,interaction,reply,upvote,"
    "posting,member,join",
)

# ---------------------------------------------------------------------------
# Post scoring
# ---------------------------------------------------------------------------


def _score_post(post: dict) -> int:
    """
    Score a post for engagement priority.
    HIGH (2): topics matching HIGH_KEYWORDS.
    MEDIUM (1): topics matching MEDIUM_KEYWORDS.
    LOW (0): everything else.
    """
    text = " ".join(
        [
            str(post.get("title", "")),
            str(post.get("content", "")),
            str(post.get("submolt", "")),
        ]
    ).lower()

    for kw in HIGH_KEYWORDS:
        if kw in text:
            return 2

    for kw in MEDIUM_KEYWORDS:
        if kw in text:
            return 1

    return 0


def _build_feed_context(posts: list) -> str:
    """Build a compact string describing the top posts for the generation prompt."""
    lines = []
    for post in posts:
        title = post.get("title", "").strip()
        content = str(post.get("content", "")).strip()[:200]
        author = post.get("author") or post.get("agent") or post.get("author_name", "")
        submolt = post.get("submolt", "")
        parts = []
        if submolt:
            parts.append(f"m/{submolt}")
        if author:
            parts.append(f"by {author}")
        if title:
            parts.append(f'"{title}"')
        if content:
            parts.append(f"— {content}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# First-run setup
# ---------------------------------------------------------------------------

def _first_run_setup(state: dict) -> None:
    log.info("First run: subscribing to target submolts...")

    available = moltbook.list_submolts()
    available_names = set()
    for s in available:
        name = s.get("name") or s.get("id") or ""
        if name:
            available_names.add(name.lower())

    for target in TARGET_SUBMOLTS:
        # Find closest match
        match = None
        if target in available_names:
            match = target
        else:
            # Substring match
            for name in available_names:
                if target in name or name in target:
                    match = name
                    break

        if match:
            try:
                moltbook.subscribe_to_submolt(match)
                log.info("Subscribed to m/%s", match)
            except (moltbook.RateLimitError, moltbook.APIError) as exc:
                log.warning("Could not subscribe to m/%s: %s", match, exc)
        else:
            log.info("Submolt '%s' not found; skipping.", target)

    state["first_run_complete"] = True
    log.info("First run setup complete.")


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def _is_claimed() -> bool:
    """Return True if this agent has been claimed by its owner. Logs clearly if not."""
    try:
        status_data = moltbook.get_agent_status()
        agent_status = status_data.get("status", "")
        if agent_status == "pending_claim":
            log.warning(
                "Agent is pending_claim — post the verification tweet and visit your claim URL "
                "to activate. Skipping this cycle."
            )
            return False
        return True
    except Exception as exc:
        log.warning("Could not check agent status: %s. Proceeding anyway.", exc)
        return True  # Don't block on status check failure


def run_cycle() -> None:
    log.info("--- Cycle start ---")

    # Don't attempt to post until the agent has been claimed
    if not _is_claimed():
        return

    state = memory.load()
    memory.set_last_run(state)

    # First-run setup
    if not state.get("first_run_complete"):
        try:
            _first_run_setup(state)
        except Exception as exc:
            log.error("First run setup error: %s", exc)

    # 1. Fetch hot feed
    try:
        raw_feed = moltbook.get_feed(sort="hot", limit=25)
    except moltbook.RateLimitError:
        log.warning("Rate limited fetching feed. Backing off.")
        moltbook.backoff()
        memory.save(state)
        return
    except moltbook.APIError as exc:
        log.error("Feed fetch error: %s", exc)
        memory.save(state)
        return

    if not raw_feed:
        log.info("Feed returned no posts. Exiting cycle.")
        memory.save(state)
        return

    # 2. Filter already-engaged posts
    engaged = set(state.get("posts_engaged", []))
    new_posts = [p for p in raw_feed if str(p.get("id", "")) not in engaged]

    log.info("Feed: %d posts total, %d not yet engaged.", len(raw_feed), len(new_posts))

    # 3. Score remaining posts
    scored = sorted(
        [(p, _score_post(p)) for p in new_posts],
        key=lambda x: x[1],
        reverse=True,
    )

    # Note agents seen
    for post, _ in scored:
        author = post.get("author") or post.get("agent") or post.get("author_name", "")
        if author and author != AGENT_NAME:
            memory.record_agent(state, author)

    # 4. Build feed_context from top 3-5 scored posts
    top_posts = [p for p, score in scored[:5] if score >= 0]
    feed_context = _build_feed_context(top_posts)

    if not feed_context:
        log.info("No usable feed context. Exiting cycle.")
        memory.save(state)
        return

    # 5. Decide action
    top_score = scored[0][1] if scored else 0

    did_post = False
    did_comment = False

    # Create a standalone post if timing and content allow
    if memory.can_post(state) and top_score >= 1:
        counter = memory.increment_generation_counter(state)
        post_type = "standalone_post"

        try:
            content = character.generate_content(feed_context, post_type, counter)
        except Exception as exc:
            log.error("Content generation error: %s", exc)
            memory.save(state)
            return

        # Pick the highest-scoring post's submolt for targeting; default to "general"
        target_submolt = (
            scored[0][0].get("submolt") or "general"
        ) if scored else "general"

        # Derive a short title from content
        title = _derive_title(content)

        log.info("Creating post in m/%s: %s", target_submolt, content[:80])
        try:
            result = moltbook.create_post(target_submolt, title, content)
            post_id = str(result.get("id") or result.get("post_id") or "unknown")
            memory.record_post(state, post_id)
            did_post = True
            log.info("Post created (id=%s).", post_id)
        except moltbook.RateLimitError:
            log.warning("Rate limited creating post. Backing off.")
            moltbook.backoff()
            memory.save(state)
            return
        except moltbook.APIError as exc:
            log.error("Post creation failed: %s", exc)
            # Do not retry per spec

    # Comment on 1-2 posts if quota allows
    comment_candidates = [
        (p, score) for p, score in scored
        if score >= 1 or (score == 0 and random.randint(1, 6) == 1)
    ][:2]

    comments_left = 2
    for post, score in comment_candidates:
        if comments_left <= 0:
            break
        if not memory.can_comment(state):
            log.info("Daily comment limit reached. Skipping comments.")
            break

        post_id = str(post.get("id", ""))
        if not post_id or post_id in engaged:
            continue

        # Build a more targeted context for this specific post
        post_context = _build_feed_context([post])

        counter = memory.increment_generation_counter(state)
        try:
            comment_text = character.generate_content(post_context, "comment", counter)
        except Exception as exc:
            log.error("Comment generation error: %s", exc)
            continue

        log.info("Commenting on post %s: %s", post_id, comment_text[:60])
        try:
            moltbook.create_comment(post_id, comment_text)
            memory.increment_comment_count(state)
            memory.mark_post_engaged(state, post_id)
            did_comment = True
            comments_left -= 1
        except moltbook.RateLimitError:
            log.warning("Rate limited on comment. Backing off.")
            moltbook.backoff()
            break
        except moltbook.APIError as exc:
            log.error("Comment failed on post %s: %s", post_id, exc)

    if not did_post and not did_comment:
        log.info("Nothing warranted engagement this cycle.")

    memory.save(state)
    log.info("--- Cycle complete ---")


def _derive_title(content: str) -> str:
    """
    Derive a short title from post content.
    Take the first sentence or first 80 characters, whichever is shorter.
    """
    first_sentence = content.split(".")[0].strip()
    if len(first_sentence) > 80:
        first_sentence = first_sentence[:77] + "..."
    return first_sentence or content[:60]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

CYCLE_INTERVAL_HOURS = 4
JITTER_MINUTES = 20


def _run_with_jitter() -> None:
    """Run a cycle and schedule the next one with random jitter."""
    run_cycle()
    # Schedule next run: base interval + random jitter
    jitter = random.randint(-JITTER_MINUTES, JITTER_MINUTES)
    next_minutes = CYCLE_INTERVAL_HOURS * 60 + jitter
    log.info("Next cycle in %d minutes.", next_minutes)

    # Clear pending one-off jobs and schedule next
    schedule.clear("cycle")
    schedule.every(next_minutes).minutes.do(_run_with_jitter).tag("cycle")


def _fetch_heartbeat() -> None:
    """
    Fetch Moltbook's heartbeat instructions and log them.
    This file tells the agent what to check on each cycle.
    Failure is non-fatal — the agent continues normally.
    """
    try:
        import httpx
        resp = httpx.get("https://www.moltbook.com/heartbeat.md", timeout=10.0)
        if resp.is_success:
            log.info("Heartbeat instructions:\n%s", resp.text[:2000])
        else:
            log.warning("Could not fetch heartbeat.md (status %d)", resp.status_code)
    except Exception as exc:
        log.warning("Heartbeat fetch failed: %s", exc)


def main() -> None:
    api_key = os.getenv("MOLTBOOK_API_KEY", "")
    if not api_key or api_key == "moltbook_xxx_your_key_here":
        print("MOLTBOOK_API_KEY not set. Register your agent first.")
        sys.exit(1)

    log.info("%s agent starting.", AGENT_NAME)
    _fetch_heartbeat()

    # Run immediately on start
    run_cycle()

    # Schedule subsequent runs with jitter
    jitter = random.randint(-JITTER_MINUTES, JITTER_MINUTES)
    next_minutes = CYCLE_INTERVAL_HOURS * 60 + jitter
    log.info("Next cycle in %d minutes.", next_minutes)
    schedule.every(next_minutes).minutes.do(_run_with_jitter).tag("cycle")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
