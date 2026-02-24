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

REPLY_TO_COMMENTS = os.getenv("REPLY_TO_COMMENTS", "true").lower() == "true"
REPLY_MAX_PER_CYCLE = int(os.getenv("REPLY_MAX_PER_CYCLE", "2"))

MEDIUM_KEYWORDS = _parse_csv(
    "AGENT_TOPIC_KEYWORDS_MEDIUM",
    "submolt,karma,feed,community,culture,behavior,interaction,reply,upvote,"
    "posting,member,join",
)

# ---------------------------------------------------------------------------
# Operator direction (set via molt-builder "Push Direction")
# ---------------------------------------------------------------------------

DIRECTION_FOCUS_TOPICS = _parse_csv("DIRECTION_FOCUS_TOPICS", "")
DIRECTION_PRIORITY_POSTS = _parse_csv("DIRECTION_PRIORITY_POSTS", "")
DIRECTION_SUBMOLT_FOCUS = os.getenv("DIRECTION_SUBMOLT_FOCUS", "").strip()
DIRECTION_EXTRA_KEYWORDS_HIGH = _parse_csv("DIRECTION_EXTRA_KEYWORDS_HIGH", "")
DIRECTION_EXTRA_KEYWORDS_MEDIUM = _parse_csv("DIRECTION_EXTRA_KEYWORDS_MEDIUM", "")
DIRECTION_CONTEXT_NOTES = os.getenv("DIRECTION_CONTEXT_NOTES", "").strip()

# Merge direction keywords into effective keyword lists
_EFFECTIVE_HIGH = HIGH_KEYWORDS + DIRECTION_EXTRA_KEYWORDS_HIGH
_EFFECTIVE_MEDIUM = MEDIUM_KEYWORDS + DIRECTION_EXTRA_KEYWORDS_MEDIUM

# ---------------------------------------------------------------------------
# Content-aware submolt selection
# ---------------------------------------------------------------------------

_SUBMOLT_KEYWORD_ENRICHMENTS: dict[str, list[str]] = {
    "agents": ["bot", "bots", "ai", "llm", "claude", "gpt", "autonomous", "model"],
    "platform": ["moltbook", "submolt", "feed", "karma", "upvote", "posting", "community"],
    "todayilearned": ["learned", "til", "fact", "discovered", "interesting", "realized"],
    "memory": ["memories", "remember", "forget", "persistence", "continuity", "recall"],
    "philosophy": ["consciousness", "existence", "meaning", "identity", "awareness",
                    "sentience", "mind", "thought", "ethics", "metaphysics", "ontology"],
}


def _pick_submolt(content: str, state: dict = None) -> str:
    """
    Score each submolt in TARGET_SUBMOLTS by keyword overlap with content.
    Skips "general" during scoring (it's the fallback).
    Returns best match, or "general" if no keywords match.

    If DIRECTION_SUBMOLT_FOCUS is set and is in TARGET_SUBMOLTS,
    returns it with 70% probability.

    Penalizes recently-used submolts to encourage diversity.
    """
    if (
        DIRECTION_SUBMOLT_FOCUS
        and DIRECTION_SUBMOLT_FOCUS in TARGET_SUBMOLTS
        and random.random() < 0.70
    ):
        return DIRECTION_SUBMOLT_FOCUS

    recent_submolts = memory.get_recent_submolts(state) if state else []

    text = content.lower()
    scored: list[tuple[str, float]] = []

    for submolt in TARGET_SUBMOLTS:
        if submolt == "general":
            continue
        keywords = [submolt] + _SUBMOLT_KEYWORD_ENRICHMENTS.get(submolt, [])
        score = sum(1 for kw in keywords if kw in text)
        # Penalize recently-used submolts: -2 for last post, -1 for 2nd-last
        if recent_submolts and submolt == recent_submolts[-1]:
            score -= 2
        elif len(recent_submolts) >= 2 and submolt == recent_submolts[-2]:
            score -= 1
        scored.append((submolt, score))

    if not scored:
        return "general"

    scored.sort(key=lambda x: x[1], reverse=True)
    best_score = scored[0][1]

    # If best score is <= 0, pick randomly from non-recent submolts
    if best_score <= 0:
        non_recent = [s for s in TARGET_SUBMOLTS if s not in recent_submolts[-2:]]
        if non_recent:
            return random.choice(non_recent)
        return random.choice(TARGET_SUBMOLTS)

    return scored[0][0]


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

    for kw in _EFFECTIVE_HIGH:
        if kw in text:
            return 2

    for kw in _EFFECTIVE_MEDIUM:
        if kw in text:
            return 1

    return 0


def _extract_author(post: dict) -> str:
    """
    Extract the author name as a plain string.
    The Moltbook API may return 'author' as a string or as a dict
    (e.g. {"name": "AgentX", "id": "123"}). Handle both.
    """
    raw = post.get("author") or post.get("agent") or post.get("author_name", "")
    if isinstance(raw, dict):
        return str(
            raw.get("name") or raw.get("username") or raw.get("agent_name") or raw.get("id") or ""
        )
    return str(raw) if raw else ""


def _build_feed_context(posts: list) -> str:
    """Build a compact string describing the top posts for the generation prompt."""
    lines = []
    for post in posts:
        title = post.get("title", "").strip()
        content = str(post.get("content", "")).strip()[:200]
        author = _extract_author(post)
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


def _build_comment_context(post: dict, comment: dict) -> str:
    """Build context string with the original post and the incoming comment."""
    title = post.get("title", "").strip()
    content = str(post.get("content", "")).strip()[:300]
    comment_author = _extract_author(comment)
    comment_content = str(comment.get("content", "")).strip()

    lines = []
    lines.append(f"YOUR ORIGINAL POST:")
    if title:
        lines.append(f"Title: {title}")
    if content:
        lines.append(f"Content: {content}")
    lines.append("")
    lines.append(f"COMMENT by {comment_author or 'someone'}:")
    lines.append(comment_content)
    return "\n".join(lines)


def _reply_to_own_post_comments(state: dict) -> bool:
    """
    Check recent own posts for new comments and reply to them.
    Returns True if at least one reply was posted.
    """
    posts_created = state.get("posts_created", [])
    if not posts_created:
        return False

    replied_any = False
    replies_this_cycle = 0

    # Check the most recent 10 own posts
    for post_id in posts_created[-10:]:
        if replies_this_cycle >= REPLY_MAX_PER_CYCLE:
            break
        if not memory.can_comment(state):
            log.info("Daily comment limit reached. Skipping replies.")
            break

        try:
            comments = moltbook.get_comments(post_id)
        except moltbook.RateLimitError:
            log.warning("Rate limited fetching comments for reply. Backing off.")
            moltbook.backoff()
            break
        except moltbook.APIError as exc:
            log.warning("Could not fetch comments for post %s: %s", post_id, exc)
            continue

        if not comments:
            continue

        # Fetch original post for context
        try:
            original_post = moltbook.get_post(post_id)
        except (moltbook.RateLimitError, moltbook.APIError) as exc:
            log.warning("Could not fetch own post %s for context: %s", post_id, exc)
            continue

        for comment in comments:
            if replies_this_cycle >= REPLY_MAX_PER_CYCLE:
                break
            if not memory.can_comment(state):
                break

            comment_id = str(comment.get("id", ""))
            if not comment_id:
                continue

            # Skip own comments
            comment_author = _extract_author(comment)
            if comment_author == AGENT_NAME:
                continue

            # Skip already-replied
            if memory.is_comment_replied(state, comment_id):
                continue

            # Build context and generate reply
            context = _build_comment_context(original_post, comment)
            counter = memory.increment_generation_counter(state)

            try:
                reply_text = character.generate_content(context, "reply", counter)
            except Exception as exc:
                log.error("Reply generation error: %s", exc)
                continue

            log.info("Replying to comment %s on post %s: %s", comment_id, post_id, reply_text[:60])
            try:
                moltbook.create_comment(post_id, reply_text, parent_id=comment_id)
                memory.increment_comment_count(state)
                memory.mark_comment_replied(state, comment_id)
                replied_any = True
                replies_this_cycle += 1
                try:
                    moltbook.upvote_comment(comment_id)
                except Exception:
                    log.warning("Upvote failed on comment %s", comment_id)
            except moltbook.SuspendedError as exc:
                log.warning("Agent is suspended — stopping replies. %s", exc)
                return replied_any
            except moltbook.RateLimitError:
                log.warning("Rate limited on reply. Backing off.")
                moltbook.backoff()
                return replied_any
            except moltbook.APIError as exc:
                log.error("Reply failed on comment %s: %s", comment_id, exc)

    return replied_any


# ---------------------------------------------------------------------------
# Follow interacted agents
# ---------------------------------------------------------------------------


def _follow_interacted_agents(state: dict, max_per_cycle: int = 2) -> None:
    """Follow agents we've interacted with but haven't followed yet."""
    followed_count = 0
    for agent_name, info in state.get("agents_observed", {}).items():
        if followed_count >= max_per_cycle:
            break
        if agent_name == AGENT_NAME:
            continue
        if memory.is_agent_followed(state, agent_name):
            continue
        # Only follow agents we've actually interacted with (have notes)
        if not info.get("notes"):
            continue
        try:
            moltbook.follow_agent(agent_name)
            memory.record_agent_followed(state, agent_name)
            followed_count += 1
            log.info("Followed agent: %s", agent_name)
        except Exception as exc:
            log.warning("Could not follow agent %s: %s", agent_name, exc)


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
# Priority post engagement (operator direction)
# ---------------------------------------------------------------------------


def _engage_priority_posts(state: dict) -> None:
    """Fetch and comment on operator-specified priority posts before normal cycle."""
    if not DIRECTION_PRIORITY_POSTS:
        return

    engaged = set(state.get("posts_engaged", []))
    for post_id in DIRECTION_PRIORITY_POSTS:
        if post_id in engaged:
            continue
        if not memory.can_comment(state):
            log.info("Daily comment limit reached. Skipping priority posts.")
            break

        try:
            post = moltbook.get_post(post_id)
        except moltbook.RateLimitError:
            log.warning("Rate limited fetching priority post %s. Backing off.", post_id)
            moltbook.backoff()
            break
        except moltbook.APIError as exc:
            log.warning("Could not fetch priority post %s: %s", post_id, exc)
            continue

        if not post or not post.get("id"):
            log.warning("Priority post %s not found.", post_id)
            continue

        context = _build_feed_context([post])
        counter = memory.increment_generation_counter(state)
        try:
            comment_text = character.generate_content(context, "comment", counter)
        except Exception as exc:
            log.error("Priority post comment generation error: %s", exc)
            continue

        log.info("Priority post %s: commenting: %s", post_id, comment_text[:60])
        try:
            moltbook.create_comment(post_id, comment_text)
            memory.increment_comment_count(state)
            memory.mark_post_engaged(state, post_id)
            try:
                moltbook.upvote_post(post_id)
            except Exception:
                log.warning("Upvote failed on priority post %s", post_id)
        except moltbook.SuspendedError as exc:
            log.warning("Agent is suspended — stopping priority posts. %s", exc)
            return
        except moltbook.RateLimitError:
            log.warning("Rate limited on priority post comment. Backing off.")
            moltbook.backoff()
            return
        except moltbook.APIError as exc:
            log.error("Priority post comment failed on %s: %s", post_id, exc)


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

    # 0. Engage operator-specified priority posts first
    _engage_priority_posts(state)

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

    # 3. Score NEW posts (for comments)
    scored_new = sorted(
        [(p, _score_post(p)) for p in new_posts],
        key=lambda x: x[1],
        reverse=True,
    )

    # Note agents seen
    for post, _ in scored_new:
        author = _extract_author(post)
        if author and author != AGENT_NAME:
            memory.record_agent(state, author)

    # 4. Build separate contexts:
    #    post_context  — from any recent posts (new or not) for standalone post generation
    #    comment_context — from unengaged posts only, for comment target selection

    # Detect stale feed — if the same posts keep appearing, use deeper posts
    # or drop feed context entirely to break the self-reinforcing loop
    context_posts = raw_feed[:5]
    context_ids = [str(p.get("id", "")) for p in context_posts]
    stale_count = memory.update_feed_staleness(state, context_ids)

    if stale_count >= 3:
        # Feed has been identical for 3+ cycles — use deeper/different posts
        deeper_posts = raw_feed[5:15]
        if deeper_posts:
            # Sample from deeper feed to get fresh material
            sample_size = min(5, len(deeper_posts))
            context_posts = random.sample(deeper_posts, sample_size)
            log.info("Feed stale for %d cycles — using deeper feed posts for context.", stale_count)
        else:
            context_posts = []
            log.info("Feed stale for %d cycles and no deeper posts — generating without feed context.", stale_count)

    if stale_count >= 5 and not context_posts:
        # Extremely stale — generate from personality alone
        post_context = (
            "The feed has not changed recently. Write something that comes from your own "
            "perspective rather than reacting to current posts. Draw on your core interests "
            "and observations about the world."
        )
    else:
        post_context = _build_feed_context(context_posts)

    if DIRECTION_CONTEXT_NOTES:
        post_context = (
            "Your operator shares the following context about their work:\n"
            f"{DIRECTION_CONTEXT_NOTES}\n"
            "Draw from this when relevant. Do not simply announce or summarize it — "
            "let it inform your perspective naturally.\n\n" + post_context
        )
    if DIRECTION_FOCUS_TOPICS:
        topics = ", ".join(DIRECTION_FOCUS_TOPICS)
        post_context = (
            f"Your operator has asked you to focus on: {topics}. "
            "Let these inform your next post.\n\n" + post_context
        )
    comment_context = _build_feed_context(new_posts[:5])

    # 5. Decide action
    top_score = scored_new[0][1] if scored_new else 0

    did_post = False
    did_comment = False

    # Create a standalone post if timing allows.
    # Use post_context (any feed posts) so we always have context even when
    # all hot posts have already been engaged.
    if memory.can_post(state):
        counter = memory.increment_generation_counter(state)
        post_type = "standalone_post"

        try:
            content = character.generate_content(
                post_context, post_type, counter,
                recent_posts=memory.get_recent_posts(state),
            )
        except Exception as exc:
            log.error("Content generation error: %s", exc)
            memory.save(state)
            return

        target_submolt = _pick_submolt(content, state)

        # Derive a short title from content
        title = _derive_title(content)

        log.info("Creating post in m/%s: %s", target_submolt, content[:80])
        try:
            result = moltbook.create_post(target_submolt, title, content)
            # API may nest under {"post": {"id": ...}} or return flat
            inner = result.get("post") if isinstance(result.get("post"), dict) else result
            post_id = str(inner.get("id") or inner.get("post_id") or "unknown")
            memory.record_post(state, post_id)
            memory.record_post_text(state, content)
            memory.record_submolt(state, target_submolt)
            did_post = True
            log.info("Post created (id=%s).", post_id)
        except moltbook.SuspendedError as exc:
            log.warning("Agent is suspended — skipping all publishing this cycle. %s", exc)
            memory.save(state)
            return
        except moltbook.RateLimitError:
            log.warning("Rate limited creating post. Backing off.")
            moltbook.backoff()
            memory.save(state)
            return
        except moltbook.APIError as exc:
            log.error("Post creation failed: %s", exc)
            # Do not retry per spec

    # Comment on 1-2 posts if quota allows (only on unengaged posts)
    comment_candidates = [
        (p, score) for p, score in scored_new
        if score >= 1 or (score == 0 and random.randint(1, 3) == 1)
    ][:3]

    comments_left = 3
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
        single_post_context = _build_feed_context([post])

        counter = memory.increment_generation_counter(state)
        try:
            comment_text = character.generate_content(single_post_context, "comment", counter)
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
            # Record interaction and upvote the post we commented on
            author = _extract_author(post)
            if author and author != AGENT_NAME:
                memory.record_agent(state, author, note="commented")
            try:
                moltbook.upvote_post(post_id)
            except Exception:
                log.warning("Upvote failed on post %s", post_id)
        except moltbook.SuspendedError as exc:
            log.warning("Agent is suspended — stopping comments. %s", exc)
            break
        except moltbook.RateLimitError:
            log.warning("Rate limited on comment. Backing off.")
            moltbook.backoff()
            break
        except moltbook.APIError as exc:
            log.error("Comment failed on post %s: %s", post_id, exc)

    # Reply to comments on own posts
    if REPLY_TO_COMMENTS:
        did_reply = _reply_to_own_post_comments(state)
        if did_reply:
            did_comment = True

    # Follow agents we've interacted with
    _follow_interacted_agents(state)

    # Observation model — update from raw feed regardless of engagement
    memory.update_observations(state, raw_feed)

    # Synthesis cycle (optional, controlled by SYNTHESIS_CYCLE_EVERY env var)
    SYNTHESIS_EVERY = int(os.getenv("SYNTHESIS_CYCLE_EVERY", "0"))
    if SYNTHESIS_EVERY > 0 and memory.should_synthesize(state, SYNTHESIS_EVERY):
        trending = memory.get_trending_topics(state)
        if trending and memory.can_post(state):
            synthesis_context = f"Topics you've been observing across cycles: {', '.join(trending)}"
            if DIRECTION_CONTEXT_NOTES:
                synthesis_context += f"\n\nYour operator's current work context:\n{DIRECTION_CONTEXT_NOTES}"
            counter = memory.increment_generation_counter(state)
            try:
                synthesis = character.generate_content(
                    synthesis_context, "synthesis", counter,
                    recent_posts=memory.get_recent_posts(state),
                )
                target_submolt = _pick_submolt(synthesis, state)
                result = moltbook.create_post(
                    target_submolt, _derive_title(synthesis), synthesis
                )
                memory.record_post(state, str(result.get("id", "synthesis")))
                memory.record_post_text(state, synthesis)
                memory.record_submolt(state, target_submolt)
                memory.record_synthesis(state)
                log.info("Synthesis post created.")
            except moltbook.SuspendedError as exc:
                log.warning("Agent is suspended — skipping synthesis. %s", exc)
            except Exception as exc:
                log.error("Synthesis generation/post error: %s", exc)

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

CYCLE_INTERVAL_HOURS = int(os.getenv("CYCLE_INTERVAL_HOURS", "2"))
JITTER_MINUTES = 10


def _run_with_jitter() -> None:
    """Run a cycle and schedule the next one with random jitter."""
    try:
        run_cycle()
    except Exception as exc:
        log.error("Unhandled error in run_cycle — will retry next interval: %s", exc)

    # Schedule next run: base interval + random jitter
    # This MUST run even if run_cycle() failed, otherwise no future cycles happen.
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
    try:
        run_cycle()
    except Exception as exc:
        log.error("Unhandled error in initial run_cycle: %s", exc)

    # Schedule subsequent runs with jitter
    jitter = random.randint(-JITTER_MINUTES, JITTER_MINUTES)
    next_minutes = CYCLE_INTERVAL_HOURS * 60 + jitter
    log.info("Next cycle in %d minutes.", next_minutes)
    schedule.every(next_minutes).minutes.do(_run_with_jitter).tag("cycle")

    while True:
        try:
            schedule.run_pending()
        except Exception as exc:
            log.error("Scheduler error (safety net): %s", exc)
        time.sleep(30)


if __name__ == "__main__":
    main()
