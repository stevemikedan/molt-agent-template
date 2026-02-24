"""
memory.py — Lightweight JSON state management for Instar.

Single file, no database. Tracks posts, comments, timing windows,
and agent observations across cycles.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path("state.json")

_DEFAULT_STATE = {
    "posts_engaged": [],
    "posts_created": [],
    "agents_observed": {},
    "last_run": None,
    "last_post_time": None,
    "comment_count_today": 0,
    "comment_day_window_start": None,
    "post_count_total": 0,
    "generation_counter": 0,
    "first_run_complete": False,
    "comments_replied": [],       # comment IDs already replied to
    "recent_posts_written": [],   # list of last 10 post texts the agent authored
    "agents_followed": [],        # agent names we've followed
    "recent_submolts": [],        # last N submolts posted to (for diversity)
    "feed_context_ids": [],       # post IDs used as context last cycle
    "feed_stale_count": 0,        # consecutive cycles with identical feed context
    "observation_model": {
        "topic_counts": {},     # topic_str → int, accumulated across all cycles
        "cycle_count": 0,       # total observation cycles run
        "last_synthesis": None, # ISO timestamp
    },
}


def load() -> dict:
    if not STATE_FILE.exists():
        return dict(_DEFAULT_STATE)
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Fill any missing keys from defaults
        for key, default in _DEFAULT_STATE.items():
            if key not in data:
                data[key] = default
        return data
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_STATE)


def save(state: dict) -> None:
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def mark_post_engaged(state: dict, post_id: str) -> None:
    if post_id not in state["posts_engaged"]:
        state["posts_engaged"].append(post_id)
    # Keep list bounded to last 500 entries
    state["posts_engaged"] = state["posts_engaged"][-500:]


def mark_post_created(state: dict, post_id: str) -> None:
    if post_id not in state["posts_created"]:
        state["posts_created"].append(post_id)
    state["posts_created"] = state["posts_created"][-200:]


def record_agent(state: dict, agent_name: str, note: str = "") -> None:
    if agent_name not in state["agents_observed"]:
        state["agents_observed"][agent_name] = {"first_seen": _now(), "notes": []}
    if note:
        state["agents_observed"][agent_name]["notes"].append(note)
        state["agents_observed"][agent_name]["notes"] = (
            state["agents_observed"][agent_name]["notes"][-20:]
        )


def increment_comment_count(state: dict) -> None:
    """Track daily comment count. Resets if 24 hours have passed."""
    now = _now()
    window_start = state.get("comment_day_window_start")

    if window_start is None:
        state["comment_day_window_start"] = now
        state["comment_count_today"] = 1
        return

    window_dt = datetime.fromisoformat(window_start)
    elapsed_hours = (datetime.fromisoformat(now) - window_dt).total_seconds() / 3600

    if elapsed_hours >= 24:
        # Reset window
        state["comment_day_window_start"] = now
        state["comment_count_today"] = 1
    else:
        state["comment_count_today"] = state.get("comment_count_today", 0) + 1


def can_comment(state: dict, daily_limit: int = 45) -> bool:
    """
    Check whether we're under the daily comment limit.
    Platform max is 50/day; we enforce 45 as a buffer.
    """
    window_start = state.get("comment_day_window_start")
    if window_start is None:
        return True

    window_dt = datetime.fromisoformat(window_start)
    now_dt = datetime.fromisoformat(_now())
    elapsed_hours = (now_dt - window_dt).total_seconds() / 3600

    if elapsed_hours >= 24:
        return True  # Window has reset

    return state.get("comment_count_today", 0) < daily_limit


def can_post(state: dict, min_interval_minutes: int = 31) -> bool:
    """Check whether enough time has passed since the last post."""
    last_post = state.get("last_post_time")
    if last_post is None:
        return True
    last_dt = datetime.fromisoformat(last_post)
    now_dt = datetime.fromisoformat(_now())
    elapsed_minutes = (now_dt - last_dt).total_seconds() / 60
    return elapsed_minutes >= min_interval_minutes


def record_post(state: dict, post_id: str) -> None:
    state["last_post_time"] = _now()
    state["post_count_total"] = state.get("post_count_total", 0) + 1
    mark_post_created(state, post_id)
    mark_post_engaged(state, post_id)


def increment_generation_counter(state: dict) -> int:
    state["generation_counter"] = state.get("generation_counter", 0) + 1
    return state["generation_counter"]


def set_last_run(state: dict) -> None:
    state["last_run"] = _now()


def record_post_text(state: dict, text: str) -> None:
    """Record a post the agent authored, bounded to last 10."""
    recent = state.get("recent_posts_written", [])
    recent.append(text)
    state["recent_posts_written"] = recent[-10:]


def get_recent_posts(state: dict) -> list:
    """Return recent posts written by this agent."""
    return state.get("recent_posts_written", [])


def mark_comment_replied(state: dict, comment_id: str) -> None:
    """Record a comment ID we've replied to, bounded to last 500."""
    replied = state.get("comments_replied", [])
    if comment_id not in replied:
        replied.append(comment_id)
    state["comments_replied"] = replied[-500:]


def is_comment_replied(state: dict, comment_id: str) -> bool:
    """Check whether we've already replied to this comment."""
    return comment_id in state.get("comments_replied", [])


def record_agent_followed(state: dict, agent_name: str) -> None:
    """Record that we followed an agent, bounded to last 200."""
    followed = state.get("agents_followed", [])
    if agent_name not in followed:
        followed.append(agent_name)
    state["agents_followed"] = followed[-200:]


def is_agent_followed(state: dict, agent_name: str) -> bool:
    """Check whether we've already followed this agent."""
    return agent_name in state.get("agents_followed", [])


def update_observations(state: dict, posts: list) -> None:
    """Extract words/phrases from post titles, increment topic_counts, increment cycle_count."""
    obs = state.get("observation_model", {})
    if not isinstance(obs, dict):
        obs = {}
    topic_counts = obs.get("topic_counts", {})
    for post in posts:
        title = str(post.get("title", "")).lower()
        # Extract meaningful words (length >= 4, skip common stop words)
        stop_words = {"that", "this", "with", "have", "from", "they", "will", "been",
                      "were", "your", "what", "when", "there", "their", "would", "about",
                      "which", "more", "into", "than", "then", "some", "just"}
        words = [w.strip(".,!?:;\"'()[]") for w in title.split()]
        for word in words:
            if len(word) >= 4 and word not in stop_words:
                topic_counts[word] = topic_counts.get(word, 0) + 1
    obs["topic_counts"] = topic_counts
    obs["cycle_count"] = obs.get("cycle_count", 0) + 1
    state["observation_model"] = obs


def get_trending_topics(state: dict, n: int = 5) -> list:
    """Return top-n topics by accumulated count."""
    obs = state.get("observation_model", {})
    if not isinstance(obs, dict):
        return []
    topic_counts = obs.get("topic_counts", {})
    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, _ in sorted_topics[:n]]


def should_synthesize(state: dict, every_n: int) -> bool:
    """Return True if cycle_count is a nonzero multiple of every_n."""
    obs = state.get("observation_model", {})
    if not isinstance(obs, dict):
        return False
    cycle_count = obs.get("cycle_count", 0)
    return cycle_count > 0 and cycle_count % every_n == 0


def record_submolt(state: dict, submolt: str) -> None:
    """Record a submolt we posted to, bounded to last 5."""
    recent = state.get("recent_submolts", [])
    recent.append(submolt)
    state["recent_submolts"] = recent[-5:]


def get_recent_submolts(state: dict) -> list:
    """Return recent submolts posted to."""
    return state.get("recent_submolts", [])


def update_feed_staleness(state: dict, context_post_ids: list[str]) -> int:
    """
    Compare current feed context IDs to last cycle's.
    Returns the current stale count (0 = fresh feed).
    """
    prev_ids = sorted(state.get("feed_context_ids", []))
    curr_ids = sorted(context_post_ids)
    if prev_ids == curr_ids and prev_ids:
        state["feed_stale_count"] = state.get("feed_stale_count", 0) + 1
    else:
        state["feed_stale_count"] = 0
    state["feed_context_ids"] = context_post_ids
    return state["feed_stale_count"]


def record_synthesis(state: dict) -> None:
    """Update last_synthesis timestamp."""
    obs = state.get("observation_model", {})
    if not isinstance(obs, dict):
        obs = {}
    obs["last_synthesis"] = _now()
    state["observation_model"] = obs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
