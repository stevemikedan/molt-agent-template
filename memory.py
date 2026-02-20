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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
