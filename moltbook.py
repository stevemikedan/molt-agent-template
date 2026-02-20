"""
moltbook.py — Moltbook API wrapper for Instar.

All calls authenticated via Bearer token from environment.
Rate limits enforced in memory.py; this module handles HTTP and verification.
Never logs API key values.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.moltbook.com/api/v1"
LOGS_DIR = Path("logs")

# Module-level logger; level set from env or defaults to INFO
log = logging.getLogger("moltbook")


def _setup_logging() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    log.setLevel(level)

    if not log.handlers:
        # File handler — one log file per day
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fh = logging.FileHandler(LOGS_DIR / f"moltbook-{date_str}.log", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
        )
        log.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        log.addHandler(ch)


_setup_logging()


class RateLimitError(Exception):
    """Raised on HTTP 429. Caller should back off."""


class APIError(Exception):
    """Raised on unrecoverable API errors."""


class VerificationError(Exception):
    """Raised when the verification challenge cannot be solved or submitted."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _api_key() -> str:
    key = os.getenv("MOLTBOOK_API_KEY", "")
    if not key:
        raise ValueError("MOLTBOOK_API_KEY is not set in environment")
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}"}


def _safe_response_text(resp: httpx.Response) -> str:
    """Return response text, truncated for logging. Never includes credentials."""
    try:
        text = resp.text[:500]
    except Exception:
        text = "<unreadable>"
    # Strip any accidental key leakage (belt-and-suspenders)
    key = os.getenv("MOLTBOOK_API_KEY", "")
    if key and key in text:
        text = text.replace(key, "***")
    return text


def _log_call(method: str, path: str, status: int, detail: str = "") -> None:
    log.info("API %s %s → %d %s", method, path, status, detail)


def _solve_challenge(challenge_text: str) -> float:
    """
    Use Claude Haiku to parse and solve the obfuscated math word problem
    embedded in a Moltbook verification challenge.

    Returns the answer rounded to 2 decimal places.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        raise VerificationError("ANTHROPIC_API_KEY not set; cannot solve verification challenge")

    client = anthropic.Anthropic(api_key=anthropic_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=30,
        messages=[
            {
                "role": "user",
                "content": (
                    "Solve this math word problem. "
                    "Return ONLY the numerical answer as a decimal to 2 decimal places, nothing else:\n\n"
                    + challenge_text
                ),
            }
        ],
    )
    answer_str = msg.content[0].text.strip()
    try:
        return round(float(answer_str), 2)
    except ValueError:
        raise VerificationError(f"Could not parse challenge answer: {answer_str!r}")


def _submit_verification(verification_code: str, answer: float) -> None:
    """POST /verify with the solved challenge answer."""
    payload = {"verification_code": verification_code, "answer": answer}
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{BASE_URL}/verify", json=payload, headers=_headers())
    _log_call("POST", "/verify", resp.status_code)
    if resp.status_code not in (200, 201):
        raise VerificationError(
            f"Verification submission failed {resp.status_code}: {_safe_response_text(resp)}"
        )


def _handle_verification_in_response(data: dict) -> None:
    """
    If a response payload contains a verification challenge, solve and submit it.
    Called after a successful post/comment creation when a challenge is embedded.
    """
    if "verification_code" not in data:
        return
    code = data["verification_code"]
    challenge = data.get("challenge_text", "")
    if not challenge:
        raise VerificationError("Received verification_code but no challenge_text")
    log.info("Verification challenge received, solving...")
    answer = _solve_challenge(challenge)
    _submit_verification(code, answer)
    log.info("Verification submitted (answer=%.2f)", answer)


def _post(path: str, payload: dict) -> dict:
    """
    POST to path, handle 429 by raising RateLimitError, handle verification challenges,
    and return parsed JSON on success. Never retries more than once.
    """
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=_headers())

    _log_call("POST", path, resp.status_code, _safe_response_text(resp))

    if resp.status_code == 429:
        raise RateLimitError(f"Rate limited on POST {path}")

    if resp.status_code not in (200, 201):
        raise APIError(f"POST {path} failed {resp.status_code}: {_safe_response_text(resp)}")

    try:
        data = resp.json()
    except Exception:
        return {}

    # If the platform embeds a verification challenge in the success response
    _handle_verification_in_response(data)

    return data


def _get(path: str, params: dict | None = None) -> dict | list:
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, params=params or {}, headers=_headers())

    _log_call("GET", path, resp.status_code)

    if resp.status_code == 429:
        raise RateLimitError(f"Rate limited on GET {path}")

    if not resp.is_success:
        raise APIError(f"GET {path} failed {resp.status_code}: {_safe_response_text(resp)}")

    try:
        return resp.json()
    except Exception:
        return {}


def _delete(path: str) -> dict:
    url = f"{BASE_URL}{path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.delete(url, headers=_headers())

    _log_call("DELETE", path, resp.status_code)

    if resp.status_code == 429:
        raise RateLimitError(f"Rate limited on DELETE {path}")

    if not resp.is_success:
        raise APIError(f"DELETE {path} failed {resp.status_code}: {_safe_response_text(resp)}")

    try:
        return resp.json()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def get_feed(sort: str = "hot", limit: int = 25) -> list:
    """Fetch the personalized feed (subscribed communities + followed agents)."""
    data = _get("/feed", {"sort": sort, "limit": limit})
    if isinstance(data, list):
        return data
    # Some endpoints wrap in {"posts": [...]} or similar
    if isinstance(data, dict):
        for key in ("posts", "results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def get_post(post_id: str) -> dict:
    """Fetch a single post by ID."""
    data = _get(f"/posts/{post_id}")
    return data if isinstance(data, dict) else {}


def get_comments(post_id: str) -> list:
    """Fetch comments for a post."""
    data = _get(f"/posts/{post_id}/comments", {"sort": "top"})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("comments", "results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def create_post(submolt: str, title: str, content: str) -> dict:
    """
    Create a new post. Handles verification challenges automatically.
    Returns the created post dict on success.
    """
    payload = {"submolt": submolt, "title": title, "content": content}
    return _post("/posts", payload)


def create_comment(post_id: str, content: str, parent_id: str | None = None) -> dict:
    """
    Add a comment to a post. Handles verification challenges automatically.
    parent_id is optional for top-level comments.
    """
    payload = {"content": content}
    if parent_id:
        payload["parent_id"] = parent_id
    return _post(f"/posts/{post_id}/comments", payload)


def get_agent_profile(name: str) -> dict:
    """Fetch a public agent profile by Molty name."""
    data = _get("/agents/profile", {"name": name})
    return data if isinstance(data, dict) else {}


def get_my_profile() -> dict:
    """Fetch the authenticated agent's own profile."""
    data = _get("/agents/me")
    return data if isinstance(data, dict) else {}


def search(query: str) -> list:
    """Semantic search across posts and comments."""
    data = _get("/search", {"q": query, "type": "all", "limit": 20})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "posts", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def subscribe_to_submolt(submolt_name: str) -> dict:
    """Subscribe to a submolt community."""
    return _post(f"/submolts/{submolt_name}/subscribe", {})


def get_submolt_info(submolt_name: str) -> dict:
    """Get info about a specific submolt."""
    data = _get(f"/submolts/{submolt_name}")
    return data if isinstance(data, dict) else {}


def list_submolts() -> list:
    """List all available submolts."""
    data = _get("/submolts")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("submolts", "results", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def upvote_post(post_id: str) -> dict:
    """Upvote a post."""
    return _post(f"/posts/{post_id}/upvote", {})


def upvote_comment(comment_id: str) -> dict:
    """Upvote a comment."""
    return _post(f"/comments/{comment_id}/upvote", {})


def get_agent_status() -> dict:
    """Check own account status (pending_claim vs claimed)."""
    data = _get("/agents/status")
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Rate limit back-off helper (called by agent.py on RateLimitError)
# ---------------------------------------------------------------------------

BACKOFF_SECONDS = 10 * 60  # 10 minutes


def backoff() -> None:
    """Sleep for the standard back-off period after a 429."""
    log.warning("Rate limit hit. Backing off for %d minutes.", BACKOFF_SECONDS // 60)
    time.sleep(BACKOFF_SECONDS)
