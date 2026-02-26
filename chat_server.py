"""
chat_server.py — Lightweight HTTP server for direct chat with the agent.

Runs as a daemon thread alongside the scheduler so it never blocks the
main agent loop.  Persists messages to {STATE_DIR}/chat_messages.json.

Endpoints:
    POST /chat          — Send a message, get a response in the agent's voice
    GET  /chat/history  — Retrieve stored conversation history
    GET  /health        — Connectivity check
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import character

log = logging.getLogger("chat_server")

_STATE_DIR = os.getenv("STATE_DIR", "")
_HISTORY_FILE = Path(_STATE_DIR) / "chat_messages.json" if _STATE_DIR else Path("chat_messages.json")
_AGENT_NAME = os.getenv("AGENT_NAME", "Agent")
_API_KEY = os.getenv("MOLTBOOK_API_KEY", "")

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

def _load_history() -> list[dict]:
    with _lock:
        if _HISTORY_FILE.exists():
            try:
                return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []


def _save_history(history: list[dict]) -> None:
    with _lock:
        _HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _append_messages(owner_msg: dict, agent_msg: dict) -> None:
    history = _load_history()
    history.append(owner_msg)
    history.append(agent_msg)
    # Keep last 200 messages to avoid unbounded growth
    if len(history) > 200:
        history = history[-200:]
    _save_history(history)

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _ChatHandler(BaseHTTPRequestHandler):
    """Minimal request handler — no framework dependencies."""

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    # ---- auth ----
    def _check_auth(self) -> bool:
        if not _API_KEY:
            return True  # no key configured — allow (local dev)
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {_API_KEY}":
            return True
        self._json_response(401, {"error": "Unauthorized"})
        return False

    # ---- helpers ----
    def _json_response(self, status: int, data: dict | list) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ---- routing ----
    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"ok": True, "agent": _AGENT_NAME})
        elif self.path == "/chat/history":
            if not self._check_auth():
                return
            self._json_response(200, _load_history())
        else:
            self._json_response(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/chat":
            if not self._check_auth():
                return
            try:
                payload = json.loads(self._read_body())
            except (json.JSONDecodeError, ValueError):
                self._json_response(400, {"error": "Invalid JSON"})
                return

            message = (payload.get("message") or "").strip()
            if not message:
                self._json_response(400, {"error": "message is required"})
                return

            # Build recent history for context
            history = _load_history()
            recent = history[-20:]  # last 10 exchanges

            try:
                response_text = character.generate_chat_response(message, recent)
            except Exception as exc:
                log.error("Chat generation error: %s", exc)
                self._json_response(500, {"error": "Generation failed"})
                return

            now = datetime.now(timezone.utc).isoformat()
            msg_id = str(uuid.uuid4())

            owner_msg = {
                "id": str(uuid.uuid4()),
                "role": "owner",
                "content": message,
                "timestamp": now,
            }
            agent_msg = {
                "id": msg_id,
                "role": "agent",
                "content": response_text,
                "timestamp": now,
            }
            _append_messages(owner_msg, agent_msg)

            self._json_response(200, {
                "id": msg_id,
                "response": response_text,
                "timestamp": now,
            })
        else:
            self._json_response(404, {"error": "Not found"})

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def start_chat_server() -> None:
    """Start the chat HTTP server on a daemon thread.

    Reads PORT from the environment (Railway sets this). Falls back to 8080.
    """
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Chat server listening on 0.0.0.0:%d", port)
