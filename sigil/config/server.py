"""
SigilGHR Config Server
======================
A zero-dependency local HTTP server (Python stdlib only) that:

  • Serves the HTML+CSS config page at  http://localhost:7272/
  • GET  /api/config   → returns current gesture→key mappings as JSON
  • POST /api/config   → saves updated mappings to sigil_config.json
  • GET  /api/status   → returns the active window title (for the UI badge)

Design notes
------------
* Runs in a **daemon thread** — the server exits automatically when the main
  process exits; no explicit shutdown required.
* Uses Python's built-in http.server; no Flask/FastAPI dependency.
* Config is stored as a flat JSON file (sigil_config.json) in the project
  working directory.  On load, MacroEngine reads the file and overrides the
  default key tuples in each plugin's _ACTION_MAP.
* The server is intentionally NOT started automatically; callers must call
  ConfigServer.start() explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from sigil.core.window_watcher import WindowWatcher

logger = logging.getLogger(__name__)

CONFIG_FILE = Path("sigil_config.json")
PORT = 7272
STATIC_DIR = Path(__file__).parent / "static"


# ── Default config schema ─────────────────────────────────────────────────────
#
# Structure: { plugin_name: { gesture_name: [key, key, ...] } }
# This mirrors each plugin's _ACTION_MAP but serialisable as JSON.

DEFAULT_CONFIG: Dict[str, Dict[str, List[str]]] = {
    "VLC Player": {
        "Swipe Right": ["ctrl", "right"],
        "Swipe Left":  ["ctrl", "left"],
        "Swipe Up":    ["ctrl", "up"],
        "Swipe Down":  ["ctrl", "down"],
        "Tap":         ["space"],
        "Grab":        ["m"],
    },
    "VSCode Editor": {
        "Swipe Right": ["ctrl", "pagedown"],
        "Swipe Left":  ["ctrl", "pageup"],
        "Swipe Up":    ["up"],
        "Swipe Down":  ["down"],
        "Pinch":       ["ctrl", "b"],
        "Expand":      ["ctrl", "`"],
        "Tap":         ["ctrl", "shift", "p"],
    },
    "PowerPoint Presentation": {
        "Swipe Right": ["right"],
        "Swipe Left":  ["left"],
        "Expand":      ["f5"],
        "Pinch":       ["escape"],
        "Tap":         ["b"],
    },
    "Web Browser": {
        "Swipe Right": ["ctrl", "tab"],
        "Swipe Left":  ["ctrl", "shift", "tab"],
        "Swipe Up":    ["pageup"],
        "Swipe Down":  ["pagedown"],
        "Tap":         ["ctrl", "r"],
        "Grab":        ["ctrl", "w"],
        "Expand":      ["ctrl", "l"],
    },
    "OBS Studio": {
        "Tap":         ["ctrl", "alt", "r"],
        "Grab":        ["ctrl", "alt", "m"],
        "Swipe Right": ["ctrl", "alt", "right"],
        "Swipe Left":  ["ctrl", "alt", "left"],
        "Expand":      ["ctrl", "alt", "s"],
        "Pinch":       ["ctrl", "alt", "s"],
    },
    "PDF Reader": {
        "Swipe Right": ["right"],
        "Swipe Left":  ["left"],
        "Expand":      ["ctrl", "plus"],
        "Pinch":       ["ctrl", "minus"],
        "Tap":         ["ctrl", "shift", "h"],
    },
}


def load_config() -> Dict[str, Any]:
    """Load config from disk, falling back to defaults if missing or corrupt."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s — using defaults.", CONFIG_FILE, exc)
    return DEFAULT_CONFIG.copy()


def save_config(data: Dict[str, Any]) -> None:
    """Persist config to disk."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Config saved to %s", CONFIG_FILE)


# ── HTTP request handler ──────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for the Sigil config server."""

    # Suppress default request-per-line access log
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, f"Not found: {path.name}")
            return
        mime = "text/html" if path.suffix == ".html" else "text/css"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
        elif path == "/style.css":
            self._send_file(STATIC_DIR / "style.css")
        elif path == "/api/config":
            self._send_json(load_config())
        elif path == "/api/status":
            self._send_json({
                "active_window": WindowWatcher.last_target_title,
                "port": PORT,
            })
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/config":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        save_config(data)
        self._send_json({"status": "saved"})

    def do_OPTIONS(self) -> None:  # noqa: N802
        """CORS pre-flight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ── Public API ────────────────────────────────────────────────────────────────

class ConfigServer:
    """
    Manages the lifecycle of the local config HTTP server.

    Usage::

        server = ConfigServer(port=7272)
        server.start()          # starts daemon thread — call once
        server.open_browser()   # optional: launch browser tab

    The server stops automatically when the Python process exits.
    """

    def __init__(self, port: int = PORT):
        self.port = port
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return  # Already running

        self._httpd = HTTPServer(("127.0.0.1", self.port), _Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
            name="SigilConfigServer",
        )
        self._thread.start()
        logger.info("Sigil config server running at http://localhost:%d", self.port)

    def open_browser(self) -> None:
        """Open the config page in the default browser."""
        webbrowser.open(f"http://localhost:{self.port}/")

    def stop(self) -> None:
        """Gracefully shut down the server (optional — daemon thread exits anyway)."""
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"
