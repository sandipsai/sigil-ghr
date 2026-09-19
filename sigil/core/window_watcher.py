"""
WindowWatcher - Zero-dependency Windows active window tracker.

The Problem
-----------
Sigil is a gesture overlay: the user must look at the Sigil webcam window while
performing gestures.  The moment they move their hand, the Sigil OpenCV window
becomes the foreground window — so naively querying GetForegroundWindow() at
dispatch time always returns "sigil ghr" and nothing fires.

The Solution: Last-Target Memory
---------------------------------
WindowWatcher runs a background poll every ~100 ms and remembers both the title
AND the HWND of the most recently focused window that is NOT Sigil itself.

  last_target_title  → used for plugin keyword matching
  last_target_hwnd   → used by key_injector to SetForegroundWindow before
                        injecting keystrokes, so keys land in the right app

Example timeline:
    1. User is typing in Visual Studio Code   → last_target_hwnd = <vscode hwnd>
    2. User looks at camera, Sigil gains focus→ last_target_hwnd unchanged
    3. User performs "Swipe Right" gesture    → key_injector focuses vscode hwnd
    4. ctrl+pagedown fires IN VSCode          ✓

Cross-platform:
    On non-Windows platforms the poller does nothing and all values stay at
    their defaults — plugins still import and test cleanly.
"""

import sys
import threading
import time
from typing import List, Optional

# ── Sigil window title fragments to ignore when tracking ────────────────────
_SIGIL_TITLES: List[str] = [
    "sigil",
    "opencv",
    "gesture control",
    "live gesture",
]


class WindowWatcher:
    """
    Tracks the last user-facing (non-Sigil) window title AND handle (HWND).

    Class attributes (shared, no instantiation needed):
        last_target_title : str  — lowercased title of last non-Sigil window
        last_target_hwnd  : int  — HWND of last non-Sigil window (0 = unknown)
    """

    _IS_WINDOWS: bool = sys.platform == "win32"
    _user32 = None
    _kernel32 = None

    last_target_title: str = ""
    last_target_hwnd: int = 0

    _poll_running: bool = False
    _poll_thread: Optional[threading.Thread] = None
    _poll_interval: float = 0.10   # seconds between polls

    # ── ctypes helpers ────────────────────────────────────────────────────────

    @classmethod
    def _get_user32(cls):
        if cls._user32 is None and cls._IS_WINDOWS:
            import ctypes
            cls._user32 = ctypes.windll.user32
        return cls._user32

    @classmethod
    def _get_kernel32(cls):
        if cls._kernel32 is None and cls._IS_WINDOWS:
            import ctypes
            cls._kernel32 = ctypes.windll.kernel32
        return cls._kernel32

    # ── Raw query ─────────────────────────────────────────────────────────────

    @classmethod
    def _query_foreground(cls):
        """
        Return (hwnd, lowercased_title) of the current foreground window.
        Returns (0, '') on failure or non-Windows.
        """
        if not cls._IS_WINDOWS:
            return 0, ""
        try:
            import ctypes
            user32 = cls._get_user32()
            hwnd: int = user32.GetForegroundWindow()
            if not hwnd:
                return 0, ""
            buf = ctypes.create_unicode_buffer(512)
            user32.GetWindowTextW(hwnd, buf, 512)
            return hwnd, buf.value.lower().strip()
        except Exception:
            return 0, ""

    # Keep the original public method for compatibility with other code
    @classmethod
    def get_active_window_title(cls) -> str:
        """Return lowercased title of the current foreground window."""
        _, title = cls._query_foreground()
        return title

    # ── Self-detection ────────────────────────────────────────────────────────

    @classmethod
    def _is_sigil_window(cls, title: str) -> bool:
        if not title:
            return True
        return any(fragment in title for fragment in _SIGIL_TITLES)

    # ── Background polling ────────────────────────────────────────────────────

    @classmethod
    def _poll_loop(cls) -> None:
        """Sample the foreground window every _poll_interval seconds."""
        while cls._poll_running:
            hwnd, title = cls._query_foreground()
            if hwnd and not cls._is_sigil_window(title):
                cls.last_target_title = title
                cls.last_target_hwnd  = hwnd
            time.sleep(cls._poll_interval)

    @classmethod
    def start_polling(cls) -> None:
        """Start the background tracking thread (idempotent)."""
        if cls._poll_running:
            return
        cls._poll_running = True
        cls._poll_thread = threading.Thread(
            target=cls._poll_loop,
            daemon=True,
            name="SigilWindowWatcher",
        )
        cls._poll_thread.start()

    @classmethod
    def stop_polling(cls) -> None:
        """Stop the background thread (optional)."""
        cls._poll_running = False

    # ── Focus helper ──────────────────────────────────────────────────────────

    @classmethod
    def focus_target_window(cls) -> bool:
        """
        Bring last_target_hwnd to the foreground so keystrokes land there.

        Windows restricts SetForegroundWindow to processes that are already
        in the foreground or have foreground lock.  The AttachThreadInput
        workaround attaches our thread to the current foreground thread so
        we can legally steal focus.

        Returns True if focus was switched (or was already correct).
        """
        hwnd = cls.last_target_hwnd
        if not hwnd or not cls._IS_WINDOWS:
            return False

        try:
            import ctypes
            user32   = cls._get_user32()
            kernel32 = cls._get_kernel32()

            # If the target window is already in front, do nothing
            if user32.GetForegroundWindow() == hwnd:
                return True

            # Attach our thread input to the current foreground thread
            fg_hwnd     = user32.GetForegroundWindow()
            fg_thread   = user32.GetWindowThreadProcessId(fg_hwnd, None)
            our_thread  = kernel32.GetCurrentThreadId()

            attached = False
            if fg_thread and fg_thread != our_thread:
                user32.AttachThreadInput(fg_thread, our_thread, True)
                attached = True

            try:
                # Restore if minimised, bring to top, set foreground
                user32.ShowWindow(hwnd, 9)       # SW_RESTORE = 9
                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.08)                 # let OS complete the switch
                return True
            finally:
                if attached:
                    user32.AttachThreadInput(fg_thread, our_thread, False)

        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("focus_target_window failed: %s", exc)
            return False

    # ── Keyword matching ──────────────────────────────────────────────────────

    @classmethod
    def matches_any_keyword(cls, title: str, keywords: List[str]) -> bool:
        return any(kw in title for kw in keywords)
