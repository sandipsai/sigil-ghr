"""
Shared key-injection helper used by all SigilGHR plugins.

Strategy
--------
1. Focus the target window (WindowWatcher.focus_target_window) so keystrokes
   land in the right app, not in Sigil's own OpenCV window.
2. Send the keys via pyautogui first, then pynput as fallback.
3. If neither is installed, log to stdout (development / test mode).

Why the focus step matters
--------------------------
pyautogui.hotkey() and pynput both send keystrokes to whichever window the OS
currently considers active.  When a gesture fires, the Sigil/OpenCV window is
still in focus — so without an explicit SetForegroundWindow() call, all keys
would go to Sigil and the target app would never see them.
"""

from __future__ import annotations
import logging
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Try importing key-injection back-ends (lazy, once) ──────────────────────

try:
    import pyautogui
    pyautogui.FAILSAFE = False   # Disable corner-of-screen failsafe
    pyautogui.PAUSE = 0.0        # No inter-key delay (we control timing)
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False

try:
    from pynput.keyboard import Controller as _KbController, Key as _Key
    _PYNPUT_AVAILABLE = True
    _PYNPUT_SPECIAL: dict = {
        "ctrl":      _Key.ctrl,
        "alt":       _Key.alt,
        "shift":     _Key.shift,
        "win":       _Key.cmd,
        "enter":     _Key.enter,
        "space":     _Key.space,
        "backspace": _Key.backspace,
        "tab":       _Key.tab,
        "escape":    _Key.esc,
        "esc":       _Key.esc,
        "f1":  _Key.f1,  "f2":  _Key.f2,  "f3":  _Key.f3,  "f4":  _Key.f4,
        "f5":  _Key.f5,  "f6":  _Key.f6,  "f7":  _Key.f7,  "f8":  _Key.f8,
        "f9":  _Key.f9,  "f10": _Key.f10, "f11": _Key.f11, "f12": _Key.f12,
        "right":    _Key.right,    "left":  _Key.left,
        "up":       _Key.up,       "down":  _Key.down,
        "pageup":   _Key.page_up,  "pagedown": _Key.page_down,
        "home":     _Key.home,     "end":   _Key.end,
        "delete":   _Key.delete,
        "plus":     "+",           "minus": "-",
        "`":        "`",
    }
    _pynput_kb = _KbController()
except ImportError:
    _PYNPUT_AVAILABLE = False


# ── Focus helper ──────────────────────────────────────────────────────────────

def _focus_target() -> bool:
    """
    Bring the last known non-Sigil window to the foreground.

    Called automatically before every send_keys() so keystrokes land in the
    correct application instead of Sigil's own OpenCV window.

    Returns True if focus was successfully switched (or was already correct).
    On non-Windows or if no target window has been seen yet, returns False
    and keys are sent to whatever is currently active.
    """
    try:
        from sigil.core.window_watcher import WindowWatcher
        return WindowWatcher.focus_target_window()
    except Exception as exc:
        logger.debug("_focus_target failed: %s", exc)
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def send_keys(*keys: str) -> bool:
    """
    Focus the target application then press a key combination.

    Examples::

        send_keys("ctrl", "right")       # Ctrl + Right Arrow  (VLC next chapter)
        send_keys("space")               # Space bar           (play/pause)
        send_keys("ctrl", "shift", "p")  # Ctrl + Shift + P    (VSCode command palette)
        send_keys("f5")                  # F5                  (PowerPoint slideshow)

    Returns True if a back-end successfully fired the keys.
    """
    # Step 1: Move OS focus to the target window so keys go there
    _focus_target()

    # Step 2: Inject the keys
    if _PYAUTOGUI_AVAILABLE:
        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception as exc:
            logger.debug("pyautogui.hotkey(%s) failed: %s — trying pynput", keys, exc)

    if _PYNPUT_AVAILABLE:
        try:
            resolved = [_PYNPUT_SPECIAL.get(k.lower(), k) for k in keys]
            for k in resolved:
                _pynput_kb.press(k)
            for k in reversed(resolved):
                _pynput_kb.release(k)
            return True
        except Exception as exc:
            logger.debug("pynput failed: %s", exc)

    # Fallback: test / CI stub — log what would have fired
    logger.info("[keys] %s (no back-end available — install pyautogui or pynput)", " + ".join(keys))
    return False
