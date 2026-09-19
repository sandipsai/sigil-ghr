"""
Web Browser Macro Plugin for SigilGHR.

Supports Chrome, Edge, Firefox, and any Chromium-based browser.
"""

from typing import List, Dict, Tuple
from sigil.core.interfaces import IPlugin
from sigil.plugins.key_injector import send_keys


class BrowserPlugin(IPlugin):
    """
    Macro plugin controlling web browsers (Chrome, Edge, Firefox).

    Window matching: "chrome", "firefox", "edge", "mozilla", "brave".
    Uses standard browser keyboard shortcuts.
    """

    window_keywords: List[str] = ["chrome", "firefox", "edge", "mozilla", "brave", "opera"]

    _ACTION_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "Swipe Right": ("Next Tab",           ("ctrl", "tab")),
        "Swipe Left":  ("Previous Tab",       ("ctrl", "shift", "tab")),
        "Swipe Up":    ("Scroll Up",          ("pageup",)),
        "Swipe Down":  ("Scroll Down",        ("pagedown",)),
        "Tap":         ("Reload Page",        ("ctrl", "r")),
        "Grab":        ("Close Tab",          ("ctrl", "w")),
        "Expand":      ("Focus Address Bar",  ("ctrl", "l")),
    }

    @property
    def name(self) -> str:
        return "Web Browser"

    def get_supported_gestures(self) -> List[str]:
        return list(self._ACTION_MAP.keys())

    def execute(self, gesture_name: str, confidence: float) -> bool:
        if gesture_name not in self._ACTION_MAP:
            return False
        label, keys = self._ACTION_MAP[gesture_name]
        print(f"[{self.name}] {label}  (confidence {confidence:.2f})")
        return send_keys(*keys)
