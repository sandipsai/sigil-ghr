"""
VSCode Macro Plugin for SigilGHR.
"""

from typing import List, Dict, Tuple
from sigil.core.interfaces import IPlugin
from sigil.plugins.key_injector import send_keys


class VSCodePlugin(IPlugin):
    """
    Macro plugin controlling Visual Studio Code.

    Window matching: "visual studio code", "code", "vscode".
    All shortcuts use VSCode's default key bindings — no VSCode setup needed.
    """

    window_keywords: List[str] = ["visual studio code", " - visual studio code", "code"]

    _ACTION_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "Swipe Right": ("Next Tab",           ("ctrl", "pagedown")),
        "Swipe Left":  ("Previous Tab",       ("ctrl", "pageup")),
        "Swipe Up":    ("Scroll Up",          ("up",)),
        "Swipe Down":  ("Scroll Down",        ("down",)),
        "Pinch":       ("Toggle Side Bar",    ("ctrl", "b")),
        "Expand":      ("Toggle Terminal",    ("ctrl", "`")),
        "Tap":         ("Command Palette",    ("ctrl", "shift", "p")),
    }

    @property
    def name(self) -> str:
        return "VSCode Editor"

    def get_supported_gestures(self) -> List[str]:
        return list(self._ACTION_MAP.keys())

    def execute(self, gesture_name: str, confidence: float) -> bool:
        if gesture_name not in self._ACTION_MAP:
            return False
        label, keys = self._ACTION_MAP[gesture_name]
        print(f"[{self.name}] {label}  (confidence {confidence:.2f})")
        return send_keys(*keys)
