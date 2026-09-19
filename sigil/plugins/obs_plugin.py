"""
OBS Studio Macro Plugin for SigilGHR.
"""

from typing import List, Dict, Tuple
from sigil.core.interfaces import IPlugin
from sigil.plugins.key_injector import send_keys


class OBSPlugin(IPlugin):
    """
    Macro plugin controlling OBS Studio.

    Window matching: "obs".

    Note: OBS's default global hotkeys may need to be configured inside OBS
    (Tools → Hotkeys) to match the keys sent here.  The defaults below follow
    OBS's built-in shortcut conventions.
    """

    window_keywords: List[str] = ["obs"]

    _ACTION_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "Tap":         ("Toggle Recording",    ("ctrl", "alt", "r")),
        "Grab":        ("Mute Microphone",     ("ctrl", "alt", "m")),
        "Swipe Right": ("Next Scene",          ("ctrl", "alt", "right")),
        "Swipe Left":  ("Previous Scene",      ("ctrl", "alt", "left")),
        "Expand":      ("Start Streaming",     ("ctrl", "alt", "s")),
        "Pinch":       ("Stop Streaming",      ("ctrl", "alt", "s")),
    }

    @property
    def name(self) -> str:
        return "OBS Studio"

    def get_supported_gestures(self) -> List[str]:
        return list(self._ACTION_MAP.keys())

    def execute(self, gesture_name: str, confidence: float) -> bool:
        if gesture_name not in self._ACTION_MAP:
            return False
        label, keys = self._ACTION_MAP[gesture_name]
        print(f"[{self.name}] {label}  (confidence {confidence:.2f})")
        return send_keys(*keys)
