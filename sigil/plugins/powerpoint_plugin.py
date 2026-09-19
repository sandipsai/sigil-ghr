"""
PowerPoint Macro Plugin for SigilGHR.
"""

from typing import List, Dict, Tuple
from sigil.core.interfaces import IPlugin
from sigil.plugins.key_injector import send_keys


class PowerPointPlugin(IPlugin):
    """
    Macro plugin controlling Microsoft PowerPoint.

    Window matching: "powerpoint".
    Shortcuts work both in editing view and during a slideshow.
    """

    window_keywords: List[str] = ["powerpoint"]

    _ACTION_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "Swipe Right": ("Next Slide",       ("right",)),
        "Swipe Left":  ("Previous Slide",   ("left",)),
        "Expand":      ("Start Slideshow",  ("f5",)),
        "Pinch":       ("Exit Slideshow",   ("escape",)),
        "Tap":         ("Black Screen",     ("b",)),
    }

    @property
    def name(self) -> str:
        return "PowerPoint Presentation"

    def get_supported_gestures(self) -> List[str]:
        return list(self._ACTION_MAP.keys())

    def execute(self, gesture_name: str, confidence: float) -> bool:
        if gesture_name not in self._ACTION_MAP:
            return False
        label, keys = self._ACTION_MAP[gesture_name]
        print(f"[{self.name}] {label}  (confidence {confidence:.2f})")
        return send_keys(*keys)
