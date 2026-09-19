"""
VLC Media Player Macro Plugin for SigilGHR.
"""

from typing import List, Dict, Tuple
from sigil.core.interfaces import IPlugin
from sigil.plugins.key_injector import send_keys


class VLCPlugin(IPlugin):
    """
    Macro plugin controlling VLC Media Player.

    Window matching keywords: any window whose title contains "vlc".
    VLC's default keyboard shortcuts are used so no VLC configuration is needed.
    """

    # Keywords used by MacroEngine to identify the active VLC window
    window_keywords: List[str] = ["vlc"]

    # gesture_name -> (human_label, key_args_tuple)
    _ACTION_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "Swipe Right": ("Next Chapter",     ("ctrl", "right")),
        "Swipe Left":  ("Prev Chapter",     ("ctrl", "left")),
        "Swipe Up":    ("Volume +10 %",     ("ctrl", "up")),
        "Swipe Down":  ("Volume -10 %",     ("ctrl", "down")),
        "Tap":         ("Play / Pause",     ("space",)),
        "Grab":        ("Mute",             ("m",)),
    }

    @property
    def name(self) -> str:
        return "VLC Player"

    def get_supported_gestures(self) -> List[str]:
        return list(self._ACTION_MAP.keys())

    def execute(self, gesture_name: str, confidence: float) -> bool:
        if gesture_name not in self._ACTION_MAP:
            return False
        label, keys = self._ACTION_MAP[gesture_name]
        print(f"[{self.name}] {label}  (confidence {confidence:.2f})")
        return send_keys(*keys)
