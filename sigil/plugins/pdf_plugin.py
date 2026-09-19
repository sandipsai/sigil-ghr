"""
PDF Viewer Macro Plugin for SigilGHR.

Targets Acrobat Reader, Foxit, SumatraPDF, and browser-rendered PDFs.
"""

from typing import List, Dict, Tuple
from sigil.core.interfaces import IPlugin
from sigil.plugins.key_injector import send_keys


class PDFPlugin(IPlugin):
    """
    Macro plugin controlling PDF readers.

    Window matching: "adobe", "acrobat", "foxit", "pdf", "sumatra".
    Arrow keys are used for page navigation — universal across all major PDF readers.
    """

    window_keywords: List[str] = ["adobe", "acrobat", "foxit", "pdf", "sumatra"]

    _ACTION_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "Swipe Right": ("Next Page",   ("right",)),
        "Swipe Left":  ("Prev Page",   ("left",)),
        "Expand":      ("Zoom In",     ("ctrl", "plus")),
        "Pinch":       ("Zoom Out",    ("ctrl", "minus")),
        "Tap":         ("Fit Width",   ("ctrl", "shift", "h")),
    }

    @property
    def name(self) -> str:
        return "PDF Reader"

    def get_supported_gestures(self) -> List[str]:
        return list(self._ACTION_MAP.keys())

    def execute(self, gesture_name: str, confidence: float) -> bool:
        if gesture_name not in self._ACTION_MAP:
            return False
        label, keys = self._ACTION_MAP[gesture_name]
        print(f"[{self.name}] {label}  (confidence {confidence:.2f})")
        return send_keys(*keys)
