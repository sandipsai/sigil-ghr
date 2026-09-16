"""
Macro Engine - Event-driven gesture macro dispatcher with debouncing, refractory
filters, and active-window routing.

Active-window routing design (SRP + OCP):
    When the user performs a gesture, Sigil's own OpenCV window is in focus —
    querying the live foreground window at dispatch time would always return
    Sigil itself and nothing would fire.

    Instead, MacroEngine reads WindowWatcher.last_target_title — the most
    recently focused window that was NOT Sigil (tracked by a 100 ms background
    poll).  This is always the app the user was working in before they turned
    to the camera, which is exactly what should receive the keypress.

    Plugins declare their keywords via `window_keywords`.  MacroEngine matches
    last_target_title against those keywords to pick the right plugin.
"""

import time
from typing import List, Dict, Optional, Tuple

from sigil.core.interfaces import IPlugin, GesturePrediction
from sigil.core.window_watcher import WindowWatcher


class MacroEngine:
    """
    Manages plugin registrations, gesture mapping, debouncing, and refractory filters.

    Parameters
    ----------
    plugins : list[IPlugin], optional
        Plugins to register at construction time.
    confidence_threshold : float
        Minimum prediction confidence required before any dispatch (default 0.75).
    cooldown_seconds : float
        Minimum elapsed time between successive dispatches (default 3.0 s).
    refractory_period : float
        After a Swipe gesture, this period suppresses Expand/Pinch/Tap triggers
        that arise as the hand returns to rest (default 2.5 s).
    min_frames : int
        Minimum number of raw skeleton frames a sequence must contain before
        it is eligible for dispatch.  Prevents very-short transient detections.
    active_window_routing : bool
        When True (default), only dispatch to the plugin whose window_keywords
        match the currently focused OS window.  Set False to broadcast to all
        matching plugins regardless of focus (useful for testing).
    """

    def __init__(
        self,
        plugins: Optional[List[IPlugin]] = None,
        confidence_threshold: float = 0.75,
        cooldown_seconds: float = 3.0,
        refractory_period: float = 2.5,
        min_frames: int = 20,
        active_window_routing: bool = True,
    ):
        self.plugins: Dict[str, IPlugin] = {}
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.refractory_period = refractory_period
        self.min_frames = min_frames
        self.active_window_routing = active_window_routing

        self.last_trigger_time: float = 0.0
        self.last_gesture: str = ""

        if plugins:
            for p in plugins:
                self.register_plugin(p)

    # ── Plugin registry ──────────────────────────────────────────────────────

    def register_plugin(self, plugin: IPlugin) -> None:
        """Register an application macro plugin."""
        self.plugins[plugin.name] = plugin

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin by name."""
        self.plugins.pop(plugin_name, None)

    # ── Active-window helpers ────────────────────────────────────────────────

    def _get_active_plugins(self) -> List[IPlugin]:
        """
        Return only the plugins that match the last non-Sigil window the user
        had focused (WindowWatcher.last_target_title).

        Using last_target_title instead of the live foreground window is the
        fix for the core routing problem: when a gesture fires, Sigil's own
        window is in focus — but we want to target the app the user was using
        before they turned to the camera.

        If active_window_routing is disabled, returns all registered plugins.
        """
        if not self.active_window_routing:
            return list(self.plugins.values())

        # Use the remembered pre-Sigil window, not the live foreground
        target_title = WindowWatcher.last_target_title

        if not target_title:
            # No prior window recorded yet — nothing safe to dispatch to
            return []

        matched: List[IPlugin] = []
        for plugin in self.plugins.values():
            keywords = getattr(plugin, "window_keywords", [])
            if WindowWatcher.matches_any_keyword(target_title, keywords):
                matched.append(plugin)

        return matched

    # ── Main dispatch ────────────────────────────────────────────────────────

    def process_prediction(
        self,
        prediction: GesturePrediction,
        raw_frame_count: int = 999,
    ) -> Tuple[bool, bool]:
        """
        Evaluate prediction, filter noise/cooldown, and trigger plugin action.

        Parameters
        ----------
        prediction : GesturePrediction
            The recognised gesture with confidence.
        raw_frame_count : int
            Number of raw skeleton frames used to build the prediction.
            Predictions built from too few frames are rejected (min_frames gate).

        Returns
        -------
        (executed_any, should_reset_buffer) : Tuple[bool, bool]
        """
        now = time.time()
        time_since_last = now - self.last_trigger_time

        # Gate 1 – Confidence threshold
        if prediction.confidence < self.confidence_threshold:
            return False, False

        # Gate 2 – Minimum frames (prevent very-short transient sequences)
        if raw_frame_count < self.min_frames:
            return False, False

        # Gate 3 – Global cooldown
        if time_since_last < self.cooldown_seconds:
            return False, False

        gesture_name = prediction.gesture_name

        # Gate 4 – Post-Swipe refractory: suppress Expand/Pinch/Tap as hand
        #          returns to rest position
        if time_since_last < self.refractory_period:
            if "Swipe" in self.last_gesture and gesture_name in ("Expand", "Pinch", "Tap"):
                return False, False

        # Resolve active-window plugin list
        active_plugins = self._get_active_plugins()
        if not active_plugins:
            # No matching plugin for the current window — silent drop
            return False, False

        executed_any = False

        for plugin in active_plugins:
            if gesture_name in plugin.get_supported_gestures():
                success = plugin.execute(gesture_name, prediction.confidence)
                if success:
                    executed_any = True

        if executed_any:
            self.last_trigger_time = now
            self.last_gesture = gesture_name
            # Signal to caller: flush trajectory buffer immediately
            return True, True

        return False, False
