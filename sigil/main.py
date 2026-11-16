"""
SigilGHR Main Interactive Application.

Demonstrates live webcam video stream, 3D MediaPipe hand pose estimation,
data-level spatiotemporal multi-view rendering, neural inference, and macro dispatch.

Config UI
---------
Press 'c' during the live session to open the gesture configuration page in
your default browser at http://localhost:7272.  The server starts lazily on
first 'c' press and runs in a daemon thread for the lifetime of the process.
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

from sigil.core.camera import CameraCapture
from sigil.core.skeleton import MediaPipeSkeletonProvider, SyntheticSkeletonProvider
from sigil.core.temporal import TemporalEncoder
from sigil.core.fusion import SpatiotemporalFusionRenderer, DEFAULT_VIEW_ORIENTATIONS, HAND_CONNECTIONS, FINGER_COLORS_BGR
from sigil.core.recognizer import GestureRecognizer
from sigil.core.macro_engine import MacroEngine
from sigil.core.window_watcher import WindowWatcher
from sigil.plugins import VLCPlugin, VSCodePlugin, PowerPointPlugin, BrowserPlugin, OBSPlugin, PDFPlugin

CONFIG_FILE = Path("sigil_config.json")


# ── Config helpers ────────────────────────────────────────────────────────────

def _apply_saved_config(plugins: list) -> None:
    """
    If sigil_config.json exists, override each plugin's _ACTION_MAP with the
    user-saved key tuples.

    The JSON schema is:
        { "Plugin Name": { "Gesture Name": ["key1", "key2", ...] } }
    """
    if not CONFIG_FILE.exists():
        return
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    plugin_map = {p.name: p for p in plugins}
    for plugin_name, gesture_map in cfg.items():
        plugin = plugin_map.get(plugin_name)
        if plugin is None:
            continue
        for gesture_name, keys in gesture_map.items():
            if gesture_name in plugin._ACTION_MAP:
                # Preserve the human label, replace key tuple
                label, _ = plugin._ACTION_MAP[gesture_name]
                plugin._ACTION_MAP[gesture_name] = (label, tuple(keys))


# ── Config server (lazy singleton) ───────────────────────────────────────────

_config_server = None

def _ensure_config_server():
    """Start the config HTTP server on first demand."""
    global _config_server
    if _config_server is None:
        try:
            from sigil.config import ConfigServer
            _config_server = ConfigServer()
            _config_server.start()
            print(f"[Sigil Config] Server running → open http://localhost:7272 in your browser")
        except Exception as exc:
            print(f"[Sigil Config] Could not start server: {exc}")
    _config_server.open_browser()


# ── Webcam interactive mode ───────────────────────────────────────────────────

def run_webcam_interactive():
    """Runs interactive live webcam gesture control interface."""
    print("=" * 75)
    print(" SIGIL GHR Core - Live Interactive Webcam Interface")
    print(" Press 'q' to Quit | 'r' to Reset Buffer | 'c' to Open Config UI")
    print("=" * 75)

    # Start background window tracker BEFORE opening the camera so it has
    # time to record the user's last non-Sigil window (e.g. VSCode/VLC).
    WindowWatcher.start_polling()
    print("[Sigil] Window tracker active — switch to your target app, then return here.")

    camera = CameraCapture(device_id=0, frame_width=640, frame_height=480)
    if not camera.start():
        print("[Error] Could not open webcam device 0. Falling back to synthetic mode.")
        run_synthetic_demo(num_cycles=3)
        return

    skeleton_provider = MediaPipeSkeletonProvider()
    temporal_encoder  = TemporalEncoder(target_frames=250, padding_gamma=0.125)
    fusion_renderer   = SpatiotemporalFusionRenderer()
    gesture_recognizer = GestureRecognizer(num_classes=14, use_heuristics=True)

    plugins = [
        VLCPlugin(), VSCodePlugin(), PowerPointPlugin(),
        BrowserPlugin(), OBSPlugin(), PDFPlugin(),
    ]
    _apply_saved_config(plugins)

    macro_engine = MacroEngine(
        plugins=plugins,
        confidence_threshold=0.75,
        cooldown_seconds=3.0,
        refractory_period=2.5,
        min_frames=20,
        active_window_routing=True,
    )

    selected_vos = DEFAULT_VIEW_ORIENTATIONS[:3]

    gesture_buffer: list = []
    last_status_text = "Ready — Perform Gesture (Swipe, Pinch, Expand…)"
    last_action_text = "None"
    last_target_text = "(focus a target app first)"
    fps_counter  = 0
    t_fps_start  = time.time()
    current_fps  = 0.0
    pip_render   = None

    print("[Sigil] Live video stream active. Wave your hand in front of the camera…")

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                break

            # FPS counter
            fps_counter += 1
            if time.time() - t_fps_start >= 1.0:
                current_fps = fps_counter / (time.time() - t_fps_start)
                fps_counter = 0
                t_fps_start = time.time()

            skeleton_frame = skeleton_provider.process_frame(frame)
            display_frame  = frame.copy()
            h, w, _ = display_frame.shape

            if skeleton_frame and skeleton_frame.is_detected and len(skeleton_frame.joints) == 21:
                gesture_buffer.append(skeleton_frame)

                # Draw skeleton overlay
                px_pts = [(int(j.x * w), int(j.y * h)) for j in skeleton_frame.joints]
                for idx_a, idx_b, finger_grp in HAND_CONNECTIONS:
                    if idx_a < len(px_pts) and idx_b < len(px_pts):
                        color = FINGER_COLORS_BGR.get(finger_grp, (255, 255, 255))
                        cv2.line(display_frame, px_pts[idx_a], px_pts[idx_b], color, 2, cv2.LINE_AA)
                for pt in px_pts:
                    cv2.circle(display_frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

                # Rolling buffer cap
                if len(gesture_buffer) > 40:
                    gesture_buffer.pop(0)

                # Only run inference when we have enough frames
                if len(gesture_buffer) >= 15:
                    try:
                        resampled_seq  = temporal_encoder.encode_sequence(gesture_buffer)
                        view_images    = fusion_renderer.render_views(resampled_seq, selected_vos, image_size=(160, 160))
                        pip_render     = view_images[0]

                        prediction     = gesture_recognizer.predict_sequence(resampled_seq, view_images)
                        last_status_text = f"Gesture: {prediction.gesture_name} ({prediction.confidence * 100:.1f}%)"

                        triggered, should_reset = macro_engine.process_prediction(
                            prediction,
                            raw_frame_count=len(gesture_buffer),
                        )
                        # Update target window display
                        target = WindowWatcher.last_target_title
                        last_target_text = (target[:38] + "…") if len(target) > 38 else (target or "(none)")
                        if triggered:
                            last_action_text = f"Triggered '{prediction.gesture_name}'!"
                        if should_reset:
                            gesture_buffer.clear()
                            skeleton_provider.reset()
                            pip_render = None
                    except Exception:
                        pass
            else:
                if len(gesture_buffer) > 0 and (not skeleton_frame or not skeleton_frame.is_detected):
                    gesture_buffer.clear()

            # ── PiP 2D render ──────────────────────────────────────────────
            if pip_render is not None:
                pip_h, pip_w, _ = pip_render.shape
                display_frame[10:10 + pip_h, 470:470 + pip_w] = pip_render
                cv2.rectangle(display_frame, (470, 10), (470 + pip_w, 10 + pip_h), (0, 255, 255), 2)
                cv2.putText(display_frame, "2D Spatiotemporal Render", (475, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

            # ── HUD overlay ────────────────────────────────────────────────
            cv2.rectangle(display_frame, (0, 0), (460, 118), (20, 20, 20), -1)
            cv2.putText(display_frame, f"SIGIL GHR Live Engine | FPS: {current_fps:.1f}", (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Target: {last_target_text}", (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            cv2.putText(display_frame, f"Status: {last_status_text}", (15, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Action: {last_action_text}", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
            cv2.putText(display_frame, "q=quit  r=reset  c=config", (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 120), 1)

            cv2.imshow("SigilGHR - Live Gesture Control", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                gesture_buffer.clear()
                skeleton_provider.reset()
                last_status_text = "Buffer Reset"
                last_action_text = "None"
                pip_render = None
            elif key == ord('c'):
                _ensure_config_server()

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("[Sigil] Live webcam interface stopped.")


# ── Synthetic demo ────────────────────────────────────────────────────────────

def run_synthetic_demo(num_cycles: int = 3):
    """Executes the SigilGHR pipeline in synthetic test mode."""
    print("=" * 65)
    print(" SIGIL GHR Core - Synthetic Test Mode (Habib et al. 2025)")
    print("=" * 65)

    skeleton_provider  = SyntheticSkeletonProvider(gesture_type="swipe_right")
    temporal_encoder   = TemporalEncoder(target_frames=250, padding_gamma=0.125)
    fusion_renderer    = SpatiotemporalFusionRenderer()
    gesture_recognizer = GestureRecognizer(num_classes=14, use_heuristics=True)

    plugins = [
        VLCPlugin(), VSCodePlugin(), PowerPointPlugin(),
        BrowserPlugin(), OBSPlugin(), PDFPlugin(),
    ]
    macro_engine = MacroEngine(
        plugins=plugins,
        confidence_threshold=0.40,
        cooldown_seconds=0.5,
        active_window_routing=False,  # broadcast all in test mode
    )

    selected_vos = DEFAULT_VIEW_ORIENTATIONS[:3]

    for cycle in range(1, num_cycles + 1):
        print(f"\n--- [Cycle {cycle}/{num_cycles}] Capturing Gesture Sequence ---")
        skeleton_provider.reset()
        raw_frames = []

        for _ in range(40):
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = skeleton_provider.process_frame(dummy_img)
            if frame and frame.is_detected:
                raw_frames.append(frame)

        resampled_seq = temporal_encoder.encode_sequence(raw_frames)
        view_images   = fusion_renderer.render_views(resampled_seq, selected_vos, image_size=(224, 224))
        prediction    = gesture_recognizer.predict_sequence(resampled_seq, view_images)

        print(f"[Sigil] Prediction: '{prediction.gesture_name}' (Confidence: {prediction.confidence * 100:.2f}%)")
        triggered, _ = macro_engine.process_prediction(prediction, raw_frame_count=len(raw_frames))
        print(f"[Sigil] Macro Dispatch: {'SUCCESS' if triggered else 'DEBOUNCED/FILTERED'}")

    print("\n" + "=" * 65)
    print(" SIGIL GHR Pipeline Completed Successfully!")
    print("=" * 65)


# ── CLI entry point ───────────────────────────────────────────────────────────

def cli() -> None:
    """
    Console script entry point registered by pyproject.toml.

    After `pip install -e .` this becomes the `sigil` command::

        sigil                   # live webcam mode
        sigil --synthetic       # headless pipeline test
        sigil --config          # open config page immediately
        sigil --synthetic --cycles 5
    """
    parser = argparse.ArgumentParser(
        prog="sigil",
        description="SigilGHR -- real-time hand gesture macro engine",
    )
    parser.add_argument(
        "--mode",
        choices=["webcam", "synthetic"],
        default="webcam",
        help="webcam: live camera mode (default). synthetic: headless pipeline test.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Shorthand for --mode synthetic (no webcam required).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="Number of gesture cycles to run in synthetic mode (default: 3).",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Start the config server and open http://localhost:7272 in the browser.",
    )
    args = parser.parse_args()

    if args.config:
        _ensure_config_server()

    mode = "synthetic" if args.synthetic else args.mode
    if mode == "webcam":
        run_webcam_interactive()
    else:
        run_synthetic_demo(num_cycles=args.cycles)


if __name__ == "__main__":
    cli()
