# SigilGHR

SigilGHR is a real-time hand gesture recognition engine that maps hand gestures to keyboard shortcuts in desktop applications. It runs entirely on a local webcam with no cloud dependency. The system implements the multi-stream ensemble CNN architecture described in Habib, Yusuf and Moustafa (MDPI Technologies, 2025) and extends it with a practical macro dispatch layer for applications such as VLC, VSCode, PowerPoint, web browsers, OBS Studio, and PDF readers.

## Overview

The system takes a live webcam feed, extracts 3D hand skeleton data via MediaPipe, converts the skeletal trajectory into multi-view 2D spatiotemporal images, classifies the gesture using a convolutional ensemble, and dispatches the corresponding keyboard shortcut to whichever desktop application the user had focused before interacting with Sigil.

```
Webcam -> MediaPipe Skeleton -> Temporal Resampler (T=250 frames)
       -> 2D Spatiotemporal Fusion (6 view angles)
       -> Multi-Stream Ensemble CNN
       -> Gesture Prediction
       -> MacroEngine (debounce + active-window routing)
       -> OS Key Injection -> Target Application
```

The gesture-to-key mappings are fully configurable through a local web interface served at `http://localhost:7272`.

## Academic Reference

> Habib, M., Yusuf, I. and Moustafa, M. (2025). Skeleton-Based Real-Time Hand Gesture Recognition Using Data Fusion and Ensemble Multi-Stream CNN Architecture. *Technologies*, 13(11), 484. MDPI.

The codebase implements Equations 1-4 from that paper: temporal resampling to a fixed T=250 frame window, spatial zero-centering for position invariance, dynamic zoom normalization for scale invariance, and data-level fusion of spatial pose and temporal motion trail information rendered across six virtual camera orientations.

## Supported Gestures

| Gesture | Motion |
|---|---|
| Swipe Right / Left | Horizontal wrist translation |
| Swipe Up / Down | Vertical wrist translation |
| Expand | Fingertips spreading outward |
| Pinch | Fingertips closing inward |
| Tap | Short stationary movement |
| Grab | Fist formation |

## Default Application Bindings

| Application | Swipe Right | Swipe Left | Tap | Expand | Pinch |
|---|---|---|---|---|---|
| VLC | Next chapter | Prev chapter | Play/Pause | Volume up | Volume down |
| VSCode | Next tab | Prev tab | Command palette | Toggle terminal | Toggle sidebar |
| PowerPoint | Next slide | Prev slide | Black screen | Start slideshow | Exit slideshow |
| Browser | Next tab | Prev tab | Reload | Focus address bar | Close tab |
| OBS | Next scene | Prev scene | Toggle recording | Start stream | Stop stream |
| PDF Reader | Next page | Prev page | Fit width | Zoom in | Zoom out |

Bindings are remappable through the config interface with no code changes required.

## Installation

Requires Python 3.10 or later. Tested on Windows 10/11. Active-window routing and key injection are Windows-specific; the core recognition pipeline is cross-platform.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SigilGHR.git
cd SigilGHR

# Install the package and its dependencies
pip install -e .

# Optional: install PyTorch for CNN training support
pip install -e ".[torch]"
```

The install registers a `sigil` command globally in the current Python environment.

## Usage

```bash
# Start the live webcam interface
sigil

# Run a headless pipeline test without a webcam
sigil --synthetic

# Open the configuration page in the browser on startup
sigil --config

# Run tests
python -m unittest discover -s tests -p "test_*.py"
```

### Keyboard shortcuts during a live session

| Key | Action |
|---|---|
| q | Quit |
| r | Reset gesture buffer |
| c | Open configuration page at localhost:7272 |

## Configuration

Press `c` during any session to open the local configuration page. Each plugin's gesture-to-key bindings are displayed in editable fields. Saving writes to `sigil_config.json` in the working directory; the changes take effect immediately without restarting.

## Project Structure

```
sigil/
  core/
    interfaces.py       Abstract base classes (SOLID)
    camera.py           Hardware video capture wrapper
    skeleton.py         MediaPipe hand tracking with hand isolation filters
    temporal.py         T=250 resampler, spatial centering, dynamic zoom (Eq. 1-4)
    fusion.py           2D spatiotemporal image renderer across 6 view orientations
    models.py           Multi-stream ensemble CNN (PyTorch) and NumPy fallback
    recognizer.py       Gesture classifier with false-positive hardening
    macro_engine.py     Debounced dispatcher with active-window routing
    window_watcher.py   Background OS window tracker (ctypes, no extra dependencies)
    trainer.py          Online fine-tuning with homoscedastic uncertainty loss
  plugins/
    key_injector.py     Focus-then-inject key helper (pyautogui / pynput)
    vlc_plugin.py
    vscode_plugin.py
    powerpoint_plugin.py
    browser_plugin.py
    obs_plugin.py
    pdf_plugin.py
  config/
    server.py           Local HTTP server (Python stdlib, no framework)
    static/
      index.html        Configuration web page
      style.css         Dark theme stylesheet
tests/
pyproject.toml
```

## Adding a Plugin

Plugins are self-contained classes implementing `IPlugin`. Each declares the window title keywords that identify its target application, and a map of gesture names to OS key combinations.

```python
# sigil/plugins/blender_plugin.py
from typing import List, Dict, Tuple
from sigil.core.interfaces import IPlugin
from sigil.plugins.key_injector import send_keys

class BlenderPlugin(IPlugin):
    window_keywords: List[str] = ["blender"]

    _ACTION_MAP: Dict[str, Tuple[str, Tuple[str, ...]]] = {
        "Swipe Right": ("Rotate View Right", ("numpad6",)),
        "Expand":      ("Zoom In",           ("numpadadd",)),
    }

    @property
    def name(self) -> str:
        return "Blender 3D"

    def get_supported_gestures(self) -> List[str]:
        return list(self._ACTION_MAP.keys())

    def execute(self, gesture_name: str, confidence: float) -> bool:
        if gesture_name not in self._ACTION_MAP:
            return False
        label, keys = self._ACTION_MAP[gesture_name]
        print(f"[{self.name}] {label}")
        return send_keys(*keys)
```

Register the plugin by adding it to the `plugins` list in `sigil/main.py`.

## License

MIT
