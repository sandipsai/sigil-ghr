"""
SigilGHR Core Launcher Script.
Run this script to start the live interactive camera gesture engine:
    python run_demo.py
Or run headless synthetic mode:
    python run_demo.py --synthetic
"""

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigilGHR Interactive Launcher")
    parser.add_argument("--synthetic", action="store_true", help="Run in synthetic test mode without webcam")
    args = parser.parse_args()

    mode = "synthetic" if args.synthetic else "webcam"

    from sigil.main import run_webcam_interactive, run_synthetic_demo
    if mode == "webcam":
        run_webcam_interactive()
    else:
        run_synthetic_demo(num_cycles=3)
