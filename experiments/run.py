"""Experiment CLI runner (Hydra-powered)."""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from fedlora_poison.cli import main

if __name__ == "__main__":
    main()
