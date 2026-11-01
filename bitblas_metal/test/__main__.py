"""Run the smoke or stress suite using Python's standard-library unittest."""

import argparse
import os
from pathlib import Path
import unittest

from ._reporting import ColorTextTestRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("smoke", "stress", "all"), default="smoke", nargs="?")
    parser.add_argument("--device", choices=("cpu", "mps", "all"), default="all")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                        help="auto: color terminal output; always: force colors; never: plain text")
    args = parser.parse_args()
    os.environ["BITBLAS_METAL_TEST_DEVICE"] = args.device
    # Exercise current kernel generation rather than an older disk-cache entry.
    os.environ["TILELANG_DISABLE_CACHE"] = "1"

    import torch

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was explicitly requested, but is unavailable.")

    print(f"Suite: {args.suite}; device: {args.device}; fixed random seeds", flush=True)
    print("MPS runs include both Torch and TileLang; stress checks correctness, not speed.", flush=True)
    test_root = Path(__file__).resolve().parent
    project_root = test_root.parent.parent
    names = ("smoke", "stress") if args.suite == "all" else (args.suite,)
    suite = unittest.TestSuite()
    for name in names:
        suite.addTests(unittest.defaultTestLoader.discover(
            str(test_root / name), pattern="test_*.py", top_level_dir=str(project_root)
        ))
    result = ColorTextTestRunner(verbosity=2, color=args.color).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
