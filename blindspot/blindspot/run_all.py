"""Run the whole pipeline after normalise, in order.

    python -m blindspot.run_all                    # data/
    python -m blindspot.run_all --data data/sample
    python -m blindspot.run_all --from engine      # skip earlier steps

Stops at the first step that fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

STEPS = [
    ("plant", "blindspot.plant"),
    ("engine", "blindspot.engine"),
    ("evaluate", "blindspot.evaluate"),
    ("investigate", "blindspot.investigate"),
    ("coverage", "blindspot.checks.coverage"),
    ("calibration", "blindspot.checks.calibration"),
    ("below_line", "blindspot.checks.below_line"),
    ("advisory", "blindspot.checks.advisory"),
    ("report", "blindspot.report"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--from", dest="start", default="plant", choices=[s for s, _ in STEPS])
    args = ap.parse_args()

    names = [s for s, _ in STEPS]
    for name, module in STEPS[names.index(args.start):]:
        cmd = [sys.executable, "-m", module]
        if name == "plant":
            if args.data != "data":
                cmd += ["--base", f"{args.data}/base", "--out", args.data]
        elif args.data != "data":
            cmd += ["--data", args.data]
        print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"\nstopped: {name} failed")
    print("\nall steps finished")


if __name__ == "__main__":
    main()