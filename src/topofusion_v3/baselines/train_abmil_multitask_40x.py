#!/usr/bin/env python
from __future__ import annotations

import sys

from topofusion_v3.baselines.train_baseline_multitask import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "abmil"]
    main()
