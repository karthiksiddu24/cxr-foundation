#!/usr/bin/env python3
"""Print the learning rate with the best *validation* mean AUROC among sweep runs.

Usage: uv run scripts/select_lr.py <run-name-prefix>     e.g. lrsweep_densenet121_imagenet_lr
Only `best_val_mean_auroc` is read; test metrics (if any) are ignored by design.
"""

from __future__ import annotations

import json
import sys

from cxr_ml.paths import DataPaths

prefix = sys.argv[1]
runs = sorted(p for p in DataPaths.from_env().runs.glob(f"{prefix}*") if (p / "DONE").exists())
if not runs:
    raise SystemExit(f"no finished runs match {prefix}*")
scores = {
    p.name.removeprefix(prefix): json.loads((p / "summary.json").read_text())["best_val_mean_auroc"]
    for p in runs
}
best = max(scores, key=lambda k: scores[k])
print(json.dumps({"selected_lr": best, "val_mean_auroc": scores}), file=sys.stderr)
print(best)
