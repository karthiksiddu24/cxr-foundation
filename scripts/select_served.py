#!/usr/bin/env python3
"""Pick the model to serve by *validation* mean AUROC, export its bundle, and point `current` at it.

    uv run scripts/select_served.py [--dry-run]

Candidates: all evaluated 100%-label runs (Phase 2 baselines and Phase 4 ViT-B fine-tunes).
The test set plays no role in the choice.
"""

from __future__ import annotations

import argparse
import json

from cxr_ml.export.bundle import export_run
from cxr_ml.paths import DataPaths

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--version", default="1.0.0")
a = ap.parse_args()
p = DataPaths.from_env()
cands = {}
for m in p.eval_artifacts.glob("*/metrics.json"):
    run = m.parent.name
    if run.startswith("sup_") or (run.startswith("ft_vitb_") and "_f100_" in run):
        cands[run] = json.loads(m.read_text())["val_mean_auroc"]
if not cands:
    raise SystemExit("no evaluated candidate runs")
for r, v in sorted(cands.items(), key=lambda kv: -kv[1]):
    print(f"  val mean AUROC {v:.4f}  {r}")
best = max(cands, key=lambda r: cands[r])
print(f"selected: {best}")
if not a.dry_run:
    out = export_run(best, p.bundles / best, a.version)
    cur = p.bundles / "current"
    if cur.is_symlink() or cur.exists():
        cur.unlink()
    cur.symlink_to(out.name)
    print(f"current -> {out}")
