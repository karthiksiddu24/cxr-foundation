#!/usr/bin/env python3
"""Prepare CheXpert (expert-labeled valid/test) and VinDr-CXR (test boxes) if present.

Usage: uv run scripts/prep_external.py [--sizes 224 512]
Run `uv run scripts/check_data.py` first to see what is expected where.
"""

from __future__ import annotations

import argparse

from cxr_ml.data.external import prep_chexpert, prep_vindr
from cxr_ml.paths import DataPaths

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[224, 512])
a = ap.parse_args()
p = DataPaths.from_env()
prep_chexpert(p.chexpert_raw, p.root / "processed", a.sizes)
prep_vindr(p.vindr_raw, p.root / "processed", a.sizes)
