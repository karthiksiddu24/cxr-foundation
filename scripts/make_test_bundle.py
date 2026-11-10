#!/usr/bin/env python3
"""Write a tiny, untrained-but-valid model bundle for CI end-to-end tests.

    uv run scripts/make_test_bundle.py models/current

Uses the same hand-built ONNX graph as the server unit tests; needs only `onnx`, not torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "cxr_server" / "tests"))
from server_helpers import write_tiny_bundle

out = write_tiny_bundle(Path(sys.argv[1] if len(sys.argv) > 1 else "models/current"))
print(f"wrote test bundle to {out}")
