#!/usr/bin/env python3
"""Write the server's OpenAPI schema to docs/openapi.json (source of the web app's API types).

The schema depends only on code, not on a loaded model, so no bundle is needed.
CI regenerates it and fails if the committed file or web/src/lib/api/schema.d.ts is stale.
"""

from __future__ import annotations

import json
from pathlib import Path

from cxr_server.app import create_app
from cxr_server.settings import Settings

out = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
app = create_app(Settings(warmup=False))
out.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n")
print(f"wrote {out}")
