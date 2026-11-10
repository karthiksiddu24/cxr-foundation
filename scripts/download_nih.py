#!/usr/bin/env python3
"""Download NIH ChestX-ray14 (images + official metadata) with resume and SHA-256 checks.

The original NIH Box links (batch_download_zips.py) return HTTP 404 as of 2026-11. We use a
HuggingFace mirror pinned to a fixed commit; every archive is verified against the SHA-256 that
the mirror's Git LFS pointer publishes, so a corrupted or altered file is rejected.

Stdlib only, so it can run before the project environment exists.

Usage:
    python scripts/download_nih.py --out /data/cxr/raw/nih [--workers 4] [--keep-zips]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "alkzar90/NIH-Chest-X-ray-dataset"
REVISION = "36778e3b0e4f4b4fad31d1728d6190f3eda5b543"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data"

METADATA = [
    "Data_Entry_2017_v2020.csv",
    "BBox_List_2017.csv",
    "train_val_list.txt",
    "test_list.txt",
]

# (filename, size_bytes, sha256) from the mirror's LFS pointers at REVISION.
ARCHIVES: list[tuple[str, int, str]] = [
    (
        "images_001.zip",
        2010553014,
        "9d88bf7cd51536d6ad75661d8fe889ca344df8537edb7f0abb291c1b6653caac",
    ),
    (
        "images_002.zip",
        3952086082,
        "ba53b938b22d600afd0916f3368a9862bf352b82587fce695802e59233ab8d4e",
    ),
    (
        "images_003.zip",
        3933374676,
        "d8eb39e2fddc42ca7181c83ebecc9ffc4ea502bb12e33cc65753818e43e9c976",
    ),
    (
        "images_004.zip",
        3843097813,
        "4067730a77adade8ebbea83fd28f882bfabe7cca8fdd948978b0de8cb5f51329",
    ),
    (
        "images_005.zip",
        3939677731,
        "c130b87084b1a122b66d45043e32276f1aa29ba71aa710f35ee78ca1158a0569",
    ),
    (
        "images_006.zip",
        3985815542,
        "f8195557de196c6657a34a13c4c73522ff271386b63ff2f3abe7112e5b7da679",
    ),
    (
        "images_007.zip",
        4015817466,
        "337a3a66c0f4d2163349a4ea1f4f02b2f9313b52519f68ec1c969627b03ef36c",
    ),
    (
        "images_008.zip",
        4017866318,
        "70357d1a54bb4d39f43b34fdaa5a9b4d8fbac5035ef049ca687b0a981c07ab5c",
    ),
    (
        "images_009.zip",
        4110841707,
        "bc83a6c194bd4fdd0fbdf99c3799a2222f9f3337a1bf31871184be7b402ab246",
    ),
    (
        "images_010.zip",
        4181069714,
        "3b329df9b489d5b49862d1a65fbf976b43f6746287e1a8b8de432ef9145277e2",
    ),
    (
        "images_011.zip",
        4186590070,
        "a3a79894a83c225d9d81ded5d2b4a81a245968fc3cfc2e881b08d2053975a564",
    ),
    (
        "images_012.zip",
        2913827342,
        "ab8f1b7c599e7c6eaaa2ee8cb14a8ceb9849e994690889c87dd11be88757a0f9",
    ),
]

CHUNK = 8 * 1024 * 1024
USER_AGENT = "cxr-foundation-downloader/1.0"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_size: int | None, retries: int = 8) -> None:
    """Download url to dest, resuming from a .part file via HTTP Range."""
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if expected_size is not None and have == expected_size:
            break
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:
                    have = 0  # server ignored Range; restart
                mode = "ab" if have else "wb"
                with part.open(mode) as f:
                    shutil.copyfileobj(resp, f, CHUNK)
            if expected_size is None or part.stat().st_size == expected_size:
                break
        except Exception as exc:  # network errors: back off and resume
            log(f"{dest.name}: attempt {attempt} failed ({exc!r}); retrying")
            time.sleep(min(60, 2**attempt))
    else:
        raise RuntimeError(f"{dest.name}: failed after {retries} attempts")
    part.rename(dest)


def get_archive(out: Path, name: str, size: int, sha: str, keep_zip: bool) -> str:
    images_dir = out / "images"
    marker = out / ".extracted" / f"{name}.ok"
    if marker.exists():
        return f"{name}: already extracted"
    zpath = out / "zips" / name
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists():
        log(f"{name}: downloading {size / 1e9:.2f} GB")
        fetch(f"{BASE}/images/{name}", zpath, size)
    log(f"{name}: verifying sha256")
    got = sha256_of(zpath)
    if got != sha:
        zpath.unlink()
        raise RuntimeError(f"{name}: sha256 mismatch (got {got}); file deleted, rerun to retry")
    log(f"{name}: extracting")
    images_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            # Some mirror archives carry macOS AppleDouble files ("._x.png", "__MACOSX/").
            if (
                info.is_dir()
                or not base.lower().endswith(".png")
                or base.startswith("._")
                or "__MACOSX" in info.filename
            ):
                continue
            target = images_dir / base  # flatten, ignore archive paths
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            n += 1
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{n}\n")
    if not keep_zip:
        zpath.unlink()
    return f"{name}: ok, {n} images"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-zips", action="store_true", help="keep archives after extraction")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    for name in METADATA:
        dest = out / name
        if not dest.exists():
            log(f"metadata: {name}")
            fetch(f"{BASE}/{name}", dest, None)
    (out / "SOURCE.txt").write_text(f"repo={REPO}\nrevision={REVISION}\n")

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(get_archive, out, n, s, h, args.keep_zips): n for n, s, h in ARCHIVES}
        for fut in as_completed(futs):
            try:
                log(fut.result())
            except Exception as exc:
                failures += 1
                log(f"ERROR {futs[fut]}: {exc}")
    total = sum(1 for p in (out / "images").glob("*.png") if not p.name.startswith("._"))
    log(f"done: {total} PNGs in {out / 'images'}; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
