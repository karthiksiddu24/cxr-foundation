"""COCO compressed run-length encoding (the format of CheXlocalize segmentation masks).

A mask is flattened in column-major (Fortran) order into alternating runs of 0s and 1s, starting
with 0s. Run lengths are written as variable-length 5-bit groups offset by 48 ('0'); from the
third run on, each value is stored as a difference from the run two positions earlier. This is a
dependency-free port of pycocotools' rleFrString/rleToString.
"""

from __future__ import annotations

import numpy as np


def decode_counts(s: str) -> list[int]:
    counts: list[int] = []
    p = 0
    while p < len(s):
        x = k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def encode_counts(counts: list[int]) -> str:
    out: list[str] = []
    for i, c in enumerate(counts):
        x = c - counts[i - 2] if i > 2 else c
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = not (((ch & 0x10) == 0 and x == 0) or ((ch & 0x10) != 0 and x == -1))
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
    return "".join(out)


def rle_decode(rle: dict[str, object]) -> np.ndarray:
    """{'size': [H, W], 'counts': str} -> bool mask [H, W]."""
    h, w = (int(v) for v in rle["size"])  # type: ignore[attr-defined]
    counts = decode_counts(str(rle["counts"]))
    if sum(counts) != h * w:
        raise ValueError(f"RLE covers {sum(counts)} pixels, expected {h * w}")
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2:
            flat[pos : pos + c] = True
        pos += c
    return flat.reshape((w, h)).T


def rle_encode(mask: np.ndarray) -> dict[str, object]:
    h, w = mask.shape
    flat = mask.T.reshape(-1).astype(bool)
    starts = np.concatenate([[0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [flat.size]])
    counts = [int(n) for n in np.diff(starts)]
    if flat.size and flat[0]:
        counts.insert(0, 0)  # runs always start with zeros
    return {"size": [h, w], "counts": encode_counts(counts)}


def rle_is_empty(rle: dict[str, object]) -> bool:
    return len(decode_counts(str(rle["counts"]))) <= 1
