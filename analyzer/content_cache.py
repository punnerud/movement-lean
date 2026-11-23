"""A store addressed by content: the key is a hash of everything the result
depends on, so a result is found again by whoever asks the same question,
whatever the run was called. Used for two-electron integrals (the key is the
basis itself: exponents, positions, contraction) and for lists of
determinants (the key is the integrals they were grown in).

Files live in analyzer/ci_cache/content/. The total size is capped; the
entries read longest ago are removed first.

    MOVEMENT_CACHE=0        switch the store off
    MOVEMENT_CACHE_GB=4     size cap in GB (default 4)
"""
import hashlib
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "ci_cache" / "content"
ENABLED = os.environ.get("MOVEMENT_CACHE", "1") != "0"
CAP_BYTES = float(os.environ.get("MOVEMENT_CACHE_GB", 4)) * 1e9
ENTRY_BYTES = 1.5e9          # larger results are recomputed rather than stored
hits, misses = 0, 0


def key(tag, *parts):
    """Hash of a tag (name and version of the calculation) and its inputs."""
    h = hashlib.sha256(tag.encode())
    for part in parts:
        if isinstance(part, np.ndarray):
            a = np.ascontiguousarray(part)
            h.update(f"|{a.dtype}{a.shape}|".encode())
            h.update(a.tobytes())
        else:
            h.update(f"|{part!r}|".encode())
    return h.hexdigest()[:32]


def path(tag, *parts, suffix=".npy"):
    """Where the entry for these inputs lives (it need not exist)."""
    return ROOT / f"{tag.split(':')[0]}_{key(tag, *parts)}{suffix}"


def _trim():
    files = sorted((p for p in ROOT.glob("*") if p.is_file()), key=lambda p: p.stat().st_atime)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= CAP_BYTES:
            break
        total -= p.stat().st_size
        p.unlink()


def array(tag, parts, compute):
    """The array compute() returns, from the store if these inputs were seen before."""
    global hits, misses
    if not ENABLED:
        return compute()
    file = path(tag, *parts)
    if file.exists():
        try:
            out = np.load(file)
            os.utime(file)
            hits += 1
            return out
        except Exception:
            file.unlink(missing_ok=True)                    # a half-written or damaged entry
    misses += 1
    out = compute()
    if out.nbytes <= ENTRY_BYTES:
        ROOT.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            np.save(f, out)
        os.replace(tmp, file)
        _trim()
    return out
