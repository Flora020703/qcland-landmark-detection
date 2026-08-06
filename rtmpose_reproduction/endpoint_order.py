"""Endpoint canonicalisation ("DOD") re-sort, matching HRNet's own
projection-based ordering exactly (see dod_vectors.py for the frozen
prototype vectors this operates on).

Ports the tie-safe projection logic from the audited upstream
lib/datasets/fetal.py (`proj = pts @ d_vec / |d_vec|`, `keep = proj0 <= proj1`)
operating directly in ORIGINAL image pixel coordinates -- this project's
converter runs it once per CSV row, before any augmentation, since the
frozen d_vect is itself computed from un-augmented training coordinates.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def canonical_order(p0: Point, p1: Point,
                     d_vect: tuple[Point, Point]) -> tuple[Point, Point]:
    """Return (p0, p1) reordered so the projection onto d_vect's direction
    is non-decreasing -- identical tie-break rule to HRNet's `keep = proj0
    <= proj1` (ties keep the original order, never swap on exact equality).
    """
    d0, d1 = d_vect
    dvx, dvy = d1[0] - d0[0], d1[1] - d0[1]
    denom = math.hypot(dvx, dvy) + 1e-12

    proj0 = (p0[0] * dvx + p0[1] * dvy) / denom
    proj1 = (p1[0] * dvx + p1[1] * dvy) / denom

    if proj0 <= proj1:
        return p0, p1
    return p1, p0
