"""Pure NumPy joint-angle and vector geometry utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class HasXY(Protocol):
    """Protocol for landmark-like objects exposing x and y attributes."""

    x: float
    y: float


PointLike = Sequence[float] | np.ndarray | HasXY


def _to_xy(point: PointLike) -> np.ndarray:
    """Convert a point-like object into a 2D float64 NumPy array."""
    if hasattr(point, "x") and hasattr(point, "y"):
        return np.array([float(point.x), float(point.y)], dtype=np.float64)
    arr = np.asarray(point, dtype=np.float64)
    if arr.shape[0] < 2:
        raise ValueError(f"Point must have at least 2 coordinates, got shape {arr.shape}")
    return arr[:2]


def calculate_angle(a: PointLike, b: PointLike, c: PointLike) -> float:
    """Compute the interior angle (in degrees, [0, 180]) at vertex `b` via cosine law."""
    pa = _to_xy(a)
    pb = _to_xy(b)
    pc = _to_xy(c)

    ba = pa - pb
    bc = pc - pb

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine_clipped = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_clipped)))


def calculate_vertical_angle(top: PointLike, bottom: PointLike) -> float:
    """Compute angle (in degrees, [0, 180]) of segment `bottom -> top` from upward vertical.

    In normalized image coordinates, y increases downward, so upward vertical is (0, -1).
    """
    pt_top = _to_xy(top)
    pt_bottom = _to_xy(bottom)
    vec = pt_top - pt_bottom
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return 0.0

    upward = np.array([0.0, -1.0], dtype=np.float64)
    cosine = float(np.dot(vec, upward) / norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def midpoint(a: PointLike, b: PointLike) -> tuple[float, float]:
    """Return the 2D midpoint between `a` and `b`."""
    pa = _to_xy(a)
    pb = _to_xy(b)
    mid = (pa + pb) / 2.0
    return (float(mid[0]), float(mid[1]))


def euclidean_distance(a: PointLike, b: PointLike) -> float:
    """Return the 2D Euclidean distance between `a` and `b`."""
    pa = _to_xy(a)
    pb = _to_xy(b)
    return float(np.linalg.norm(pa - pb))
