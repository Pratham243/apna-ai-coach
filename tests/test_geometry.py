"""Unit tests for NumPy joint-angle and vector geometry utilities."""

import numpy as np
import pytest

from src.vision.geometry import (
    calculate_angle,
    calculate_vertical_angle,
    euclidean_distance,
    midpoint,
)
from src.vision.pose import LandmarkPoint


def test_calculate_angle_right_angle():
    a = (0.0, 1.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    angle = calculate_angle(a, b, c)
    assert pytest.approx(angle, abs=1e-2) == 90.0


def test_calculate_angle_straight_line():
    a = (-1.0, 0.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    angle = calculate_angle(a, b, c)
    assert pytest.approx(angle, abs=1e-2) == 180.0


def test_calculate_angle_acute():
    a = (1.0, 1.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    angle = calculate_angle(a, b, c)
    assert pytest.approx(angle, abs=1e-2) == 45.0


def test_calculate_angle_with_landmarks():
    a = LandmarkPoint(x=0.0, y=1.0)
    b = LandmarkPoint(x=0.0, y=0.0)
    c = LandmarkPoint(x=1.0, y=0.0)
    angle = calculate_angle(a, b, c)
    assert pytest.approx(angle, abs=1e-2) == 90.0


def test_calculate_angle_degenerate_points():
    a = (0.0, 0.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert calculate_angle(a, b, c) == 0.0


def test_calculate_vertical_angle():
    # In normalized image coords, upward is (0, -1)
    bottom = LandmarkPoint(x=0.5, y=0.8)
    top = LandmarkPoint(x=0.5, y=0.2)  # perfectly vertical
    angle = calculate_vertical_angle(top, bottom)
    assert pytest.approx(angle, abs=1e-2) == 0.0

    # 45 degrees lean to the right
    top_leaning = LandmarkPoint(x=0.8, y=0.5)
    bottom_center = LandmarkPoint(x=0.5, y=0.8)
    angle_lean = calculate_vertical_angle(top_leaning, bottom_center)
    assert pytest.approx(angle_lean, abs=1.0) == 45.0


def test_midpoint_and_distance():
    p1 = LandmarkPoint(x=0.0, y=0.0)
    p2 = LandmarkPoint(x=2.0, y=2.0)
    mid = midpoint(p1, p2)
    dist = euclidean_distance(p1, p2)
    assert pytest.approx(mid[0], abs=1e-4) == 1.0
    assert pytest.approx(mid[1], abs=1e-4) == 1.0
    assert pytest.approx(dist, abs=1e-4) == np.sqrt(8.0)
