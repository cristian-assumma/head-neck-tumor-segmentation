import sys
import os
import numpy as np
import pytest

# Add src to the Python path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluation.metrics import compute_dice_coefficient, calculate_volume_ml


class MockHeader:
    """Mock NIfTI header used to simulate voxel spatial metadata."""

    def __init__(self, zooms):
        self.zooms = zooms

    def get_zooms(self):
        return self.zooms


def test_compute_dice_coefficient_perfect_overlap():
    """Tests that two identical masks yield a Dice Score of 1.0."""
    mask_manual = np.array([[0, 1, 1], [0, 1, 0]])
    mask_auto = np.array([[0, 1, 1], [0, 1, 0]])

    dsc = compute_dice_coefficient(mask_manual, mask_auto, label=1)
    assert dsc == 1.0


def test_compute_dice_coefficient_no_overlap():
    """Tests that two disjoint masks yield a Dice Score of 0.0."""
    mask_manual = np.array([[1, 1, 0], [0, 0, 0]])
    mask_auto = np.array([[0, 0, 0], [0, 1, 1]])

    dsc = compute_dice_coefficient(mask_manual, mask_auto, label=1)
    assert dsc == 0.0


def test_compute_dice_coefficient_empty_masks():
    """Tests the edge case where both prediction and ground truth masks are empty.
    The function should return np.nan to avoid division by zero."""
    mask_manual = np.zeros((3, 3))
    mask_auto = np.zeros((3, 3))

    dsc = compute_dice_coefficient(mask_manual, mask_auto, label=1)
    assert np.isnan(dsc)


def test_calculate_volume_ml():
    """Tests volume computation based on voxel physical dimensions (NIfTI header)."""
    mask = np.array([
        [[1, 1], [0, 0]],
        [[1, 0], [0, 0]]
    ])  # Total: 3 voxels with label 1

    # Voxel size: 2mm x 2mm x 2mm = 8 mm^3 per voxel
    mock_header = MockHeader(zooms=(2.0, 2.0, 2.0))

    # Expected total volume: 3 voxels * 8 mm^3 = 24 mm^3 = 0.024 ml
    volume_ml = calculate_volume_ml(mask, mock_header, label=1)
    assert pytest.approx(volume_ml, 0.0001) == 0.024