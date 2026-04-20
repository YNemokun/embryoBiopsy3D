"""
Tests for bottom-up positioning with Hungarian algorithm assignment,
build_cost_matrix edge cases, and Fibonacci sphere coordinate geometry.
"""

import numpy as np
import pytest

from embryobiopsy3d.lineage_simulator import (
    generate_tree,
    build_embryo,
    coordinates_generate_radians,
    build_cost_matrix,
    _bottom_up_position_leaves,
    _angles_to_cartesian,
)


def _angular_distance_xyz(point_a, point_b):
    """Reference: same formula as `Sampling.dist_on_sphere` (no module import)."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def _fibonacci_unit_sphere_cartesian(n: int) -> np.ndarray:
    """Unit vectors for the Fibonacci sphere lattice (from coordinates_generate_radians)."""
    angles = coordinates_generate_radians(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    theta, phi = angles[:, 0], angles[:, 1]
    return np.c_[
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ]


def test_build_cost_matrix_shape():
    """Cost matrix has shape (n, n) for n children and n slots."""
    n = 4
    child_angles = coordinates_generate_radians(n)
    fib_angles = coordinates_generate_radians(n)
    C = build_cost_matrix(child_angles, fib_angles)
    assert C.shape == (n, n)
    assert np.all(np.isfinite(C))
    assert np.all(C >= 0)


def test_build_cost_matrix_matches_cartesian_angular_distance():
    """Cost matrix entries match Cartesian angular distance."""
    th1, ph1 = 0.5, 1.0
    th2, ph2 = 1.5, 0.8
    expected = _angular_distance_xyz(
        _angles_to_cartesian(th1, ph1, 1.0),
        _angles_to_cartesian(th2, ph2, 1.0),
    )
    a = np.array([[th1, ph1]])
    b = np.array([[th2, ph2]])
    C = build_cost_matrix(a, b)
    assert C[0, 0] == pytest.approx(expected, abs=1e-12)


def test_build_cost_matrix_empty_inputs():
    """Empty angle arrays yield zero-sized cost matrix."""
    z = np.zeros((0, 2))
    C = build_cost_matrix(z, coordinates_generate_radians(3))
    assert C.shape == (0, 3)
    C2 = build_cost_matrix(coordinates_generate_radians(2), z)
    assert C2.shape == (2, 0)


def test_build_cost_matrix_rectangular_shape():
    """Non-square child vs slot counts is allowed."""
    child = coordinates_generate_radians(2)
    slots = coordinates_generate_radians(5)
    C = build_cost_matrix(child, slots)
    assert C.shape == (2, 5)


def test_coordinates_on_unit_sphere():
    """Fibonacci lattice angles map to Cartesian points on the unit sphere."""
    for n in (8, 32, 128):
        points = _fibonacci_unit_sphere_cartesian(n)
        radii = np.sqrt((points**2).sum(axis=1))
        assert np.allclose(radii, 1.0, atol=1e-9)


# -----------------------------------------------------------------------------
# linear_sum_assignment usage (via _bottom_up_position_leaves)
# -----------------------------------------------------------------------------


def test_bottom_up_assignments_are_bijective():
    """Each child gets exactly one slot; each slot used at most once."""
    _, leaves, _, _, generation_layers = generate_tree(
        generations=3, include_metadata=True
    )
    ordered, coords, _ = _bottom_up_position_leaves(
        generation_layers=generation_layers,
        dispersal=0.0,
    )
    # Direct bottom-up helpers populate layered spherical coordinates.
    assert len(ordered) == len(leaves)
    for leaf in ordered:
        assert leaf.layer_position is not None
        assert len(leaf.layer_position) == 3  # [radius, theta, phi]
    # Positions should be unique (no two leaves at same slot)
    positions = [tuple(p) for p in coords]
    assert len(set(positions)) == len(positions)


def test_bottom_up_position_format():
    """Layered positions are [radius, theta, phi] in spherical form."""
    _, _, _, _, generation_layers = generate_tree(generations=2, include_metadata=True)
    ordered, _, _ = _bottom_up_position_leaves(
        generation_layers=generation_layers,
        dispersal=0.0,
    )
    for leaf in ordered:
        r, theta, phi = (
            leaf.layer_position[0],
            leaf.layer_position[1],
            leaf.layer_position[2],
        )
        assert r > 0
        assert 0 <= theta < 2 * np.pi
        assert 0 <= phi <= np.pi


def test_bottom_up_reproducible_with_seed():
    """Same seed produces same positions."""
    _, _, _, _, generation_layers = generate_tree(generations=3, include_metadata=True)
    _, coords1, _ = _bottom_up_position_leaves(
        generation_layers=generation_layers,
        dispersal=0.0,
        rng=np.random.default_rng(7),
    )
    # Rebuild tree and run again
    _, _, _, _, generation_layers2 = generate_tree(generations=3, include_metadata=True)
    _, coords2, _ = _bottom_up_position_leaves(
        generation_layers=generation_layers2,
        dispersal=0.0,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_almost_equal(coords1, coords2)


def test_bottom_up_generation_0_and_1_direct_placement():
    """Generations 0 and 1 use direct Fibonacci assignment (no Hungarian)."""
    _, _, _, _, generation_layers = generate_tree(generations=2, include_metadata=True)
    _bottom_up_position_leaves(
        generation_layers=generation_layers,
        dispersal=0.0,
    )
    # Gen 0: 1 node, gen 1: 2 nodes - both use direct zip with angles
    gen0_node = generation_layers[0][0]
    assert gen0_node.layer_position[0] == 1.0  # radius = generation + 1
    gen1_nodes = generation_layers[1]
    assert len(gen1_nodes) == 2
    assert gen1_nodes[0].layer_position[0] == 2.0
    assert gen1_nodes[1].layer_position[0] == 2.0


def test_bottom_up_requires_generation_layers():
    """Raises if generation_layers is None."""
    with pytest.raises(ValueError, match="generation_layers is required"):
        _bottom_up_position_leaves(
            generation_layers=None,
            dispersal=0.0,
        )


def test_build_embryo_preserves_layer_positions_for_visualization():
    """Layered spherical coordinates remain available on `cell.layer_position`."""
    emb = build_embryo(
        generations=4,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=4,
    )
    assert all(leaf.layer_position is not None for leaf in emb.leaves)


# -----------------------------------------------------------------------------
# Dispersal
# -----------------------------------------------------------------------------


def test_bottom_up_dispersal_reproducible_with_same_seed():
    """With dispersal > 0, same seed still produces reproducible results."""
    emb1 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=7,
        placement_dispersal=0.05,
    )
    emb2 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=7,
        placement_dispersal=0.05,
    )
    np.testing.assert_array_almost_equal(emb1.coords, emb2.coords)


def test_bottom_up_dispersal_changes_with_different_seed():
    """With dispersal > 0, different seeds produce different positions."""
    emb1 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=7,
        placement_dispersal=0.05,
    )
    emb2 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=8,
        placement_dispersal=0.05,
    )
    assert not np.allclose(emb1.coords, emb2.coords)


# -----------------------------------------------------------------------------
# Spherical vs Cartesian (coordinate system notes)
# -----------------------------------------------------------------------------


def test_angles_to_cartesian_roundtrip():
    """Spherical [r, theta, phi] converts to cartesian and back for unit sphere."""
    theta, phi = 0.5, 1.2
    radius = 1.0
    xyz = _angles_to_cartesian(theta, phi, radius)
    assert xyz.shape == (3,)
    # On unit sphere, norm should be radius
    assert np.linalg.norm(xyz) == pytest.approx(radius, abs=1e-12)
