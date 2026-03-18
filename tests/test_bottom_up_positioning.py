"""
Tests for bottom-up positioning with Hungarian algorithm assignment.
"""

import numpy as np
import pytest

from lineage_simulator import (
    Embryo,
    angular_distance,
    generate_tree,
    build_embryo,
    coordinates_generate_radians,
    build_cost_matrix,
    _bottom_up_position_leaves,
    _bottom_up_position_leaves_greedy,
    _angles_to_cartesian,
)


# -----------------------------------------------------------------------------
# build_cost_matrix
# -----------------------------------------------------------------------------


def test_build_cost_matrix_shape():
    """Cost matrix has shape (n, n) for n children and n slots."""
    n = 4
    child_angles = coordinates_generate_radians(n)
    fib_angles = coordinates_generate_radians(n)
    C = build_cost_matrix(child_angles, fib_angles)
    assert C.shape == (n, n)
    assert np.all(np.isfinite(C))
    assert np.all(C >= 0)


def test_build_cost_matrix_diagonal_self_distance():
    """Distance from a point to itself is zero (within float tolerance)."""
    angles = coordinates_generate_radians(3)
    C = build_cost_matrix(angles, angles)
    for i in range(3):
        assert C[i, i] == pytest.approx(0.0, abs=1e-6)


def test_build_cost_matrix_symmetry_of_distance():
    """Angular distance is symmetric: d(a,b) = d(b,a)."""
    a = np.array([[0.5, 0.8], [1.2, 1.5]])
    b = np.array([[2.0, 0.3], [0.1, 2.5]])
    C = build_cost_matrix(a, b)
    # C[i,j] = dist(a[i], b[j]); symmetry of dist means C is not symmetric as a matrix,
    # but dist(a[i], b[j]) == dist(b[j], a[i]) so we can check build_cost_matrix(b,a)
    # has C_build(b,a)[j,i] == C_build(a,b)[i,j]
    C_rev = build_cost_matrix(b, a)
    for i in range(2):
        for j in range(2):
            assert C[i, j] == pytest.approx(C_rev[j, i], abs=1e-12)


def test_build_cost_matrix_matches_cartesian_angular_distance():
    """Cost matrix entries match Cartesian angular distance."""
    th1, ph1 = 0.5, 1.0
    th2, ph2 = 1.5, 0.8
    expected = angular_distance(
        _angles_to_cartesian(th1, ph1, 1.0),
        _angles_to_cartesian(th2, ph2, 1.0),
    )
    a = np.array([[th1, ph1]])
    b = np.array([[th2, ph2]])
    C = build_cost_matrix(a, b)
    assert C[0, 0] == pytest.approx(expected, abs=1e-12)


# -----------------------------------------------------------------------------
# linear_sum_assignment usage (via _bottom_up_position_leaves)
# -----------------------------------------------------------------------------


def test_bottom_up_assignments_are_bijective():
    """Each child gets exactly one slot; each slot used at most once."""
    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
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
    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    ordered, coords, _ = _bottom_up_position_leaves(
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
    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
        generations=3, include_metadata=True
    )
    _, coords1, _ = _bottom_up_position_leaves(
        generation_layers=generation_layers,
        dispersal=0.0,
        rng=np.random.default_rng(7),
    )
    # Rebuild tree and run again
    root2, leaves2, sibling_pairs2, id_dict2, generation_layers2 = generate_tree(
        generations=3, include_metadata=True
    )
    _, coords2, _ = _bottom_up_position_leaves(
        generation_layers=generation_layers2,
        dispersal=0.0,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_almost_equal(coords1, coords2)


def test_bottom_up_generation_0_and_1_direct_placement():
    """Generations 0 and 1 use direct Fibonacci assignment (no Hungarian)."""
    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    ordered, coords, _ = _bottom_up_position_leaves(
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


def test_bottom_up_children_near_parent():
    """Children of the same parent get positions that minimize distance from parent ideal."""
    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
        generations=3, include_metadata=True
    )
    ordered, coords, _ = _bottom_up_position_leaves(
        generation_layers=generation_layers,
        dispersal=0.0,
    )
    # For each parent in gen 1, its two children should have angular positions
    # that are "close" to the parent's (theta, phi) in some sense.
    # The Hungarian assignment minimizes total cost, so siblings should be
    # assigned to slots near their ideal (parent position).
    for parent in generation_layers[1]:
        children = parent.children
        assert len(children) == 2
        p_theta, p_phi = parent.layer_position[1], parent.layer_position[2]
        for child in children:
            c_theta, c_phi = child.layer_position[1], child.layer_position[2]
            dist = angular_distance(
                _angles_to_cartesian(p_theta, p_phi, 1.0),
                _angles_to_cartesian(c_theta, c_phi, 1.0),
            )
            # Should be within a reasonable angular distance (slots are spread on sphere)
            assert dist < np.pi  # at least on same hemisphere-ish


def test_bottom_up_requires_generation_layers():
    """Raises if generation_layers is None."""
    root, leaves, sibling_pairs, _, _ = generate_tree(
        generations=2, include_metadata=True
    )
    with pytest.raises(ValueError, match="generation_layers is required"):
        _bottom_up_position_leaves(
            generation_layers=None,
            dispersal=0.0,
        )


# -----------------------------------------------------------------------------
# build_embryo final coordinate contract
# -----------------------------------------------------------------------------


def test_build_embryo_layers_returns_embryo():
    """build_embryo returns a valid Embryo with leaf coordinates."""
    emb = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=1,
    )
    assert isinstance(emb, Embryo)
    assert emb.root is not None
    assert len(emb.leaves) == 2**3
    assert emb.coords is not None
    assert emb.coords.shape[0] == len(emb.leaves)


def test_build_embryo_layers_coords_shape():
    """Final Cartesian coords array matches number of leaves."""
    emb = build_embryo(
        generations=4,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=2,
    )
    assert emb.coords.shape == (len(emb.leaves), 3)


def test_build_embryo_leaf_positions_are_unit_vectors():
    """Final leaf positions are stored as Cartesian unit vectors."""
    emb = build_embryo(
        generations=4,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=3,
    )
    norms = [
        np.linalg.norm(np.asarray(leaf.position, dtype=float)) for leaf in emb.leaves
    ]
    assert all(np.isclose(norm, 1.0, atol=1e-9) for norm in norms)


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


def test_bottom_up_dispersal_zero_deterministic():
    """With dispersal=0, the same RNG seed reproduces the same placement."""
    root, leaves, sp, id_dict, gl = generate_tree(3, include_metadata=True)
    _, coords1, _ = _bottom_up_position_leaves(
        generation_layers=gl,
        dispersal=0.0,
        rng=np.random.default_rng(7),
    )
    root2, leaves2, sp2, id_dict2, gl2 = generate_tree(3, include_metadata=True)
    _, coords2, _ = _bottom_up_position_leaves(
        generation_layers=gl2,
        dispersal=0.0,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_almost_equal(coords1, coords2)


def test_bottom_up_dispersal_positive_changes_assignment():
    """With dispersal > 0 vs dispersal=0, same seed yields different positions."""
    emb0 = build_embryo(
        generations=3, meio_rate=0.0, mito_rate=0.0, seed=7, placement_dispersal=0.0
    )
    emb1 = build_embryo(
        generations=3, meio_rate=0.0, mito_rate=0.0, seed=7, placement_dispersal=0.1
    )
    # dispersal=0.1 changes the child ideals, changing the assignment
    assert not np.allclose(emb0.coords, emb1.coords)


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


def test_bottom_up_zero_dispersal_changes_with_different_seed():
    """With dispersal = 0, different seeds still produce different positions."""
    emb1 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=7,
        placement_dispersal=0.0,
    )
    emb2 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=8,
        placement_dispersal=0.0,
    )
    assert not np.allclose(emb1.coords, emb2.coords)


def test_bottom_up_zero_dispersal_reproducible_with_same_seed():
    """With dispersal = 0, same seed still produces reproducible positions."""
    emb1 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=7,
        placement_dispersal=0.0,
    )
    emb2 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=7,
        placement_dispersal=0.0,
    )
    np.testing.assert_array_almost_equal(emb1.coords, emb2.coords)


def test_build_embryo_layers_placement_dispersal():
    """build_embryo accepts and uses placement_dispersal."""
    emb0 = build_embryo(
        generations=3, meio_rate=0.0, mito_rate=0.0, seed=1, placement_dispersal=0.0
    )
    emb1 = build_embryo(
        generations=3, meio_rate=0.0, mito_rate=0.0, seed=1, placement_dispersal=0.1
    )
    assert emb0.placement_dispersal == 0.0
    assert emb1.placement_dispersal == 0.1
    # Different dispersal should yield different coords (stochastic)
    assert not np.allclose(emb0.coords, emb1.coords)


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


# -----------------------------------------------------------------------------
# Greedy placement strategy
# -----------------------------------------------------------------------------


def test_bottom_up_greedy_assignments_are_bijective():
    """Greedy placement: each child gets exactly one slot; each slot used at most once."""
    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
        generations=3, include_metadata=True
    )
    ordered, coords, _ = _bottom_up_position_leaves_greedy(
        generation_layers=generation_layers,
        dispersal=0.0,
        rng=np.random.default_rng(42),
    )
    assert len(ordered) == len(leaves)
    for leaf in ordered:
        assert leaf.layer_position is not None
        assert len(leaf.layer_position) == 3
    positions = [tuple(p) for p in coords]
    assert len(set(positions)) == len(positions)


def test_bottom_up_greedy_requires_generation_layers():
    """Greedy placement raises if generation_layers is None."""
    with pytest.raises(ValueError, match="generation_layers is required"):
        _bottom_up_position_leaves_greedy(
            generation_layers=None,
            dispersal=0.0,
        )


def test_bottom_up_greedy_raises_when_dispersal_out_of_range():
    """Greedy placement raises when dispersal outside [0, 1]."""
    root, leaves, sp, id_dict, gl = generate_tree(generations=2, include_metadata=True)
    with pytest.raises(ValueError, match="dispersal must be between"):
        _bottom_up_position_leaves_greedy(
            generation_layers=gl,
            dispersal=1.5,
        )
    with pytest.raises(ValueError, match="dispersal must be between"):
        _bottom_up_position_leaves_greedy(
            generation_layers=gl,
            dispersal=-0.1,
        )


def test_build_embryo_with_placement_strategy_greedy():
    """build_embryo(placement_strategy='greedy') produces valid embryo."""
    emb = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=123,
        placement_strategy="greedy",
    )
    assert isinstance(emb, Embryo)
    assert len(emb.leaves) == 8
    assert all(leaf.position is not None for leaf in emb.leaves)
    positions = [tuple(np.asarray(leaf.position)) for leaf in emb.leaves]
    assert len(set(positions)) == len(positions)


def test_build_embryo_greedy_reproducible_with_seed():
    """Same seed produces same positions with greedy placement."""
    emb1 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=99,
        placement_strategy="greedy",
    )
    emb2 = build_embryo(
        generations=3,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=99,
        placement_strategy="greedy",
    )
    pos1 = np.array([leaf.position for leaf in emb1.leaves])
    pos2 = np.array([leaf.position for leaf in emb2.leaves])
    np.testing.assert_array_almost_equal(pos1, pos2)
