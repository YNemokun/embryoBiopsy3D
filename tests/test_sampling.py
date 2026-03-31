# sampling tests for the basic cases
import math
import numpy as np
import pytest

from embryobiopsy3d.biopsy import Sampling
from embryobiopsy3d.lineage_simulator import Cell


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n or 1)


def make_cell(pos):
    c = Cell(None, 0)
    c.position = unit(pos).tolist()
    return c


def make_simple_leaves():
    return [
        make_cell(p)
        for p in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1])
    ]


def make_cells_with_flags(flags):
    """Create positioned cells and set aneuploidy flags to the provided booleans."""
    leaves = make_simple_leaves()[: len(flags)]
    for cell, flag in zip(leaves, flags):
        cell.is_aneuploid = bool(flag)
    return leaves


def deg(rad):
    return rad * 180.0 / math.pi


def test_dist_on_sphere_orthogonal_is_pi_over_two():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert s.dist_on_sphere(a, b) == pytest.approx(math.pi / 2, abs=1e-12)


def test_current_biopsy_includes_center_and_size():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[0]
    n_cells = 3
    res = s.current_biopsy(n_cells=n_cells, center_leaf=center)

    assert len(res["selected"]) == n_cells
    assert center in res["selected"]
    # ensure no duplicates
    assert len({id(c) for c in res["selected"]}) == n_cells


def test_current_biopsy_respects_rng_for_center_choice():
    leaves = make_simple_leaves()
    rng = np.random.default_rng(123)
    s = Sampling(leaves, rng=rng)

    res = s.current_biopsy(n_cells=2)
    center = res["center_leaf"]
    assert center in res["selected"]
    assert len(res["selected"]) == 2


def test_current_biopsy_n_cells_one_returns_center():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[1]

    res = s.current_biopsy(n_cells=1, center_leaf=center)
    assert res["selected"] == [center]


def test_current_biopsy_is_invariant_to_coordinate_scale():
    leaves = make_simple_leaves()
    center = leaves[0]
    baseline = Sampling(leaves).current_biopsy(n_cells=4, center_leaf=center)

    scaled = make_simple_leaves()
    for cell in scaled:
        cell.position = (10.0 * np.asarray(cell.position, dtype=float)).tolist()
    scaled_center = scaled[0]
    scaled_res = Sampling(scaled).current_biopsy(n_cells=4, center_leaf=scaled_center)

    assert len(baseline["selected"]) == len(scaled_res["selected"])
    baseline_positions = {
        tuple(np.round(np.asarray(cell.position, dtype=float), 8))
        for cell in baseline["selected"]
    }
    scaled_positions = {
        tuple(
            np.round(
                np.asarray(cell.position, dtype=float) / np.linalg.norm(cell.position),
                8,
            )
        )
        for cell in scaled_res["selected"]
    }
    assert scaled_positions == baseline_positions


def test_categorize_biopsy_all_euploid():
    leaves = make_cells_with_flags([False, False, False])
    s = Sampling(leaves)
    assert s.categorize_biopsy(leaves) == ("euploid", 0)


def test_categorize_biopsy_all_aneuploid():
    leaves = make_cells_with_flags([True, True, True])
    s = Sampling(leaves)
    assert s.categorize_biopsy(leaves) == ("aneuploid", 3)


def test_categorize_biopsy_mixed_returns_mosaic():
    mixed_configs = [
        [True, False],  # aneuploid then euploid
        [False, True],  # euploid then aneuploid
        [True, False, True],  # alternating, starts aneuploid
        [False, True, False],  # alternating, starts euploid
    ]
    for flags in mixed_configs:
        leaves = make_cells_with_flags(flags)
        s = Sampling(leaves)
        assert s.categorize_biopsy(leaves) == ("mosaic", sum(flags))


def test_categorize_biopsy_empty_raises():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    with pytest.raises(ValueError, match="no leaves"):
        s.categorize_biopsy([])


def test_sampling_raises_when_no_leaf_has_position():
    """__init__ requires at least one leaf with .position (message from Sampling)."""
    c = Cell(None, 0)
    c.position = None
    with pytest.raises(ValueError, match="position"):
        Sampling([c])


def test_sampling_skips_leaves_without_position():
    """Leaves with position None are dropped; remaining cells are used."""
    leaves = make_simple_leaves()[:4]
    leaves[1].position = None
    s = Sampling(leaves)
    assert len(s.leaves) == 3
    assert leaves[1] not in s._index_map


def test_dist_on_sphere_coincident_is_zero():
    s = Sampling(make_simple_leaves())
    v = np.array([3.0, 4.0, 12.0])
    assert s.dist_on_sphere(v, v) == pytest.approx(0.0, abs=1e-12)


def test_dist_on_sphere_antipodal_is_pi():
    s = Sampling(make_simple_leaves())
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-2.0, 0.0, 0.0])
    assert s.dist_on_sphere(a, b) == pytest.approx(math.pi, abs=1e-12)


def test_pairwise_angular_matches_pairwise_dist_on_sphere():
    """Static distance matrix is consistent with dist_on_sphere on unit directions."""
    leaves = make_simple_leaves()[:4]
    s = Sampling(leaves)
    coords = np.array([np.asarray(c.position, float) for c in leaves])
    M = Sampling._pairwise_angular(coords)
    for i in range(len(leaves)):
        for j in range(len(leaves)):
            expected = s.dist_on_sphere(coords[i], coords[j])
            assert M[i, j] == pytest.approx(expected, abs=1e-10)


def test_sorted_neighbors_returns_cached_array_on_second_call():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    first = s._sorted_neighbors(0)
    second = s._sorted_neighbors(0)
    assert second is first


def test_current_biopsy_n_cells_larger_than_pool_returns_all_leaves():
    """order[:n_cells] cannot exceed the number of indexed leaves."""
    leaves = make_simple_leaves()[:4]
    s = Sampling(leaves)
    res = s.current_biopsy(n_cells=100, center_leaf=leaves[0])
    assert len(res["selected"]) == 4
    assert len({id(c) for c in res["selected"]}) == 4
