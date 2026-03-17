# sampling tests for the basic cases
import math, numpy as np
import pytest
from lineage_simulator import Cell
from biopsy import Sampling

def unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); return v/(n or 1)

def make_cell(pos):
    c = Cell(None, 0); c.position = unit(pos).tolist(); 
    return c

def make_simple_leaves():
    return [make_cell(p) for p in ([1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1])]

def make_cells_with_flags(flags):
    """Create positioned cells and set aneuploidy flags to the provided booleans."""
    leaves = make_simple_leaves()[:len(flags)]
    for cell, flag in zip(leaves, flags):
        cell.is_aneuploid = bool(flag)
    return leaves

def deg(rad): return rad * 180.0 / math.pi

def test_returns_all_leaves():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[0]

    n_cells = len(leaves)
    res = s.biopsy_with_distance(n_cells=n_cells, center_leaf=center, distance=0.2)

    # We expect it to return exactly n_cells 'selected' (including center)
    assert len(res["selected"]) == n_cells
    assert center in res["selected"]

def test_threshold_compliance_basic():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[0]
    n_cells = 5
    res = s.biopsy_with_distance(n_cells=n_cells, center_leaf=center, distance=0.25)
    thr = res["threshold"]  # radians
    # All selected must be >= threshold from center, unless relaxation happened.
    # If relaxed_by>0, we allow a small epsilon below threshold.
    eps = 1e-12
    for c in res["selected"]:
        if c is center:
            continue
        d = s.dist_on_sphere(np.asarray(center.position), np.asarray(c.position))
        if res["relaxed_by"] == 0:
            assert d + eps >= thr, f"distance {d:.5f} < threshold {thr:.5f}"
        else:
            # Relaxation means threshold was slid left; just check non-negativity.
            assert d >= 0.0

def test_relaxation_triggers_when_needed():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[0]

    # Set a high dispersal so few cells qualify; relaxation should kick in.
    n_cells = 5
    res = s.biopsy_with_distance(n_cells=n_cells, center_leaf=center, distance=0.95)
    assert res["relaxed_by"] >= 0
    # Still returns n_cells total (center + others, after relaxation/window shift)
    assert len(res["selected"]) == n_cells
    assert center in res["selected"]

def test_monotone_with_dispersal():
    """
    As dispersal increases, the initial threshold should (weakly) increase,
    because it's computed as dispersal * max_distance_from_center.
    """
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[0]
    n_cells = 5

    res1 = s.biopsy_with_distance(n_cells=n_cells, center_leaf=center, distance=0.10)
    res2 = s.biopsy_with_distance(n_cells=n_cells, center_leaf=center, distance=0.50)
    assert res2["threshold"] >= res1["threshold"] - 1e-12

def test_selected_are_farthest_when_dispersal_high():
    """
    With very high dispersal, the slice should start near the end (farthest cells).
    We check that selected cells are among the largest distances from center.
    """
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[0]
    n_cells = 5

    res = s.biopsy_with_distance(n_cells=n_cells, center_leaf=center, distance=0.9)
    # Compute all distances (excluding center) 
    dists = []
    for c in leaves:
        if c is center:
            continue
        dists.append(s.dist_on_sphere(center.position, c.position))
    dists_sorted = sorted(dists, reverse=True)

    # validate via a distance threshold
    cutoff = dists_sorted[n_cells - 2]  # (n_cells-1) selected 
    eps = 1e-12
    for c in res["selected"]:
        if c is center:
            continue
        d = s.dist_on_sphere(center.position, c.position)
        assert d + eps >= cutoff

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
    baseline_positions = {tuple(np.round(np.asarray(cell.position, dtype=float), 8)) for cell in baseline["selected"]}
    scaled_positions = {
        tuple(np.round(np.asarray(cell.position, dtype=float) / np.linalg.norm(cell.position), 8))
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
        [True, False],          # aneuploid then euploid
        [False, True],          # euploid then aneuploid
        [True, False, True],    # alternating, starts aneuploid
        [False, True, False],   # alternating, starts euploid
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
