"""Integration tests for rebiopsy using real `Sampling` (via `rebiopsy_single_embryo`)."""

import math
import numpy as np

from embryobiopsy3d import rebiopsy
from embryobiopsy3d.lineage_simulator import (
    Cell,
    build_embryo as ls_build_embryo,
)


def _angular_distance_xyz(point_a, point_b):
    """Reference: same formula as `Sampling.dist_on_sphere`."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def make_cell(pos, is_aneuploid=False):
    c = Cell(None, 0)
    c.position = np.asarray(pos, float) / (np.linalg.norm(pos) or 1.0)
    c.is_aneuploid = is_aneuploid
    return c


def make_leaves():
    # Six roughly evenly spaced points on the sphere
    coords = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    return [make_cell(p) for p in coords]


def make_embryo_with_leaves(leaves):
    """Construct an Embryo via build_embryo using provided leaves."""
    coords = [leaf.position for leaf in leaves]
    root = Cell(None, 0)
    root.children = leaves
    for leaf in leaves:
        leaf.parent = root
        leaf.generation = root.generation + 1
    return rebiopsy.build_embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        coords=coords,
        generations=None,
        meio_rate=None,
        mito_rate=None,
        placement_dispersal=0.0,
        rng=None,
        seed=None,
    )


def test_rebiopsy_matches_on_all_euploid():
    embryo = make_embryo_with_leaves(make_leaves())
    assert rebiopsy.rebiopsy_single_embryo(embryo, distance=0.5, seed=0) is True


def test_rebiopsy_detects_category_mismatch():
    leaves = make_leaves()
    leaves[-1].is_aneuploid = True
    embryo = make_embryo_with_leaves(leaves)
    # With seed=3, first random center is (0,0,1); nearest five cells exclude the
    # opposite-pole aneuploid at (0,0,-1), so standard=euploid and second=aneuploid.
    assert rebiopsy.rebiopsy_single_embryo(embryo, distance=0.0, seed=3) is False


def test_rebiopsy_at_error_rate_returns_fraction(monkeypatch):
    def patched_build_embryo(*args, **kwargs):
        return ls_build_embryo(
            generations=3,
            meio_rate=kwargs.get("meio_rate", 0.0),
            mito_rate=kwargs.get("mito_rate", 0.0),
            placement_dispersal=kwargs.get("placement_dispersal", 0.0),
            seed=123,
        )

    monkeypatch.setattr(rebiopsy, "build_embryo", patched_build_embryo)

    rows = rebiopsy.rebiopsy_at_error_rate(
        p_mito=0.0, p_meio=0.0, dispersal=0.0, distance=0.5, n_trials=5
    )
    assert len(rows) == 5
    assert all("match" in r and "concordance" in r for r in rows)
    assert all(r["concordance"] in (0.0, 1.0) for r in rows)
    assert all(r["p_mito"] == 0.0 and r["p_meio"] == 0.0 for r in rows)


def test_rebiopsy_at_error_rate_reproducible_with_seed():
    """Same seed yields identical trial outcomes (shared RNG across apply_error_rates and biopsies)."""
    kwargs = dict(
        p_mito=0.2,
        p_meio=0.1,
        dispersal=0.3,
        distance=0.5,
        n_trials=12,
        seed=4242,
    )
    rows_a = rebiopsy.rebiopsy_at_error_rate(**kwargs)
    rows_b = rebiopsy.rebiopsy_at_error_rate(**kwargs)
    assert len(rows_a) == len(rows_b)
    for a, b in zip(rows_a, rows_b):
        assert a["match"] == b["match"]
        assert a["standard_category"] == b["standard_category"]
        assert a["second_category"] == b["second_category"]
        assert a["concordance"] == b["concordance"]


def test_rebiopsy_returns_error_metadata_when_no_remaining_cells():
    embryo = make_embryo_with_leaves(make_leaves()[:5])

    meta = rebiopsy.rebiopsy_single_embryo(embryo, distance=0.5, return_metadata=True)

    assert meta["match"] is False
    assert meta["second_center"] is None
    assert meta["second_leaves"] == []
    assert meta["error"] == "no remaining cells for rebiopsy"


def test_distance_param_targets_far_apart_cells():
    embryo = rebiopsy.build_embryo(
        generations=6,
        meio_rate=0.1,
        mito_rate=0.1,
        placement_dispersal=0.5,
        seed=123,
    )
    meta = rebiopsy.rebiopsy_single_embryo(embryo, distance=1.0, return_metadata=True)
    assert meta["actual_distance"] >= math.pi * 0.6  # should be on opposite side-ish


def test_actual_distance_matches_center_distance():
    embryo = rebiopsy.build_embryo(
        generations=5,
        meio_rate=0.1,
        mito_rate=0.1,
        placement_dispersal=0.5,
        seed=456,
    )
    meta = rebiopsy.rebiopsy_single_embryo(
        embryo,
        distance=0.5,
        return_metadata=True,
        seed=123,
    )
    center = np.asarray(meta["standard_center"].position)
    second = np.asarray(meta["second_center"].position)
    expected = _angular_distance_xyz(center, second)
    assert math.isclose(meta["actual_distance"], expected, rel_tol=1e-9, abs_tol=1e-9)
