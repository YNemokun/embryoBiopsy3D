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


class FakeSampling:
    """
    Deterministic sampling stub:
    - includes center in selected
    - uses simple slicing to select cells
    """

    def __init__(self, leaves, rng=None):
        self.leaves = leaves
        self.rng = rng or np.random.default_rng(0)

    def dist_on_sphere(self, a, b):
        return _angular_distance_xyz(a, b)

    def current_biopsy(self, n_cells=5, center_leaf=None):
        center_leaf = center_leaf or self.leaves[0]
        selected = list(self.leaves[: max(1, n_cells)])
        if center_leaf not in selected:
            selected[-1] = center_leaf
        return {"center_leaf": center_leaf, "selected": selected}

    def categorize_biopsy(self, biopsy_leaves):
        aneuploid_count = sum(leaf.is_aneuploid for leaf in biopsy_leaves)
        has_aneu = aneuploid_count > 0
        has_eu = any(not leaf.is_aneuploid for leaf in biopsy_leaves)
        if has_aneu and has_eu:
            return "mosaic", aneuploid_count
        if has_aneu:
            return "aneuploid", aneuploid_count
        return "euploid", 0


def test_rebiopsy_matches_on_all_euploid(monkeypatch):
    monkeypatch.setattr(rebiopsy, "Sampling", FakeSampling)
    embryo = make_embryo_with_leaves(make_leaves())

    assert rebiopsy.rebiopsy_single_embryo(embryo, distance=0.5) is True


def test_rebiopsy_detects_category_mismatch(monkeypatch):
    monkeypatch.setattr(rebiopsy, "Sampling", FakeSampling)
    leaves = make_leaves()
    leaves[-1].is_aneuploid = True
    embryo = make_embryo_with_leaves(leaves)

    assert rebiopsy.rebiopsy_single_embryo(embryo, distance=0.0) is False


def test_rebiopsy_at_error_rate_returns_fraction(monkeypatch):
    monkeypatch.setattr(rebiopsy, "Sampling", FakeSampling)

    # Patch build_embryo to always return a small deterministic embryo
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


def test_rebiopsy_returns_error_metadata_when_no_remaining_cells(monkeypatch):
    monkeypatch.setattr(rebiopsy, "Sampling", FakeSampling)
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


def test_relax_params_allow_forced_fallback(monkeypatch):
    class SingleCellSampling(FakeSampling):
        def current_biopsy(self, n_cells=5, center_leaf=None):
            center_leaf = center_leaf or self.leaves[0]
            return {"center_leaf": center_leaf, "selected": [center_leaf]}

    monkeypatch.setattr(rebiopsy, "Sampling", SingleCellSampling)
    leaves = make_leaves()
    embryo = make_embryo_with_leaves(leaves)

    meta = rebiopsy.rebiopsy_single_embryo(
        embryo,
        distance=1.0,
        return_metadata=True,
        max_attempts=0,
    )
    center = np.asarray(meta["standard_center"].position)
    remaining = [leaf for leaf in leaves if leaf is not meta["standard_center"]]
    dists = [
        _angular_distance_xyz(center, np.asarray(leaf.position)) for leaf in remaining
    ]
    assert math.isclose(meta["actual_distance"], max(dists), rel_tol=1e-9, abs_tol=1e-9)
