# Run lineage + mapping + sampling checks
import math
import numpy as np

from embryobiopsy3d.biopsy import Sampling
from embryobiopsy3d.lineage_simulator import (
    Cell,
    Embryo,
    cell_division,
    apply_error_rates,
    reset_flags,
    generate_tree,
    coordinates_generate,
    build_embryo,
)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def make_cell(pos):
    c = Cell(None, 0)
    c.position = unit(pos).tolist()
    return c


def make_simple_leaves():
    return [
        make_cell(p)
        for p in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1])
    ]


def _unit_vectors(points):
    pts = np.asarray(points, float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return pts / norms


def _rounded_point_set(points, decimals=8):
    return {tuple(np.round(np.asarray(p, float), decimals)) for p in points}


def test_distance_basics():
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    a, b = s.leaves[0], s.leaves[3]
    da = s.dist_on_sphere(np.asarray(a.position), np.asarray(a.position))
    db = s.dist_on_sphere(np.asarray(a.position), np.asarray(b.position))
    assert math.isclose(da, 0.0, abs_tol=1e-12)
    assert math.isclose(db, math.pi, abs_tol=1e-12)


def test_cell_division_counts():
    np.random.seed(0)
    root = Cell()
    # One generation yields two children
    _, new_leaves, sibs = cell_division(root, generations=1)
    assert len(new_leaves) == 2
    assert all(ch.parent is root for ch in new_leaves)
    assert len(sibs) == 1


def test_generate_tree_sizes():
    gens = 4
    root, leaves, siblings = generate_tree(generations=gens)
    assert len(leaves) == 2**gens
    assert len(siblings) == len(leaves) // 2
    assert all(leaf.generation == gens for leaf in leaves)


def test_apply_error_rates_reproducible_with_seed():
    gens = 3
    root, leaves, _ = generate_tree(generations=gens)
    flags1 = [leaf.is_aneuploid for leaf in leaves]
    mutated1 = apply_error_rates(
        root, meio_rate=0.5, mito_rate=0.5, rng=np.random.default_rng(99)
    )
    after1 = [leaf.is_aneuploid for leaf in leaves]
    reset_flags(mutated1)
    reset_after1 = [leaf.is_aneuploid for leaf in leaves]

    # rebuild and reapply with same seeds to confirm determinism
    root2, leaves2, _ = generate_tree(generations=gens)
    mutated2 = apply_error_rates(
        root2, meio_rate=0.5, mito_rate=0.5, rng=np.random.default_rng(99)
    )
    after2 = [leaf.is_aneuploid for leaf in leaves2]
    reset_flags(mutated2)
    reset_after2 = [leaf.is_aneuploid for leaf in leaves2]

    assert flags1 == [False] * len(leaves)  # no errors initially
    assert after1 == after2  # same RNG => same pattern
    assert reset_after1 == flags1
    assert reset_after2 == flags1


def test_coordinates_on_unit_sphere():
    for n in (8, 32, 128):
        P = coordinates_generate(n)
        r = np.sqrt((P**2).sum(axis=1))
        assert np.allclose(r, 1.0, atol=1e-9)


def test_build_embryo_positions():
    emb = build_embryo(generations=5, meio_rate=0.0, mito_rate=0.0, seed=0)
    leaves = emb.leaves
    # every leaf has a position and lies on sphere
    for leaf in leaves:
        assert leaf.position is not None
        r = np.linalg.norm(np.asarray(leaf.position))
        assert math.isclose(r, 1.0, rel_tol=0, abs_tol=1e-9)
    assert len(leaves) == 2**5


def test_build_embryo_assigns_unique_positions():
    emb = build_embryo(
        generations=3, meio_rate=0.0, mito_rate=0.0, placement_dispersal=0.2, seed=0
    )
    leaves = emb.leaves
    positions = [tuple(np.round(np.asarray(leaf.position), 8)) for leaf in leaves]
    assert all(leaf.position is not None for leaf in leaves)
    assert len(set(positions)) == len(leaves)


def test_build_embryo_respects_supplied_coords():
    coords = _unit_vectors(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.5, 0.5, 0.70710678],
            [-0.5, -0.5, -0.70710678],
        ]
    )
    root, leaves, siblings = generate_tree(3)
    emb = build_embryo(root=root, leaves=leaves, sibling_pairs=siblings, coords=coords)
    positions = np.asarray([leaf.position for leaf in emb.leaves])
    assert _rounded_point_set(positions) == _rounded_point_set(coords)


def test_build_embryo_dispersal_changes_placement():
    emb0 = build_embryo(
        generations=3, meio_rate=0.0, mito_rate=0.0, placement_dispersal=0.0, seed=0
    )
    emb1 = build_embryo(
        generations=3, meio_rate=0.0, mito_rate=0.0, placement_dispersal=1.0, seed=0
    )
    positions0 = np.asarray([leaf.position for leaf in emb0.leaves])
    positions1 = np.asarray([leaf.position for leaf in emb1.leaves])
    assert not np.allclose(positions0, positions1)


def test_build_embryo_packs_state():
    root, leaves, siblings = generate_tree(3)
    coords = coordinates_generate(len(leaves))
    emb = build_embryo(root=root, leaves=leaves, sibling_pairs=siblings, coords=coords)
    assert isinstance(emb, Embryo)
    assert emb.root is root
    assert emb.leaves is leaves
    assert emb.sibling_pairs is siblings
    assert emb.coords is coords
    assert all(leaf.position is not None for leaf in emb.leaves)
    assert emb.generation_rng is None


def test_build_embryo_can_generate_and_place():
    emb = build_embryo(
        generations=3,
        meio_rate=0.1,
        mito_rate=0.2,
        seed=7,
        placement_dispersal=0.5,
        rng=np.random.default_rng(0),
    )
    assert isinstance(emb, Embryo)
    assert emb.root is not None and emb.leaves
    assert len(emb.leaves) == 8  # 3 generations => 8 leaves
    assert len(emb.sibling_pairs) == 4
    assert emb.coords.shape == (8, 3)
    assert all(leaf.position is not None for leaf in emb.leaves)
    assert isinstance(emb.generation_rng, np.random.Generator)


def test_sampling_current_biopsy_behavior():
    # create simple positioned leaves
    leaves = make_simple_leaves()
    s = Sampling(leaves)
    center = leaves[0]
    res = s.current_biopsy(n_cells=3, center_leaf=center)
    sel = res["selected"]
    assert len(sel) == 3 and center in sel
    # distances nondecreasing from center
    d = [
        s.dist_on_sphere(np.asarray(center.position), np.asarray(c.position))
        for c in sel
    ]
    assert all(d[i] <= d[i + 1] for i in range(len(d) - 1))


def test_build_embryo_is_deterministic_with_seed():
    emb1 = build_embryo(
        generations=4, meio_rate=0.0, mito_rate=0.0, placement_dispersal=0.5, seed=11
    )
    emb2 = build_embryo(
        generations=4, meio_rate=0.0, mito_rate=0.0, placement_dispersal=0.5, seed=11
    )
    pos1 = np.asarray([leaf.position for leaf in emb1.leaves], dtype=float)
    pos2 = np.asarray([leaf.position for leaf in emb2.leaves], dtype=float)
    assert np.allclose(pos1, pos2)
