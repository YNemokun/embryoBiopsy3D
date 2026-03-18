import numpy as np

from embryobiopsy3d.lineage_simulator import (
    Embryo,
    apply_error_rates,
    build_embryo,
    coordinates_generate,
    generate_tree,
    reset_flags,
)


def _unit_vectors(points):
    pts = np.asarray(points, float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return pts / norms


def _rounded_point_set(points, decimals=8):
    return {tuple(np.round(np.asarray(p, float), decimals)) for p in points}


def test_generate_tree_sizes():
    gens = 4
    _, leaves, siblings = generate_tree(generations=gens)
    assert len(leaves) == 2**gens
    assert len(siblings) == len(leaves) // 2
    assert all(leaf.generation == gens for leaf in leaves)


def test_apply_error_rates_reproducible_with_seed():
    root, leaves, _ = generate_tree(generations=3)
    flags1 = [leaf.is_aneuploid for leaf in leaves]
    mutated1 = apply_error_rates(
        root, meio_rate=0.5, mito_rate=0.5, rng=np.random.default_rng(99)
    )
    after1 = [leaf.is_aneuploid for leaf in leaves]
    reset_flags(mutated1)

    # rebuild and reapply with same seeds to confirm determinism
    root2, leaves2, _ = generate_tree(generations=3)
    mutated2 = apply_error_rates(
        root2, meio_rate=0.5, mito_rate=0.5, rng=np.random.default_rng(99)
    )
    after2 = [leaf.is_aneuploid for leaf in leaves2]
    reset_flags(mutated2)

    assert flags1 == [False] * len(leaves)
    assert after1 == after2
    assert [leaf.is_aneuploid for leaf in leaves] == flags1
    assert [leaf.is_aneuploid for leaf in leaves2] == flags1


def test_coordinates_on_unit_sphere():
    for n in (8, 32, 128):
        points = coordinates_generate(n)
        radii = np.sqrt((points**2).sum(axis=1))
        assert np.allclose(radii, 1.0, atol=1e-9)


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


def test_build_embryo_packs_state_and_backfills_metadata():
    root, leaves, siblings = generate_tree(3)
    coords = coordinates_generate(len(leaves))

    emb = build_embryo(root=root, leaves=leaves, sibling_pairs=siblings, coords=coords)

    assert isinstance(emb, Embryo)
    assert emb.root is root
    assert emb.leaves is leaves
    assert emb.sibling_pairs is siblings
    assert np.allclose(emb.coords, coords)
    assert emb.generation_rng is None
    assert emb.id_dict[root.id] is root
    assert emb.generation_layers[0][0] is root


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
