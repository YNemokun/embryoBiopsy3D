"""
Tests for lineage_simulator: Embryo API, lineage distance helper, Cell.set_aneuploid,
and build_embryo / _position_leaves validation.
"""

import numpy as np
import pytest

from embryobiopsy3d.lineage_simulator import (
    Cell,
    Embryo,
    _ensure_rng,
    _ideal_angles_from_parent,
    _initialize_generation_metadata,
    _reflect_phi,
    _wrap_theta,
    apply_error_rates,
    build_cost_matrix,
    build_embryo,
    build_id_dict_and_layers,
    cell_division,
    coordinates_generate_radians,
    generate_tree,
    reset_flags,
)


def _lineage_distance(cell_a, cell_b) -> int:
    """Tree distance between two cells using parent pointers (test helper)."""
    if cell_a is cell_b:
        return 0
    a, b = cell_a, cell_b
    dist = 0
    while a.generation > b.generation:
        a = a.parent
        dist += 1
    while b.generation > a.generation:
        b = b.parent
        dist += 1
    while a is not b:
        if a is None or b is None:
            raise ValueError("Cells do not share a common ancestor.")
        a = a.parent
        b = b.parent
        dist += 2
    if a is None:
        raise ValueError("Cells do not share a common ancestor.")
    return dist


# -----------------------------------------------------------------------------
# Embryo API
# -----------------------------------------------------------------------------


def test_embryo_get_node_by_id_raises_when_id_dict_not_initialized():
    """get_node_by_id raises ValueError when id_dict is not initialized."""
    emb = Embryo(
        root=Cell(),
        leaves=[],
        sibling_pairs=[],
        id_dict=None,
        generation_layers=None,
    )
    with pytest.raises(ValueError, match="id_dict is not initialized"):
        emb.get_node_by_id("any-id")


def test_embryo_get_node_by_id_returns_node_when_found():
    """get_node_by_id returns the node when id exists."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    node_id = leaves[0].id
    found = emb.get_node_by_id(node_id)
    assert found is leaves[0]


def test_embryo_get_node_by_id_returns_none_when_not_found():
    """get_node_by_id returns None when id does not exist."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    assert emb.get_node_by_id("nonexistent-id") is None


def test_embryo_set_aneuploid_by_id_raises_when_node_not_found():
    """set_aneuploid_by_id raises ValueError when node id does not exist."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    with pytest.raises(ValueError, match="Node id not found"):
        emb.set_aneuploid_by_id("nonexistent-id")


def test_embryo_set_aneuploid_by_id_updates_subtree_when_include_subtree_true():
    """set_aneuploid_by_id with include_subtree=True updates node and descendants."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    # Set aneuploid on root - should affect entire tree
    node_id = root.id
    assert emb.mutated_cells is None
    affected = emb.set_aneuploid_by_id(node_id, is_aneuploid=True, include_subtree=True)
    assert root.is_aneuploid
    assert all(leaf.is_aneuploid for leaf in leaves)
    assert len(affected) == 1 + 2 + 4  # root + gen1 + gen2
    assert emb.mutated_cells == affected


def test_embryo_set_aneuploid_by_id_updates_only_cell_when_include_subtree_false():
    """set_aneuploid_by_id with include_subtree=False updates only that cell."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    leaf_id = leaves[0].id
    affected = emb.set_aneuploid_by_id(
        leaf_id, is_aneuploid=True, include_subtree=False
    )
    assert leaves[0].is_aneuploid
    assert len(affected) == 1
    assert emb.mutated_cells == affected
    # Other leaves should remain euploid
    assert sum(1 for leaf in leaves if leaf.is_aneuploid) == 1


def test_embryo_get_node_by_generation_index_raises_when_layers_not_initialized():
    """get_node_by_generation_index raises when generation_layers is not initialized."""
    emb = Embryo(
        root=Cell(),
        leaves=[],
        sibling_pairs=[],
        id_dict={},
        generation_layers=None,
    )
    with pytest.raises(ValueError, match="generation_layers is not initialized"):
        emb.get_node_by_generation_index(0, 0)


def test_embryo_get_node_by_generation_index_raises_for_negative_indices():
    """get_node_by_generation_index raises for negative generation or index."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        emb.get_node_by_generation_index(-1, 0)
    with pytest.raises(ValueError, match="must be non-negative"):
        emb.get_node_by_generation_index(0, -1)


def test_embryo_get_node_by_generation_index_raises_when_out_of_range():
    """get_node_by_generation_index raises when generation or index out of range."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    with pytest.raises(ValueError, match="out of range"):
        emb.get_node_by_generation_index(99, 0)
    with pytest.raises(ValueError, match="out of range"):
        emb.get_node_by_generation_index(0, 99)


def test_embryo_get_node_by_generation_index_returns_node():
    """get_node_by_generation_index returns correct node for valid indices."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    node = emb.get_node_by_generation_index(0, 0)
    assert node is root
    node = emb.get_node_by_generation_index(2, 0)
    assert node is leaves[0]


def test_embryo_set_aneuploid_by_generation_index():
    """set_aneuploid_by_generation_index delegates to set_aneuploid correctly."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    affected = emb.set_aneuploid_by_generation_index(
        2, 0, is_aneuploid=True, include_subtree=False
    )
    assert leaves[0].is_aneuploid
    assert len(affected) == 1
    assert emb.mutated_cells == affected


# -----------------------------------------------------------------------------
# Lineage distance (test-local helper)
# -----------------------------------------------------------------------------


def test_lineage_distance_same_cell_returns_zero():
    """_lineage_distance returns 0 for same cell."""
    root, leaves, _, _, _ = generate_tree(generations=2, include_metadata=True)
    assert _lineage_distance(leaves[0], leaves[0]) == 0
    assert _lineage_distance(root, root) == 0


def test_lineage_distance_parent_child_returns_one():
    """_lineage_distance returns 1 for parent-child pair."""
    root, leaves, _, _, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    parent = generation_layers[1][0]
    child = parent.children[0]
    assert _lineage_distance(parent, child) == 1
    assert _lineage_distance(child, parent) == 1


def test_lineage_distance_siblings_returns_two():
    """_lineage_distance returns 2 for sibling pair."""
    root, leaves, sibling_pairs, _, _ = generate_tree(
        generations=2, include_metadata=True
    )
    sib1, sib2 = sibling_pairs[0]
    assert _lineage_distance(sib1, sib2) == 2


def test_lineage_distance_cousins_returns_four():
    """_lineage_distance returns 4 for cousin pair (same grandparent, different parents)."""
    root, leaves, _, _, generation_layers = generate_tree(
        generations=3, include_metadata=True
    )
    # Gen 1: two children of root
    c1, c2 = generation_layers[1][0], generation_layers[1][1]
    # Gen 2: one child from each
    leaf1 = c1.children[0]
    leaf2 = c2.children[0]
    assert _lineage_distance(leaf1, leaf2) == 4


def test_lineage_distance_raises_for_unrelated_cells():
    """_lineage_distance raises ValueError when cells do not share common ancestor."""
    root1 = Cell(parent=None, generation=0)
    root2 = Cell(parent=None, generation=0)
    with pytest.raises(ValueError, match="do not share a common ancestor"):
        _lineage_distance(root1, root2)


# -----------------------------------------------------------------------------
# Cell.set_aneuploid
# -----------------------------------------------------------------------------


def test_cell_set_aneuploid_include_subtree_true_returns_all_descendants():
    """set_aneuploid with include_subtree=True returns all affected nodes."""
    root, leaves, _, _, _ = generate_tree(generations=2, include_metadata=True)
    parent = root.children[0]
    affected = parent.set_aneuploid(is_aneuploid=True, include_subtree=True)
    assert parent.is_aneuploid
    assert all(c.is_aneuploid for c in parent.children)
    assert len(affected) == 3  # parent + 2 children


def test_cell_set_aneuploid_include_subtree_false_returns_only_self():
    """set_aneuploid with include_subtree=False updates and returns only self."""
    root, leaves, _, _, _ = generate_tree(generations=2, include_metadata=True)
    parent = root.children[0]
    affected = parent.set_aneuploid(is_aneuploid=True, include_subtree=False)
    assert parent.is_aneuploid
    assert not any(c.is_aneuploid for c in parent.children)
    assert affected == [parent]


def test_cell_set_aneuploid_can_set_euploid():
    """set_aneuploid with is_aneuploid=False clears flag."""
    root, leaves, _, _, _ = generate_tree(generations=2, include_metadata=True)
    for leaf in leaves:
        leaf.is_aneuploid = True
    affected = leaves[0].set_aneuploid(is_aneuploid=False, include_subtree=False)
    assert not leaves[0].is_aneuploid
    assert len(affected) == 1


# -----------------------------------------------------------------------------
# build_embryo / _position_leaves validation
# -----------------------------------------------------------------------------


def test_build_embryo_raises_when_missing_required_params():
    """build_embryo raises when neither tree nor (generations, meio_rate, mito_rate) provided."""
    with pytest.raises(ValueError, match="build_embryo needs either"):
        build_embryo()
    with pytest.raises(ValueError, match="build_embryo needs either"):
        build_embryo(generations=3)  # missing meio_rate, mito_rate
    with pytest.raises(ValueError, match="build_embryo needs either"):
        build_embryo(generations=3, meio_rate=0.0)  # missing mito_rate


def test_build_embryo_raises_when_coords_length_mismatch():
    """build_embryo raises when coords length does not match leaves."""
    root, leaves, sibling_pairs = generate_tree(generations=3)
    a = coordinates_generate_radians(4)
    coords = np.c_[
        np.cos(a[:, 0]) * np.sin(a[:, 1]),
        np.sin(a[:, 0]) * np.sin(a[:, 1]),
        np.cos(a[:, 1]),
    ]  # wrong size: 8 leaves, 4 coords
    with pytest.raises(ValueError, match="coords length must match"):
        build_embryo(
            root=root,
            leaves=leaves,
            sibling_pairs=sibling_pairs,
            coords=coords,
        )


def test_build_embryo_raises_when_dispersal_out_of_range():
    """build_embryo raises when placement_dispersal is outside [0, 1]."""
    with pytest.raises(ValueError, match="dispersal must be between"):
        build_embryo(
            generations=3,
            meio_rate=0.0,
            mito_rate=0.0,
            placement_dispersal=1.5,
        )
    with pytest.raises(ValueError, match="dispersal must be between"):
        build_embryo(
            generations=3,
            meio_rate=0.0,
            mito_rate=0.0,
            placement_dispersal=-0.1,
        )


# -----------------------------------------------------------------------------
# _ensure_rng
# -----------------------------------------------------------------------------


def test_ensure_rng_returns_same_generator_when_rng_provided_without_seed():
    """When `rng` is passed and `seed` is None, that instance is returned."""
    rng = np.random.default_rng(42)
    out = _ensure_rng(rng, seed=None)
    assert out is rng


def test_ensure_rng_seed_wins_when_rng_and_seed_both_provided():
    """If both `rng` and `seed` are set, a new `default_rng(seed)` is returned (ignores `rng`)."""
    rng = np.random.default_rng(999)
    out = _ensure_rng(rng, seed=42)
    assert out is not rng
    expected = np.random.default_rng(42).random()
    assert out.random() == expected


def test_ensure_rng_without_rng_uses_seed():
    """Without rng, seed produces a reproducible generator."""
    a = _ensure_rng(None, seed=12345)
    b = _ensure_rng(None, seed=12345)
    assert a.random() == b.random()


def test_ensure_rng_without_rng_and_without_seed_is_unseeded():
    """Without rng and without seed, returns a new default_rng (not equal identity each call)."""
    r1 = _ensure_rng(None, None)
    r2 = _ensure_rng(None, None)
    assert r1 is not r2
    # Two draws are very unlikely to match both by accident for float64
    assert r1.random() != r2.random() or r1.random() != r2.random()


# -----------------------------------------------------------------------------
# Embryo / Cell construction
# -----------------------------------------------------------------------------


def test_cell_initialization_defaults():
    """Cell.__init__ sets lineage fields and default flags."""
    root = Cell(parent=None, generation=0)
    assert root.parent is None
    assert root.generation == 0
    assert root.children == []
    assert root.is_aneuploid is False
    assert root.is_dead is False
    assert root.position is None
    assert root.layer_position is None
    assert isinstance(root.id, str) and len(root.id) > 0

    child = Cell(parent=root, generation=1)
    assert child.parent is root
    assert child.generation == 1


def test_embryo_dataclass_initialization():
    """Embryo holds references and optional fields."""
    root = Cell()
    emb = Embryo(
        root=root,
        leaves=[],
        sibling_pairs=[],
        coords=None,
        placement_dispersal=None,
        generation_rng=None,
        mutated_cells=None,
        id_dict={root.id: root},
        generation_layers=[[root]],
    )
    assert emb.root is root
    assert emb.leaves == []
    assert emb.mutated_cells is None


def test_cell_repr_contains_key_fields():
    """__repr__ exposes id prefix, generation, flags (smoke)."""
    c = Cell(parent=None, generation=0)
    text = repr(c)
    assert "gen=0" in text
    assert "aneuploid=False" in text


# -----------------------------------------------------------------------------
# Lineage helpers: build_id_dict_and_layers, cell_division, generations
# -----------------------------------------------------------------------------


def test_build_id_dict_and_layers_empty_root():
    """None root yields empty dict and layers."""
    d, layers = build_id_dict_and_layers(None)
    assert d == {}
    assert layers == []


def test_cell_division_creates_every_generation_layer_zero_through_n():
    """For `generations` divisions, layers 0..generations exist with 2^k cells at gen k."""
    for n in (1, 2, 4, 5):
        root = Cell(parent=None, generation=0)
        _, leaves, _, id_dict, generation_layers = cell_division(
            root,
            generations=n,
            include_metadata=True,
        )
        assert len(generation_layers) == n + 1
        for k in range(n + 1):
            assert len(generation_layers[k]) == 2**k
        assert all(c.generation == n for c in leaves)
        assert len(leaves) == 2**n
        assert len(id_dict) == sum(2**k for k in range(n + 1))


def test_cell_division_generations_zero_returns_root_only():
    """generations=0 performs no divisions; leaves empty."""
    root = Cell(parent=None, generation=0)
    r, leaves, pairs, id_dict, gl = cell_division(
        root, generations=0, include_metadata=True
    )
    assert r is root
    assert leaves == []
    assert pairs == []
    assert gl[0] == [root]


def test_generate_tree_matches_cell_division_layer_counts():
    """generate_tree is consistent with full binary layer counts."""
    for gens in (0, 1, 3):
        root, leaves, _, id_dict, generation_layers = generate_tree(
            generations=gens, include_metadata=True
        )
        if gens == 0:
            assert leaves == []
            assert generation_layers[0] == [root]
        else:
            assert len(generation_layers) == gens + 1
            assert len(generation_layers[gens]) == 2**gens
            assert len(leaves) == 2**gens


# -----------------------------------------------------------------------------
# apply_error_rates, set_aneuploid returns, reset_flags
# -----------------------------------------------------------------------------


def test_apply_error_rates_returns_list_of_cells():
    """apply_error_rates returns a list of Cell instances (mutated tracking)."""
    root, _, _ = generate_tree(generations=2)
    mutated = apply_error_rates(
        root, meio_rate=0.0, mito_rate=0.0, rng=np.random.default_rng(0)
    )
    assert isinstance(mutated, list)
    assert all(isinstance(c, Cell) for c in mutated)


def test_apply_error_rates_and_set_aneuploid_both_return_affected_lists():
    """Stochastic and manual paths return lists of touched cells."""
    root, leaves, _, _, _ = generate_tree(generations=2, include_metadata=True)
    m1 = apply_error_rates(root, 0.0, 0.0, rng=np.random.default_rng(0))
    m2 = leaves[0].set_aneuploid(True, include_subtree=False)
    assert isinstance(m1, list)
    assert isinstance(m2, list)
    assert m2 == [leaves[0]]


def test_reset_flags_clears_flags_and_list_for_combined_mutations():
    """reset_flags clears aneuploid flags on listed cells and empties the list in place."""
    root, leaves, _, _, _ = generate_tree(generations=2, include_metadata=True)
    rng = np.random.default_rng(0)
    # Force some structure: apply_error_rates may leave root euploid at 0 rates
    mutated = apply_error_rates(root, meio_rate=1.0, mito_rate=0.0, rng=rng)
    manual = root.set_aneuploid(True, include_subtree=True)
    combined = []
    combined.extend(mutated)
    combined.extend(m for m in manual if m not in combined)
    assert any(c.is_aneuploid for c in combined)
    assert reset_flags(combined) is None
    assert combined == []
    assert not any(c.is_aneuploid for c in leaves) and not root.is_aneuploid


# -----------------------------------------------------------------------------
# Angle helpers: wrap, reflect, ideal child angles
# -----------------------------------------------------------------------------


def test_wrap_theta_wraps_to_zero_two_pi():
    """_wrap_theta maps to [0, 2π)."""
    assert _wrap_theta(0.0) == pytest.approx(0.0)
    assert _wrap_theta(2 * np.pi) == pytest.approx(0.0)
    assert _wrap_theta(-np.pi) == pytest.approx(np.pi)
    assert 0 <= _wrap_theta(25.3) < 2 * np.pi


def test_reflect_phi_maps_into_zero_pi():
    """_reflect_phi folds angles into [0, π]."""
    assert _reflect_phi(0.5) == pytest.approx(0.5)
    assert _reflect_phi(np.pi) == pytest.approx(np.pi)
    # Above π reflects across equator
    assert _reflect_phi(np.pi + 0.3) == pytest.approx(np.pi - 0.3)
    # Wrap then reflect: 2π -> 0
    assert _reflect_phi(2 * np.pi) == pytest.approx(0.0)


def test_ideal_angles_from_parent_respects_wrap_and_reflect():
    """Child angles stay in valid ranges via wrap/reflect."""
    th, ph = 1.0, 0.8
    child = _ideal_angles_from_parent(th, ph, alpha=0.4, beta=0.2)
    assert child.shape == (2,)
    assert 0 <= child[0] < 2 * np.pi
    assert 0 <= child[1] <= np.pi


# -----------------------------------------------------------------------------
# build_cost_matrix (additional cases)
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# build_embryo scenarios
# -----------------------------------------------------------------------------


def test_build_embryo_without_tree_generates_and_applies_error_rates():
    """No tree: generations + rates build tree, apply_error_rates fills mutated_cells."""
    emb = build_embryo(
        generations=2,
        meio_rate=0.0,
        mito_rate=0.0,
        seed=1,
    )
    assert emb.mutated_cells == []
    assert emb.generation_rng is not None
    emb2 = build_embryo(
        generations=2,
        meio_rate=1.0,
        mito_rate=0.0,
        seed=2,
    )
    assert isinstance(emb2.mutated_cells, list)
    assert emb2.root.is_aneuploid


def test_build_embryo_with_tree_does_not_call_apply_error_rates():
    """Supplying (root, leaves, sibling_pairs) ignores meio_rate/mito_rate; mutated_cells empty."""
    root, leaves, siblings = generate_tree(generations=2)
    emb = build_embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=siblings,
        meio_rate=1.0,
        mito_rate=1.0,
        seed=99,
    )
    assert emb.mutated_cells == []
    assert not root.is_aneuploid
    emb.set_aneuploid_by_id(root.id, is_aneuploid=True, include_subtree=True)
    assert len(emb.mutated_cells) == 7


def test_build_embryo_with_tree_backfills_metadata_when_missing():
    """Tree path rebuilds id_dict and generation_layers when omitted."""
    root, leaves, siblings, _, _ = generate_tree(generations=2, include_metadata=True)
    emb = build_embryo(root=root, leaves=leaves, sibling_pairs=siblings)
    assert emb.id_dict[root.id] is root
    assert len(emb.generation_layers) == 3


def test_build_embryo_tree_then_manual_apply_error_rates():
    """Callers who need error rates on an existing tree must run apply_error_rates themselves."""
    root, leaves, siblings = generate_tree(generations=2)
    rng = np.random.default_rng(0)
    mutated = apply_error_rates(root, meio_rate=1.0, mito_rate=0.0, rng=rng)
    emb = build_embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=siblings,
        rng=np.random.default_rng(1),
    )
    assert root.is_aneuploid
    assert emb.mutated_cells == []
    reset_flags(mutated)
    assert not root.is_aneuploid


# -----------------------------------------------------------------------------
# _record_affected_cells — is_aneuploid=False path (line 102)
# -----------------------------------------------------------------------------


def test_record_affected_cells_removes_cell_when_clearing_aneuploid_flag():
    """Setting is_aneuploid=False on a cell in mutated_cells removes it from the list."""
    root, leaves, _, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    emb = Embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=[],
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    emb.set_aneuploid_by_id(leaves[0].id, is_aneuploid=True)
    assert leaves[0] in emb.mutated_cells
    # Clear the flag — hits the is_aneuploid=False branch in _record_affected_cells.
    emb.set_aneuploid_by_id(leaves[0].id, is_aneuploid=False)
    assert leaves[0] not in emb.mutated_cells
    assert not leaves[0].is_aneuploid


# -----------------------------------------------------------------------------
# _initialize_generation_metadata — short layer-list extension (line 321)
# -----------------------------------------------------------------------------


def test_initialize_generation_metadata_extends_short_generation_layers():
    """A layer list shorter than generations+1 is extended in place."""
    root = Cell(parent=None, generation=0)
    short_layers = [[root]]  # covers generation 0 only
    _, layers = _initialize_generation_metadata(root, 3, generation_layers=short_layers)
    assert len(layers) >= 4  # must accommodate generations 0 through 3


# -----------------------------------------------------------------------------
# cell_division edge cases (lines 369, 402)
# -----------------------------------------------------------------------------


def test_cell_division_zero_generations_with_metadata_returns_empty_leaves():
    """generations=0 with include_metadata=True returns 5-tuple with no leaves."""
    root = Cell(parent=None, generation=0)
    result = cell_division(root, generations=0, include_metadata=True)
    assert len(result) == 5
    _, leaves, sibling_pairs, _, _ = result
    assert leaves == []
    assert sibling_pairs == []


def test_cell_division_zero_generations_without_metadata_returns_three_tuple():
    """generations=0 with include_metadata=False (default) returns bare 3-tuple."""
    root = Cell(parent=None, generation=0)
    result = cell_division(root, generations=0)
    assert len(result) == 3
    _, leaves, sibling_pairs = result
    assert leaves == []
    assert sibling_pairs == []


def test_cell_division_without_metadata_returns_three_tuple():
    """Default include_metadata=False returns (root, leaves, sibling_pairs) only."""
    root = Cell(parent=None, generation=0)
    result = cell_division(root, generations=2)
    assert len(result) == 3
    _, leaves, sibling_pairs = result
    assert len(leaves) == 4
    assert len(sibling_pairs) == 2


# -----------------------------------------------------------------------------
# coordinates_generate_radians — n <= 0 (line 538)
# -----------------------------------------------------------------------------


def test_coordinates_generate_radians_zero_returns_empty_array():
    result = coordinates_generate_radians(0)
    assert result.shape == (0, 2)


def test_coordinates_generate_radians_negative_returns_empty_array():
    result = coordinates_generate_radians(-3)
    assert result.shape == (0, 2)


# -----------------------------------------------------------------------------
# _convert_layered_positions_to_cartesian_unit_sphere — error path (lines 638-641)
# -----------------------------------------------------------------------------


def test_convert_layered_positions_raises_when_both_positions_are_none():
    from embryobiopsy3d.lineage_simulator import (
        _convert_layered_positions_to_cartesian_unit_sphere,
    )

    cell = Cell(parent=None, generation=0)
    # Both layer_position and position are None by default after construction.
    assert cell.layer_position is None
    assert cell.position is None
    with pytest.raises(ValueError, match="cell has no position"):
        _convert_layered_positions_to_cartesian_unit_sphere([cell])


def test_convert_layered_positions_uses_position_when_layer_position_is_none():
    """layer_position=None but position set → falls through to the coord assignment."""
    from embryobiopsy3d.lineage_simulator import (
        _convert_layered_positions_to_cartesian_unit_sphere,
    )

    cell = Cell(parent=None, generation=0)
    cell.position = [1.0, 0.0, 0.0]
    cell.layer_position = None
    coords = _convert_layered_positions_to_cartesian_unit_sphere([cell])
    assert coords.shape == (1, 3)
    assert np.allclose(coords[0], [1.0, 0.0, 0.0])


# -----------------------------------------------------------------------------
# _generation_targets edge cases (lines 670, 684, 701)
# -----------------------------------------------------------------------------


def test_generation_targets_skips_parents_with_no_children():
    """child_count==0 branch: parent is skipped; output arrays are empty."""
    from embryobiopsy3d.lineage_simulator import _generation_targets

    parent = Cell(parent=None, generation=0)
    parent.layer_position = [1.0, 0.5, 0.5]
    # children list is already [] by default

    next_layer, child_angles, sibling_pairs = _generation_targets(
        [parent], generation=1, total_layers=3, alpha=0.3
    )
    assert next_layer == []
    assert child_angles.shape == (0, 2)
    assert sibling_pairs == []


def test_generation_targets_raises_for_non_binary_parent():
    """child_count != 2 raises ValueError."""
    from embryobiopsy3d.lineage_simulator import _generation_targets

    parent = Cell(parent=None, generation=0)
    parent.layer_position = [1.0, 0.5, 0.5]
    for _ in range(3):
        child = Cell(parent=parent, generation=1)
        parent.children.append(child)

    with pytest.raises(ValueError, match="child count is not 2"):
        _generation_targets([parent], generation=1, total_layers=3, alpha=0.3)


# -----------------------------------------------------------------------------
# _position_leaves — existing-positions path (line 824)
# -----------------------------------------------------------------------------


def test_position_leaves_builds_coords_array_from_existing_positions():
    """When leaves already have positions and coords=None, line 824 assembles coords_array."""
    from embryobiopsy3d.lineage_simulator import _position_leaves

    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
        generations=2, include_metadata=True
    )
    # Give every leaf a position so positions_missing=False.
    for i, leaf in enumerate(leaves):
        angle = 2 * np.pi * i / len(leaves)
        leaf.position = [np.cos(angle), np.sin(angle), 0.0]

    _, coords_array, _ = _position_leaves(
        leaves=leaves,
        sibling_pairs=sibling_pairs,
        coords=None,  # no explicit coords → triggers line 824
        placement_dispersal=0.0,
        generation_layers=generation_layers,
    )
    assert coords_array is not None
    assert coords_array.shape == (len(leaves), 3)
