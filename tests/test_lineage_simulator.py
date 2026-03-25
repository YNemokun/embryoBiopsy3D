"""
Tests for lineage_simulator: Embryo API, lineage distance helper, Cell.set_aneuploid,
and build_embryo / _position_leaves validation.
"""

import numpy as np
import pytest

from embryobiopsy3d.lineage_simulator import (
    Cell,
    Embryo,
    generate_tree,
    build_embryo,
    coordinates_generate_radians,
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
    affected = emb.set_aneuploid_by_id(node_id, is_aneuploid=True, include_subtree=True)
    assert root.is_aneuploid
    assert all(leaf.is_aneuploid for leaf in leaves)
    assert len(affected) == 1 + 2 + 4  # root + gen1 + gen2


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


def test_build_embryo_raises_when_placement_strategy_invalid():
    """build_embryo raises when placement_strategy is not hungarian or greedy."""
    with pytest.raises(ValueError, match="placement_strategy must be either"):
        build_embryo(
            generations=3,
            meio_rate=0.0,
            mito_rate=0.0,
            placement_strategy="invalid",
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
