"""
Reusable visualization scene helpers for embryo and rebiopsy demos.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from ..lineage_simulator import Embryo, apply_error_rates, build_embryo, generate_tree
from ..rebiopsy import rebiopsy_single_embryo

ErrorMode = Literal["none", "random", "fixed_generation"]


@dataclass(frozen=True)
class SceneNode:
    """Serializable node state shared by 3D embryo and 2D lineage views."""

    id: str
    generation: int
    generation_index: int
    is_leaf: bool
    is_aneuploid: bool
    lineage_x: float
    lineage_y: float
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    layer_radius: Optional[float] = None
    layer_theta: Optional[float] = None
    layer_phi: Optional[float] = None
    in_first_biopsy: bool = False
    in_second_biopsy: bool = False
    is_first_center: bool = False
    is_second_center: bool = False
    # True when the cell is euploid but its division produced aneuploid daughters.
    divides_erroneously: bool = False
    # UUID of the ancestor whose erroneous division caused this cell's aneuploidy;
    # None for euploid cells and for cells that are aneuploid for other reasons.
    error_progenitor: Optional[str] = None


@dataclass(frozen=True)
class SceneEdge:
    """Connection between two nodes."""

    source_id: str
    target_id: str


@dataclass(frozen=True)
class SceneBiopsy:
    """Plain metadata for a rebiopsy visualization."""

    first_center_id: str
    second_center_id: str
    first_leaf_ids: list[str]
    second_leaf_ids: list[str]
    standard_category: str
    second_category: Optional[str]
    standard_aneuploid_count: int
    second_aneuploid_count: Optional[int]
    match: bool
    requested_distance: float
    actual_distance: Optional[float]
    error: Optional[str] = None


@dataclass(frozen=True)
class EmbryoScene:
    """Complete plot-ready scene for an embryo demo."""

    nodes: list[SceneNode]
    lineage_edges: list[SceneEdge]
    sibling_edges: list[SceneEdge]
    metadata: dict[str, Any] = field(default_factory=dict)
    biopsy: Optional[SceneBiopsy] = None


def _iter_tree_nodes(root) -> list:
    nodes = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        nodes.append(node)
        queue.extend(node.children)
    return nodes


def _lineage_layout(root, leaf_order: dict[str, int]) -> dict[str, tuple[float, float]]:
    """Return (leaf spread x, generation) per node."""
    positions: dict[str, tuple[float, float]] = {}

    def assign(node) -> float:
        if not node.children:
            x = float(leaf_order[node.id])
        else:
            x = float(
                sum(assign(child) for child in node.children) / len(node.children)
            )
        positions[node.id] = (x, float(node.generation))
        return x

    assign(root)
    return positions


def build_demo_embryo(
    *,
    generations: int = 8,
    dispersal: float = 0.0,
    error_mode: ErrorMode = "random",
    meio_rate: float = 0.1,
    mito_rate: float = 0.05,
    error_seed: int = 42,
    placement_seed: int = 7,
    target_generation: Optional[int] = None,
    target_index: int = 0,
    include_subtree: bool = True,
) -> Embryo:
    """Build an embryo for interactive demos."""
    root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
        generations=generations,
        include_metadata=True,
    )

    if error_mode == "random":
        rng = np.random.default_rng(error_seed)
        apply_error_rates(root, meio_rate, mito_rate, rng)
    elif error_mode == "fixed_generation":
        if target_generation is None:
            raise ValueError("target_generation is required in fixed_generation mode.")
    elif error_mode != "none":
        raise ValueError(f"Unsupported error_mode: {error_mode!r}")

    embryo = build_embryo(
        root=root,
        leaves=leaves,
        sibling_pairs=sibling_pairs,
        id_dict=id_dict,
        generation_layers=generation_layers,
        placement_dispersal=dispersal,
        seed=placement_seed,
    )

    if error_mode == "fixed_generation":
        embryo.set_aneuploid_by_generation_index(
            target_generation,
            target_index,
            include_subtree=include_subtree,
        )

    return embryo


def build_demo_scene(
    *,
    generations: int = 8,
    dispersal: float = 0.0,
    error_mode: ErrorMode = "random",
    meio_rate: float = 0.1,
    mito_rate: float = 0.05,
    error_seed: int = 42,
    placement_seed: int = 7,
    target_generation: Optional[int] = None,
    target_index: int = 0,
    include_subtree: bool = True,
    rebiopsy_distance: Optional[float] = None,
    biopsy_seed: int = 11,
) -> EmbryoScene:
    """Build a complete scene for construction-only or rebiopsy demos."""
    embryo = build_demo_embryo(
        generations=generations,
        dispersal=dispersal,
        error_mode=error_mode,
        meio_rate=meio_rate,
        mito_rate=mito_rate,
        error_seed=error_seed,
        placement_seed=placement_seed,
        target_generation=target_generation,
        target_index=target_index,
        include_subtree=include_subtree,
    )

    biopsy_meta = None
    if rebiopsy_distance is not None:
        biopsy_meta = rebiopsy_single_embryo(
            embryo,
            distance=rebiopsy_distance,
            return_metadata=True,
            seed=biopsy_seed,
        )

    return build_embryo_scene(
        embryo,
        biopsy_meta=biopsy_meta,
        metadata={
            "generations": generations,
            "dispersal": dispersal,
            "error_mode": error_mode,
            "meio_rate": meio_rate,
            "mito_rate": mito_rate,
            "error_seed": error_seed,
            "placement_seed": placement_seed,
            "target_generation": target_generation,
            "target_index": target_index,
            "include_subtree": include_subtree,
            "rebiopsy_distance": rebiopsy_distance,
            "biopsy_seed": biopsy_seed if rebiopsy_distance is not None else None,
        },
    )


def build_embryo_scene(
    embryo: Embryo,
    *,
    biopsy_meta: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> EmbryoScene:
    """Convert an embryo object into plot-ready scene data."""
    nodes = _iter_tree_nodes(embryo.root)
    max_gen = max(node.generation for node in nodes)
    leaf_order = {leaf.id: idx for idx, leaf in enumerate(embryo.leaves)}
    layout = _lineage_layout(embryo.root, leaf_order)

    generation_index = {}
    if embryo.generation_layers:
        for gen, layer in enumerate(embryo.generation_layers):
            for idx, node in enumerate(layer):
                generation_index[node.id] = idx

    first_ids = set()
    second_ids = set()
    first_center_id = None
    second_center_id = None
    biopsy = None
    if biopsy_meta is not None:
        first_ids = {leaf.id for leaf in biopsy_meta["standard_leaves"]}
        second_ids = {leaf.id for leaf in biopsy_meta["second_leaves"]}
        first_center_id = biopsy_meta["standard_center"].id
        second_center_id = (
            biopsy_meta["second_center"].id
            if biopsy_meta["second_center"] is not None
            else None
        )
        biopsy = SceneBiopsy(
            first_center_id=first_center_id,
            second_center_id=second_center_id,
            first_leaf_ids=sorted(first_ids),
            second_leaf_ids=sorted(second_ids),
            standard_category=biopsy_meta["standard_category"],
            second_category=biopsy_meta["second_category"],
            standard_aneuploid_count=biopsy_meta["standard_aneuploid_count"],
            second_aneuploid_count=biopsy_meta["second_aneuploid_count"],
            match=bool(biopsy_meta["match"]),
            requested_distance=float(biopsy_meta["requested_distance"]),
            actual_distance=(
                None
                if biopsy_meta["actual_distance"] is None
                else float(biopsy_meta["actual_distance"])
            ),
            error=biopsy_meta.get("error"),
        )

    scene_nodes = []
    for node in nodes:
        lineage_x, generation = layout[node.id]
        position = getattr(node, "position", None)
        layered = getattr(node, "layer_position", None)
        scene_nodes.append(
            SceneNode(
                id=node.id,
                generation=node.generation,
                generation_index=generation_index.get(node.id, 0),
                is_leaf=not bool(node.children),
                is_aneuploid=bool(node.is_aneuploid),
                lineage_x=lineage_x,
                lineage_y=float(max_gen - generation),
                x=None if position is None else float(position[0]),
                y=None if position is None else float(position[1]),
                z=None if position is None else float(position[2]),
                layer_radius=None if layered is None else float(layered[0]),
                layer_theta=None if layered is None else float(layered[1]),
                layer_phi=None if layered is None else float(layered[2]),
                in_first_biopsy=node.id in first_ids,
                in_second_biopsy=node.id in second_ids,
                is_first_center=node.id == first_center_id,
                is_second_center=node.id == second_center_id,
                divides_erroneously=bool(getattr(node, "divides_erroneously", False)),
                error_progenitor=getattr(node, "error_progenitor", None) or None,
            )
        )

    lineage_edges = [
        SceneEdge(source_id=node.id, target_id=child.id)
        for node in nodes
        for child in node.children
    ]
    sibling_edges = [
        SceneEdge(source_id=left.id, target_id=right.id)
        for left, right in embryo.sibling_pairs
    ]

    scene_metadata = dict(metadata or {})
    scene_metadata.update(
        {
            "total_nodes": len(nodes),
            "leaf_count": len(embryo.leaves),
            "aneuploid_nodes": sum(node.is_aneuploid for node in nodes),
            "aneuploid_leaves": sum(leaf.is_aneuploid for leaf in embryo.leaves),
            "erroneous_division_nodes": sum(
                bool(getattr(node, "divides_erroneously", False)) for node in nodes
            ),
        }
    )

    return EmbryoScene(
        nodes=scene_nodes,
        lineage_edges=lineage_edges,
        sibling_edges=sibling_edges,
        metadata=scene_metadata,
        biopsy=biopsy,
    )


def serialize_scene(scene: EmbryoScene) -> dict[str, Any]:
    """Convert a scene dataclass to plain Python structures."""
    payload = {
        "nodes": [asdict(node) for node in scene.nodes],
        "lineage_edges": [asdict(edge) for edge in scene.lineage_edges],
        "sibling_edges": [asdict(edge) for edge in scene.sibling_edges],
        "metadata": dict(scene.metadata),
        "biopsy": None if scene.biopsy is None else asdict(scene.biopsy),
    }
    return payload


def scene_summary_rows(scene: EmbryoScene) -> list[dict[str, Any]]:
    """Return label/value rows for quick Streamlit tables."""
    rows = [
        {"metric": "generations", "value": scene.metadata.get("generations")},
        {"metric": "dispersal", "value": scene.metadata.get("dispersal")},
        {"metric": "error_mode", "value": scene.metadata.get("error_mode")},
        {"metric": "leaf_count", "value": scene.metadata.get("leaf_count")},
        {"metric": "aneuploid_leaves", "value": scene.metadata.get("aneuploid_leaves")},
        {"metric": "total_nodes", "value": scene.metadata.get("total_nodes")},
        {"metric": "aneuploid_nodes", "value": scene.metadata.get("aneuploid_nodes")},
        # Erroneous-division progenitors (euploid cells whose division misfired).
        {
            "metric": "erroneous_division_nodes",
            "value": scene.metadata.get("erroneous_division_nodes", 0),
        },
    ]
    if scene.metadata.get("error_mode") == "random":
        rows.extend(
            [
                {"metric": "meio_rate", "value": scene.metadata.get("meio_rate")},
                {"metric": "mito_rate", "value": scene.metadata.get("mito_rate")},
                {"metric": "error_seed", "value": scene.metadata.get("error_seed")},
            ]
        )
    if scene.metadata.get("error_mode") == "fixed_generation":
        rows.extend(
            [
                {
                    "metric": "target_generation",
                    "value": scene.metadata.get("target_generation"),
                },
                {"metric": "target_index", "value": scene.metadata.get("target_index")},
                {
                    "metric": "include_subtree",
                    "value": scene.metadata.get("include_subtree"),
                },
            ]
        )
    rows.append(
        {"metric": "placement_seed", "value": scene.metadata.get("placement_seed")}
    )

    if scene.biopsy is not None:
        rows.extend(
            [
                {
                    "metric": "requested_rebiopsy_distance_radians",
                    "value": scene.biopsy.requested_distance,
                },
                {
                    "metric": "actual_rebiopsy_distance_radians",
                    "value": scene.biopsy.actual_distance,
                },
                {
                    "metric": "first_biopsy_category",
                    "value": scene.biopsy.standard_category,
                },
                {
                    "metric": "second_biopsy_category",
                    "value": scene.biopsy.second_category,
                },
                {"metric": "concordant", "value": scene.biopsy.match},
                {"metric": "biopsy_seed", "value": scene.metadata.get("biopsy_seed")},
            ]
        )
        if scene.biopsy.error:
            rows.append({"metric": "biopsy_error", "value": scene.biopsy.error})

    return rows


def scene_leaf_rows(scene: EmbryoScene) -> list[dict[str, Any]]:
    """Return leaf-centric rows for tables and debugging."""
    rows = []
    for node in scene.nodes:
        if not node.is_leaf:
            continue
        rows.append(
            {
                "id": node.id[:8],
                "generation": node.generation,
                "generation_index": node.generation_index,
                "is_aneuploid": node.is_aneuploid,
                # Which erroneous division produced this cell's aneuploidy, if any.
                "error_progenitor_id": (
                    node.error_progenitor[:8] if node.error_progenitor else ""
                ),
                "x": node.x,
                "y": node.y,
                "z": node.z,
                "first_biopsy": node.in_first_biopsy,
                "second_biopsy": node.in_second_biopsy,
                "first_center": node.is_first_center,
                "second_center": node.is_second_center,
            }
        )
    return rows


def scene_progenitor_rows(scene: EmbryoScene) -> list[dict[str, Any]]:
    """Return one row per erroneous-division progenitor node.

    A progenitor is a euploid internal cell whose division misfired, making
    all its descendants aneuploid.  This table is useful for counting how many
    independent mitotic errors occurred and at which generation each arose.
    """
    # Build a fast lookup: progenitor_id → how many aneuploid descendants it caused.
    descendant_counts: dict[str, int] = {}
    for node in scene.nodes:
        if node.error_progenitor:
            descendant_counts[node.error_progenitor] = (
                descendant_counts.get(node.error_progenitor, 0) + 1
            )

    rows = []
    for node in scene.nodes:
        if not node.divides_erroneously:
            continue
        rows.append(
            {
                "id": node.id[:8],
                "generation": node.generation,
                "generation_index": node.generation_index,
                # Number of aneuploid cells (at any depth) descended from this error.
                "aneuploid_descendants": descendant_counts.get(node.id, 0),
                "lineage_x": round(node.lineage_x, 3),
                "lineage_y": round(node.lineage_y, 3),
            }
        )
    # Sort by generation so earlier errors appear first.
    rows.sort(key=lambda r: (r["generation"], r["generation_index"]))
    return rows
