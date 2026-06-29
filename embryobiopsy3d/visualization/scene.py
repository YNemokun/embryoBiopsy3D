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
    """Serializable node state shared by 3-D embryo and 2-D lineage views.

    Attributes:
        id: Cell UUID string.
        generation: Division depth from the root (root = 0).
        generation_index: Position of this node within its generation layer.
        is_leaf: ``True`` for final-generation cells with no children.
        is_aneuploid: ``True`` when the cell carries a chromosomal error.
        lineage_x: Horizontal position in the 2-D lineage layout.
        lineage_y: Vertical position (generation depth, inverted) in the
            2-D lineage layout.
        x: Cartesian *x* coordinate on the unit sphere, or ``None``.
        y: Cartesian *y* coordinate on the unit sphere, or ``None``.
        z: Cartesian *z* coordinate on the unit sphere, or ``None``.
        layer_radius: Layered-placement radius, or ``None``.
        layer_theta: Layered-placement azimuth angle (radians), or ``None``.
        layer_phi: Layered-placement polar angle (radians), or ``None``.
        in_first_biopsy: ``True`` when this leaf was selected in the first biopsy.
        in_second_biopsy: ``True`` when this leaf was selected in the second biopsy.
        is_first_center: ``True`` when this leaf is the center of the first biopsy.
        is_second_center: ``True`` when this leaf is the center of the second biopsy.
    """

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


@dataclass(frozen=True)
class SceneEdge:
    """Directed edge between two nodes in the lineage or sibling graph.

    Attributes:
        source_id: UUID of the parent (or first sibling) node.
        target_id: UUID of the child (or second sibling) node.
    """

    source_id: str
    target_id: str


@dataclass(frozen=True)
class SceneBiopsy:
    """Plain serializable metadata produced by a paired-biopsy simulation.

    Attributes:
        first_center_id: UUID of the first-biopsy center leaf.
        second_center_id: UUID of the second-biopsy center leaf.
        first_leaf_ids: Sorted UUIDs of leaves in the first biopsy.
        second_leaf_ids: Sorted UUIDs of leaves in the second biopsy.
        standard_category: Category of the first biopsy
            (``"euploid"``, ``"mosaic"``, or ``"aneuploid"``).
        second_category: Category of the second biopsy, or ``None`` when the
            rebiopsy could not be performed.
        standard_aneuploid_count: Number of aneuploid cells in the first biopsy.
        second_aneuploid_count: Number of aneuploid cells in the second biopsy,
            or ``None``.
        match: ``True`` when both biopsies produced the same category.
        requested_distance: Target angular separation in radians (``distance × π``).
        actual_distance: Achieved angular separation in radians, or ``None``.
        error: Human-readable error message when sampling failed.
    """

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
    """Complete plot-ready scene for an embryo demo.

    Attributes:
        nodes: All cells (internal + leaf) as :class:`SceneNode` objects in
            BFS order from the root.
        lineage_edges: Parent → child edges for the lineage tree view.
        sibling_edges: Left ↔ right sibling edges for the 3-D sphere view.
        metadata: Scalar parameters used to build the scene (e.g.
            ``generations``, ``dispersal``, ``error_mode``).
        biopsy: Paired-biopsy metadata, or ``None`` when no rebiopsy was run.
    """

    nodes: list[SceneNode]
    lineage_edges: list[SceneEdge]
    sibling_edges: list[SceneEdge]
    metadata: dict[str, Any] = field(default_factory=dict)
    biopsy: Optional[SceneBiopsy] = None


def _iter_tree_nodes(root) -> list:
    """Return all cells reachable from *root* in BFS order."""
    nodes = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        nodes.append(node)
        queue.extend(node.children)
    return nodes


def _lineage_layout(root, leaf_order: dict[str, int]) -> dict[str, tuple[float, float]]:
    """Compute 2-D layout positions for every node in the lineage tree.

    Internal nodes are placed at the mean *x* of their children; leaves use
    their index in *leaf_order* as *x*.

    Args:
        root: Root cell of the lineage tree.
        leaf_order: Mapping from leaf UUID to horizontal position index.

    Returns:
        Dict mapping each node UUID to ``(x, generation)`` floats.
    """
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
    """Build a positioned :class:`~lineage_simulator.Embryo` for interactive demos.

    Args:
        generations: Number of cell-division rounds.
        dispersal: Placement dispersal in ``[0, 1]``.
        error_mode: Aneuploidy assignment strategy:

            * ``"none"`` — no errors applied.
            * ``"random"`` — stochastic meiotic + mitotic rates via
              :func:`~lineage_simulator.apply_error_rates`.
            * ``"fixed_generation"`` — a single deterministic subtree rooted
              at ``(target_generation, target_index)`` is marked aneuploid.
        meio_rate: Meiotic error probability (used when *error_mode* is ``"random"``).
        mito_rate: Mitotic error probability (used when *error_mode* is ``"random"``).
        error_seed: RNG seed for stochastic error assignment.
        placement_seed: RNG seed for sphere placement.
        target_generation: Generation of the aneuploid subtree root (required
            when *error_mode* is ``"fixed_generation"``).
        target_index: Index within *target_generation* for the subtree root.
        include_subtree: When ``True``, propagate the aneuploid flag to all
            descendants of the target cell.

    Returns:
        A fully positioned :class:`~lineage_simulator.Embryo`.

    Raises:
        ValueError: If *error_mode* is ``"fixed_generation"`` and
            *target_generation* is ``None``, or if *error_mode* is not one of
            the supported values.
    """
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
    """Build a complete :class:`EmbryoScene` for construction-only or rebiopsy demos.

    Delegates embryo construction to :func:`build_demo_embryo`, optionally
    runs a paired biopsy via :func:`~rebiopsy.rebiopsy_single_embryo`, then
    converts everything to a plot-ready scene via :func:`build_embryo_scene`.

    Args:
        generations: Number of cell-division rounds.
        dispersal: Placement dispersal in ``[0, 1]``.
        error_mode: Aneuploidy strategy — ``"none"``, ``"random"``, or
            ``"fixed_generation"``.
        meio_rate: Meiotic error probability (``"random"`` mode only).
        mito_rate: Mitotic error probability (``"random"`` mode only).
        error_seed: RNG seed for stochastic error assignment.
        placement_seed: RNG seed for sphere placement.
        target_generation: Subtree root generation (``"fixed_generation"`` only).
        target_index: Subtree root index within *target_generation*.
        include_subtree: Propagate aneuploid flag to descendants.
        rebiopsy_distance: Target biopsy separation as a fraction of π.  No
            rebiopsy is performed when ``None``.
        biopsy_seed: RNG seed for the paired-biopsy draw.

    Returns:
        A fully built :class:`EmbryoScene`.
    """
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
    """Convert a positioned :class:`~lineage_simulator.Embryo` into an :class:`EmbryoScene`.

    Args:
        embryo: Fully positioned embryo with ``generation_layers`` initialized.
        biopsy_meta: Optional dict returned by
            :func:`~rebiopsy.rebiopsy_single_embryo` with
            ``return_metadata=True``.  When provided, biopsy membership flags
            are set on each :class:`SceneNode` and a :class:`SceneBiopsy` is
            attached to the scene.
        metadata: Arbitrary scalar parameters to store in
            :attr:`EmbryoScene.metadata`.  Augmented with ``total_nodes``,
            ``leaf_count``, ``aneuploid_nodes``, and ``aneuploid_leaves``.

    Returns:
        A fully built :class:`EmbryoScene`.
    """
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
    """Convert an :class:`EmbryoScene` to plain JSON-serializable Python structures.

    Args:
        scene: Scene to serialize.

    Returns:
        Dict with keys ``"nodes"``, ``"lineage_edges"``, ``"sibling_edges"``,
        ``"metadata"``, and ``"biopsy"`` (``None`` when no rebiopsy was run).
    """
    payload = {
        "nodes": [asdict(node) for node in scene.nodes],
        "lineage_edges": [asdict(edge) for edge in scene.lineage_edges],
        "sibling_edges": [asdict(edge) for edge in scene.sibling_edges],
        "metadata": dict(scene.metadata),
        "biopsy": None if scene.biopsy is None else asdict(scene.biopsy),
    }
    return payload


def scene_summary_rows(scene: EmbryoScene) -> list[dict[str, Any]]:
    """Return ``{metric, value}`` rows summarizing key scene parameters.

    Includes conditional sections for ``"random"`` and ``"fixed_generation"``
    error modes, and biopsy outcome metrics when the scene has a
    :class:`SceneBiopsy`.

    Args:
        scene: Scene to summarize.

    Returns:
        List of dicts with keys ``"metric"`` and ``"value"``.
    """
    rows = [
        {"metric": "generations", "value": scene.metadata.get("generations")},
        {"metric": "dispersal", "value": scene.metadata.get("dispersal")},
        {"metric": "error_mode", "value": scene.metadata.get("error_mode")},
        {"metric": "leaf_count", "value": scene.metadata.get("leaf_count")},
        {"metric": "aneuploid_leaves", "value": scene.metadata.get("aneuploid_leaves")},
        {"metric": "total_nodes", "value": scene.metadata.get("total_nodes")},
        {"metric": "aneuploid_nodes", "value": scene.metadata.get("aneuploid_nodes")},
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
    """Return one row per leaf cell with position, aneuploid status, and biopsy flags.

    Args:
        scene: Scene whose leaf nodes to enumerate.

    Returns:
        List of dicts with keys ``"id"``, ``"generation"``,
        ``"generation_index"``, ``"x"``, ``"y"``, ``"z"``,
        ``"is_aneuploid"``, ``"first_biopsy"``, ``"second_biopsy"``,
        ``"first_center"``, and ``"second_center"``.
    """
    rows = []
    for node in scene.nodes:
        if not node.is_leaf:
            continue
        rows.append(
            {
                "id": node.id,
                "generation": node.generation,
                "generation_index": node.generation_index,
                "x": node.x,
                "y": node.y,
                "z": node.z,
                "is_aneuploid": node.is_aneuploid,
                "first_biopsy": node.in_first_biopsy,
                "second_biopsy": node.in_second_biopsy,
                "first_center": node.is_first_center,
                "second_center": node.is_second_center,
            }
        )
    return rows
