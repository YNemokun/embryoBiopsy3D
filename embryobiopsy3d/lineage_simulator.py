"""
Lineage and spatial placement utilities for embryo simulation.

This module builds a binary lineage tree simulating cell division,
annotates aneuploidy, and places leaf cells on a sphere for downstream
biopsy and visualization workflows.
"""

import uuid
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import arange, arccos, cos, pi, sin
from scipy.optimize import linear_sum_assignment


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _ensure_rng(
    rng: Optional[np.random.Generator] = None, seed: Optional[int] = None
) -> np.random.Generator:
    """Return a numpy `Generator`"""
    if rng is not None and seed is None:
        return rng
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class Embryo:
    """Container embryo class (data structure) for the simulated embryo state."""

    root: "Cell"
    leaves: list["Cell"]
    sibling_pairs: list[tuple["Cell", "Cell"]]  # lineage sibling pairs of leaves
    coords: Optional[np.ndarray] = (
        None  # Cartesian unit vector positions, for plotting and all latter uses
    )
    placement_dispersal: Optional[float] = None
    generation_rng: Optional[np.random.Generator] = None
    mutated_cells: Optional[list["Cell"]] = (
        None  # aneuploid cells that need to be reset to euploid
    )
    error_progenitors: Optional[list["Cell"]] = (
        None  # cells that will divide erroneously
    )
    id_dict: Optional[dict[str, "Cell"]] = None  # fast lookup by UUID
    generation_layers: Optional[list[list["Cell"]]] = None  # index leaves by generation

    def _record_affected_cells(
        self, affected: list["Cell"], is_aneuploid: bool
    ) -> list["Cell"]:
        """Keep mutated_cells in sync with manual flag updates."""
        if self.mutated_cells is None:
            self.mutated_cells = []
        if is_aneuploid:
            for cell in affected:
                if cell not in self.mutated_cells:
                    self.mutated_cells.append(cell)
        else:
            self.mutated_cells = [
                cell for cell in self.mutated_cells if cell not in affected
            ]
        return affected

    def _record_erroneous_progenitor(
        self, progenitor: "Cell", is_aneuploid: bool
    ) -> None:
        """Keep error_progenitors in sync with manual flag updates."""
        if self.error_progenitors is None:
            self.error_progenitors = []
        if is_aneuploid:
            if progenitor not in self.error_progenitors:
                self.error_progenitors.append(progenitor)

    def get_node_by_id(self, node_id: str) -> "Cell":
        """Return the node matching the given id."""
        if not self.id_dict:
            raise ValueError("Embryo id_dict is not initialized.")
        return self.id_dict.get(node_id)

    def set_aneuploid_by_id(
        self,
        node_id: str,
        is_aneuploid: bool = True,
        include_subtree: bool = True,
        *,
        erroneous_division: bool = False,
    ) -> list["Cell"]:
        """Set aneuploid status for a node id, optionally including its subtree.

        Parameters
        ----------
        erroneous_division : bool
            Keyword-only.  When True, the node itself stays euploid but its
            division misfires — only descendants are (un)marked aneuploid and
            the node is flagged ``divides_erroneously``.
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            raise ValueError(f"Node id not found: {node_id!r}")
        affected = node.set_aneuploid(
            is_aneuploid, include_subtree, erroneous_division=erroneous_division
        )
        self._record_affected_cells(affected, is_aneuploid)
        if erroneous_division:
            self._record_erroneous_progenitor(node, is_aneuploid)
        return affected

    def get_node_by_generation_index(self, generation: int, index: int) -> "Cell":
        """Return the node at generation_layers[generation][index]."""
        if not self.generation_layers:
            raise ValueError("Embryo generation_layers is not initialized.")
        if generation < 0 or index < 0:
            raise ValueError("generation and index must be non-negative.")
        try:
            return self.generation_layers[generation][index]
        except IndexError as exc:
            raise ValueError("generation or index out of range.") from exc

    def set_aneuploid_by_generation_index(
        self,
        generation: int,
        index: int,
        is_aneuploid: bool = True,
        include_subtree: bool = True,
        *,
        erroneous_division: bool = False,
    ) -> list["Cell"]:
        """Set aneuploid status for a (generation, index) node, optionally its subtree.

        Parameters
        ----------
        erroneous_division : bool
            Keyword-only.  When True, the node itself stays euploid but its
            division misfires — only descendants are (un)marked aneuploid and
            the node is flagged ``divides_erroneously``.
        """
        node = self.get_node_by_generation_index(generation, index)
        affected = node.set_aneuploid(
            is_aneuploid, include_subtree, erroneous_division=erroneous_division
        )
        self._record_affected_cells(affected, is_aneuploid)
        if erroneous_division:
            self._record_erroneous_progenitor(node, is_aneuploid)
        return affected


class Cell:
    """Cell structure with lineage info, aneuploidy flag, position, and children."""

    def __init__(self, parent=None, generation=0):
        self.id = str(uuid.uuid4())  # unique identifier
        self.parent = parent
        self.children = []
        self.generation = generation  # root is the 0th generation; nth generation = n divisions happened
        self.position = (
            None  # Cartesian unit vector positions, for plotting and all latter uses
        )
        self.layer_position = (
            None  # Radian, for calculating cell placement during construction
        )
        self.divides_erroneously = False  # its following division will have an error, causing two aneuploid daughter cells
        self.is_aneuploid = False  # default to euploid
        self.error_progenitor = (
            None  # the ancestor cell whose division causes the aneuploidy in the tree
        )

    def __repr__(self):  # for debug, changing the print behavior
        return (
            f"\n Cell(id={self.id[:8]}, "
            f"gen={self.generation}, "
            f"aneuploid={self.is_aneuploid}, "
            f"divides_erroneously={self.divides_erroneously}, "
            f"error_progenitor={self.error_progenitor}, "
            f"pos={self.position}, "
            f"layer_pos={self.layer_position})"
        )

    def set_aneuploid(
        self,
        is_aneuploid: bool = True,
        include_subtree: bool = True,
        *,
        erroneous_division: bool = False,
    ) -> list["Cell"]:
        """Set or clear aneuploid status on this cell.

        Parameters
        ----------
        is_aneuploid : bool
            True to mark as aneuploid; False to revert to euploid.
        include_subtree : bool
            When ``erroneous_division=False`` (the default):
            True  → mark this cell *and* all descendants (original behaviour).
            False → mark only this single cell with no propagation.
            Ignored when ``erroneous_division=True``.
        erroneous_division : bool
            Keyword-only.  When True, *this* cell is euploid but its division
            misfires — all descendants are (un)marked aneuploid and the cell
            is flagged ``divides_erroneously``.  The cell itself is never added
            to the affected list.  Use this to distinguish a mitotic error
            (erroneous division) from a pre-existing aneuploid cell.
        """
        affected = []
        if erroneous_division:
            # Progenitor cell stays euploid; its division propagates the error.
            self.divides_erroneously = is_aneuploid
            queue = deque(self.children)
            while queue:
                node = queue.popleft()
                node.is_aneuploid = is_aneuploid
                node.error_progenitor = self.id if is_aneuploid else None
                affected.append(node)
                queue.extend(node.children)
        elif include_subtree:
            # Original behaviour: this cell itself is aneuploid and propagates.
            queue = deque([self])
            while queue:
                node = queue.popleft()
                node.is_aneuploid = is_aneuploid
                affected.append(node)
                queue.extend(node.children)
        else:
            # Single-cell assignment only, no propagation.
            self.is_aneuploid = is_aneuploid
            self.error_progenitor = None
            affected.append(self)
        return affected


# -----------------------------------------------------------------------------
# Lineage tree generation
# -----------------------------------------------------------------------------


def build_id_dict_and_layers(
    root: "Cell",
) -> tuple[dict[str, "Cell"], list[list["Cell"]]]:
    """Return (id_dict, generation_layers) for the given root (BFS)."""
    # Empty root means no nodes and no generation buckets.
    if root is None:
        return {}, []
    # id_dict: fast lookup by UUID
    # generation_layers: index leaves by generation
    id_dict = {}
    generation_layers = []
    queue = deque([root])
    # breadth-first search through the tree
    while queue:
        node = queue.popleft()
        id_dict[node.id] = node
        gen = node.generation
        # Ensure the generation bucket exists
        while len(generation_layers) <= gen:
            generation_layers.append([])
        generation_layers[gen].append(node)
        # Continue search through descendants
        queue.extend(node.children)
    return id_dict, generation_layers


def _initialize_generation_metadata(
    root: "Cell",
    generations: int,
    *,
    id_dict: Optional[dict[str, "Cell"]] = None,
    generation_layers: Optional[list[list["Cell"]]] = None,
) -> tuple[dict[str, "Cell"], list[list["Cell"]]]:
    """Allocate id- and generation-indexed containers for the requested number of generations."""
    if id_dict is None:
        id_dict = {root.id: root}
    if generation_layers is None:
        generation_layers = [[] for _ in range(generations + 1)]
        generation_layers[0].append(root)
    elif len(generation_layers) <= generations:
        generation_layers.extend(
            [[] for _ in range(generations + 1 - len(generation_layers))]
        )
    return id_dict, generation_layers


def cell_division(
    root: "Cell",
    generations: int,
    id_dict: Optional[dict[str, "Cell"]] = None,
    generation_layers: Optional[list[list["Cell"]]] = None,
    *,
    include_metadata: bool = False,
) -> tuple[
    tuple[
        "Cell",
        list["Cell"],
        list[tuple["Cell", "Cell"]],
        dict[str, "Cell"],
        list[list["Cell"]],
    ]
]:
    """Generate cell divisions without applying aneuploidy.

    If include_metadata is True, returns id_dict and generation_layers as well.
    """
    id_dict, generation_layers = _initialize_generation_metadata(
        root,
        generations,
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    if generations <= 0:  # no cell division
        if include_metadata:
            return root, [], [], id_dict, generation_layers
        return root, [], []

    sibling_pairs = []
    # Start from the provided root and expand generation by generation.
    current_generation = [root]

    # Each loop iteration creates one new generation from the current frontier.
    for generation_index in range(generations):
        next_generation = []
        for parent in current_generation:
            # Normal leaf division
            child1 = Cell(parent=parent, generation=parent.generation + 1)
            child2 = Cell(parent=parent, generation=parent.generation + 1)

            # Log parent-child relationships and add to the new leaves
            parent.children.extend([child1, child2])
            next_generation.extend([child1, child2])

            # add the children to the id_dict and generation_layers
            id_dict[child1.id] = child1
            id_dict[child2.id] = child2
            generation_layers[parent.generation + 1].append(child1)
            generation_layers[parent.generation + 1].append(child2)

            # at the last generation, record the sibling pairs
            if generation_index == generations - 1:
                sibling_pairs.append((child1, child2))

        current_generation = next_generation

    # Optionally return metadata tables for later use
    if include_metadata:
        return root, current_generation, sibling_pairs, id_dict, generation_layers
    return root, current_generation, sibling_pairs


def apply_error_rates(
    root: "Cell",
    meio_rate: float,
    mito_rate: float,
    rng: Optional[np.random.Generator] = None,
    *,
    error_progenitors: Optional[list["Cell"]] = None,
) -> list["Cell"]:
    """Randomly assign meiotic/mitotic errors on an existing tree structure (BFS).

    Parameters
    ----------
    error_progenitors : list[Cell] or None
        If provided, cells whose division fires a mitotic error are appended
        to this list so callers can reset ``divides_erroneously`` later.
        Passing ``None`` (default) skips that bookkeeping.
    """
    rng = _ensure_rng(rng)
    mutated_cells = []

    # Meiotic error: the egg/sperm was already aneuploid — flag the root.
    if rng.random() < meio_rate:
        root.is_aneuploid = True

    # BFS to propagate inherited and new mitotic errors.
    queue = deque([root])
    while queue:
        cell = queue.popleft()

        if not cell.children:
            # Leaf node — record if aneuploid so it can be reset later.
            if cell.is_aneuploid:
                mutated_cells.append(cell)
            continue

        # Stochastic mitotic error: this cell is euploid but divides wrongly.
        if rng.random() < mito_rate:
            cell.divides_erroneously = True
            if error_progenitors is not None:
                error_progenitors.append(cell)

        if cell.is_aneuploid or cell.divides_erroneously:
            progenitor = cell.id if cell.divides_erroneously else cell.error_progenitor
            # only add the cell to the mutated cells if it is aneuploid
            if cell.is_aneuploid:
                mutated_cells.append(cell)
            # Children are all aneuploid
            for child in cell.children:
                child.is_aneuploid = True
                # Track which erroneous division caused the child's aneuploidy.
                child.error_progenitor = progenitor

        queue.extend(cell.children)

    return mutated_cells


def reset_flags(
    mutated_cells: list["Cell"],
    error_progenitors: Optional[list["Cell"]] = None,
) -> None:
    """Reset aneuploid flags and erroneous-division metadata.

    Parameters
    ----------
    mutated_cells : list[Cell]
        Aneuploid cells to clear.  ``is_aneuploid`` and ``error_progenitor``
        are both reset so plotting metadata does not go stale.
    error_progenitors : list[Cell] or None
        Cells that were flagged ``divides_erroneously`` during the same trial.
        If provided, their flag is cleared so the next trial starts clean.
    """
    for cell in mutated_cells:
        cell.is_aneuploid = False
        cell.error_progenitor = None
    mutated_cells.clear()
    if error_progenitors is not None:
        for cell in error_progenitors:
            cell.divides_erroneously = False
        error_progenitors.clear()


def generate_tree(
    generations: int, *, include_metadata: bool = False
) -> tuple[
    tuple[
        "Cell",
        list["Cell"],
        list[tuple["Cell", "Cell"]],
        dict[str, "Cell"],
        list[list["Cell"]],
    ]
]:
    """Generate a binary lineage tree for a fixed number of generations.

    If include_metadata is True, returns id_dict and generation_layers as well.
    """
    # Initialize a single-cell embryo root.
    root = Cell(parent=None, generation=0)
    leaves = []
    sibling_pairs = []
    id_dict, generation_layers = _initialize_generation_metadata(root, generations)
    # simulate the cell division
    if generations > 0:
        root, leaves, sibling_pairs, id_dict, generation_layers = cell_division(
            root,
            generations=generations,
            id_dict=id_dict,
            generation_layers=generation_layers,
            include_metadata=True,
        )
    # for later analysis
    if include_metadata:
        return root, leaves, sibling_pairs, id_dict, generation_layers
    return root, leaves, sibling_pairs


# -----------------------------------------------------------------------------
# Sphere coordinates
# -----------------------------------------------------------------------------


def coordinates_generate_radians(n: int) -> np.ndarray:
    """Generate near-uniform spherical coordinates as (theta, phi) radians."""
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    golden_ratio = (1 + 5**0.5) / 2
    i = arange(n)
    theta = (2 * pi * i / golden_ratio) % (2 * pi)
    phi = arccos(1 - 2 * (i + 0.5) / n)
    return np.c_[theta, phi]


def _angles_to_cartesian(theta: float, phi: float, radius: float) -> np.ndarray:
    """Convert spherical (theta, phi) to cartesian (x, y, z) at given radius."""
    x = radius * cos(theta) * sin(phi)
    y = radius * sin(theta) * sin(phi)
    z = radius * cos(phi)
    return np.asarray([x, y, z], dtype=float)


# -----------------------------------------------------------------------------
# Lineage ordering and placement helpers
# -----------------------------------------------------------------------------


def _wrap_theta(theta: float) -> float:
    """Wrap azimuth  (x-axis) to [0, 2pi)."""
    return float(theta % (2 * pi))


def _reflect_phi(phi: float) -> float:
    """Reflect polar angle (z-axis) into [0, pi] while preserving continuity."""
    value = float(phi)
    two_pi = 2.0 * float(pi)
    value = value % two_pi
    # reflect the angle around the equator (z-axis)
    if value > float(pi):
        value = two_pi - value
    return value


def _ideal_angles_from_parent(
    theta: float, phi: float, alpha: float, beta: float
) -> np.ndarray:
    """Return a child ideal angle from parent by angle-space perturbation."""
    # Keep finite dtheta near poles where longitude is numerically unstable.
    sin_phi = float(np.sin(phi))
    denom = max(abs(sin_phi), 1e-3)
    # calculate changes for each angle
    dphi = alpha * float(np.cos(beta))
    dtheta = alpha * float(np.sin(beta)) / denom
    # keep the angles within the valid range
    child_theta = _wrap_theta(theta + dtheta)
    child_phi = _reflect_phi(phi + dphi)
    return np.asarray([child_theta, child_phi], dtype=float)


def build_cost_matrix(
    child_ideal_angles: np.ndarray, fibonacci_angles: np.ndarray
) -> np.ndarray:
    """Build C[i, j] = angular distance from child target i to slot j."""
    child_ideal_angles = np.asarray(child_ideal_angles, dtype=float)
    fibonacci_angles = np.asarray(fibonacci_angles, dtype=float)

    if len(child_ideal_angles) == 0 or len(fibonacci_angles) == 0:
        return np.zeros((len(child_ideal_angles), len(fibonacci_angles)), dtype=float)

    # matrix operations for efficient computation
    child_theta = child_ideal_angles[:, 0][:, None]
    child_phi = child_ideal_angles[:, 1][:, None]
    slot_theta = fibonacci_angles[:, 0][None, :]
    slot_phi = fibonacci_angles[:, 1][None, :]

    sin_child_phi = np.sin(child_phi)
    cos_child_phi = np.cos(child_phi)
    sin_slot_phi = np.sin(slot_phi)
    cos_slot_phi = np.cos(slot_phi)

    cos_gamma = (
        sin_child_phi * sin_slot_phi * np.cos(child_theta - slot_theta)
        + cos_child_phi * cos_slot_phi
    )
    return np.arccos(np.clip(cos_gamma, -1.0, 1.0))


def _convert_layered_positions_to_cartesian_unit_sphere(
    cells: list["Cell"],
) -> np.ndarray:
    """Cartesian unit-vector positions from each cell's layered coordinates."""
    coords = []
    for cell in cells:
        layered = getattr(cell, "layer_position", None)
        if layered is None:
            if cell.position is None:
                # this should not happen
                raise ValueError("cell has no position")
            coord = np.asarray(cell.position, dtype=float)
        else:
            _, theta, phi = layered
            coord = _angles_to_cartesian(theta, phi, radius=1.0)
            cell.position = coord.tolist()
        coords.append(coord)
    return np.asarray(coords, dtype=float)


def _generation_targets(
    previous_layer: list["Cell"],
    generation: int,
    total_layers: int,
    alpha: float,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list["Cell"], np.ndarray, list[tuple["Cell", "Cell"]]]:
    """Return child ideals and child ordering for one generation."""
    child_ideal_angles = []
    next_layer = []
    sibling_pairs = []
    rng = _ensure_rng(rng)
    # for dispersal direction spacing, so that the directions are spread out
    golden_step = (np.sqrt(5.0) - 1.0) / 2.0
    # generate a list of angles for beta0
    beta0_angles = 2 * pi * arange(len(previous_layer)) * golden_step

    for parent_idx, parent in enumerate(previous_layer):
        child_count = len(parent.children)
        if child_count == 0:
            continue

        # Calculate base dispersal direction
        parent_theta = float(parent.layer_position[1])
        parent_phi = float(parent.layer_position[2])
        # randomly select an angle from beta0_angles
        beta0 = beta0_angles[rng.integers(0, len(beta0_angles))]

        for child_idx, child in enumerate(parent.children):
            if child_count == 2:
                # Keep sibling opposite around the parent
                beta = beta0 + (pi * child_idx)
            else:
                # should not happen
                raise ValueError("child count is not 2")
            child_ideal_angles.append(
                # calculate the ideal theta and phi for the child
                _ideal_angles_from_parent(
                    parent_theta,
                    parent_phi,
                    alpha=alpha,
                    beta=beta,
                )
            )
            next_layer.append(child)

        if generation == total_layers - 1:
            sibling_pairs.append(tuple(parent.children))

    child_ideal_angles = np.asarray(child_ideal_angles, dtype=float)
    if child_ideal_angles.size == 0:
        child_ideal_angles = np.zeros((0, 2), dtype=float)

    return next_layer, child_ideal_angles, sibling_pairs


def _assign_hungarian_slots(
    next_layer: list["Cell"],
    child_ideal_angles: np.ndarray,
    angles: np.ndarray,
    radius: float,
) -> None:
    """Assign children to Fibonacci slots by solving the full cost matrix."""
    # calculate the cost matrix for the assignment problem
    cost_matrix = build_cost_matrix(child_ideal_angles, angles)
    # the best assignment is the one with the lowest cost
    child_indices, slot_indices = linear_sum_assignment(cost_matrix)
    # assign the coordinates to the children based on the assignments
    # child_ind[i] and coord_ind[i] are paired: child child_ind[i] gets coord_ind[i]
    for child_idx, slot_idx in zip(child_indices, slot_indices):
        next_layer[child_idx].layer_position = [
            radius,
            angles[slot_idx][0],
            angles[slot_idx][1],
        ]


def _run_bottom_up_positioning(
    generation_layers: list[list["Cell"]],
    dispersal: float,
    *,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list["Cell"], np.ndarray, list[tuple["Cell", "Cell"]]]:
    """Shared implementation for layered bottom-up leaf placement."""
    if generation_layers is None:
        raise ValueError("generation_layers is required for bottom-up positioning")
    if not 0 <= dispersal <= 1:
        raise ValueError("dispersal must be between 0 and 1 (inclusive)")
    # Clear cached node positions before rebuilding a layered placement.
    for layer in generation_layers:
        for node in layer:
            node.position = None
            node.layer_position = None

    rng = _ensure_rng(rng)

    sibling_pairs = []
    total_layers = len(generation_layers)

    # add a bit of jitter to the dispersal direction to avoid perfect alignment
    alpha = float(dispersal) * (pi / 2) + rng.uniform(0, 0.05)

    # assign positions to each generation's leaves
    for generation, current_layer in enumerate(generation_layers):
        angles = coordinates_generate_radians(len(current_layer))
        radius = float(generation + 1)

        if generation <= 1:
            # Assign Fibonacci slots directly for the first two generations.
            for node, angle in zip(current_layer, angles):
                node.layer_position = [radius, angle[0], angle[1]]
            continue

        # generate child ideals for the current generation
        next_layer, child_ideal_angles, generation_sibling_pairs = _generation_targets(
            generation_layers[generation - 1],
            generation,
            total_layers,
            alpha,
            rng=rng,
        )

        _assign_hungarian_slots(next_layer, child_ideal_angles, angles, radius)

        sibling_pairs.extend(generation_sibling_pairs)
        generation_layers[generation] = next_layer

    final_layer = generation_layers[-1] if generation_layers else []
    coords_array = np.asarray(
        [leaf.layer_position for leaf in final_layer], dtype=float
    )
    return final_layer, coords_array, sibling_pairs


# -----------------------------------------------------------------------------
# Sphere placement (layered bottom-up)
# -----------------------------------------------------------------------------


def _position_leaves(
    leaves: list["Cell"],
    sibling_pairs: list[tuple["Cell", "Cell"]],
    coords: np.ndarray,
    placement_dispersal: float,
    generation_layers: list[list["Cell"]],
    rng: Optional[np.random.Generator] = None,
) -> tuple[list["Cell"], np.ndarray, list[tuple["Cell", "Cell"]]]:
    """Ensure leaves have positions and return ordered leaves and coords.
    Uses layered bottom-up placement when positions must be computed.
    Return Cartesian coordinates at the end.
    """
    positions_missing = any(getattr(cell, "position", None) is None for cell in leaves)
    ordered_leaves = leaves

    coords_array = None
    if coords is not None:
        # Supplied explicit coordinates; attach directly.
        coords_array = np.asarray(coords, dtype=float)
        if len(coords_array) != len(leaves):
            raise ValueError("coords length must match number of leaves")
        for leaf, coord in zip(ordered_leaves, coords_array):
            leaf.position = coord
        positions_missing = False

    if positions_missing:
        # Layered bottom-up placement (Hungarian assignment to Fibonacci slots).
        ordered_leaves, layer_coords, sibling_pairs = _bottom_up_position_leaves(
            generation_layers=generation_layers,
            dispersal=placement_dispersal,
            rng=rng,
        )
        _convert_layered_positions_to_cartesian_unit_sphere(ordered_leaves)
        coords_array = np.array([leaf.position for leaf in ordered_leaves], dtype=float)
    elif coords_array is None:
        coords_array = np.array([leaf.position for leaf in ordered_leaves], dtype=float)

    return ordered_leaves, coords_array, sibling_pairs


def _bottom_up_position_leaves(
    generation_layers: list[list["Cell"]],
    dispersal: float,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list["Cell"], np.ndarray, list[tuple["Cell", "Cell"]]]:
    """Assign coordinates per generation using Hungarian algorithm for parent-local children.

    - Radius stays fixed per generation: radius = generation + 1.
    - Child ideals are generated by angle-space perturbations from each parent.
    - Hungarian assignment maps those ideals to available Fibonacci slots.
    """
    return _run_bottom_up_positioning(
        generation_layers,
        dispersal,
        rng=rng,
    )


# -----------------------------------------------------------------------------
# Embryo construction
# -----------------------------------------------------------------------------


def build_embryo(
    root: "Cell" = None,
    leaves: list["Cell"] = None,
    sibling_pairs: list[tuple["Cell", "Cell"]] = None,
    coords: np.ndarray = None,
    id_dict: dict[str, "Cell"] = None,
    generation_layers: list[list["Cell"]] = None,
    *,
    generations: int = None,
    meio_rate: float = None,
    mito_rate: float = None,
    seed: int = None,
    placement_dispersal: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Embryo:
    """Build an `Embryo` from either an existing tree or simulation parameters.

    If leaf positions are missing, layered bottom-up placement is used internally
    and final leaf positions are stored as Cartesian unit vectors. Layered
    `[radius, theta, phi]` coordinates remain available on `cell.layer_position`
    for generation-based visualization.
    """
    # Determine whether caller provided a fully realized lineage tree
    have_tree = root is not None and leaves is not None and sibling_pairs is not None
    generation_rng = None
    mutated_cells = []
    error_progenitors: list["Cell"] = []

    # Build the tree if none was provided.
    if not have_tree:
        if generations is None or meio_rate is None or mito_rate is None:
            raise ValueError(
                "build_embryo needs either (root, leaves, sibling_pairs) or the "
                "simulation parameters (generations, meio_rate, mito_rate)."
            )
        # Use one RNG instance for tree-level stochastic decisions.
        rng = _ensure_rng(rng, seed=seed)
        # create the lineage tree
        root, leaves, sibling_pairs, id_dict, generation_layers = generate_tree(
            generations=generations,
            include_metadata=True,
        )
        mutated_cells = apply_error_rates(
            root, meio_rate, mito_rate, rng, error_progenitors=error_progenitors
        )
        generation_rng = rng
    else:
        # tree already provided, skips tree generation and error rate application
        rng = _ensure_rng(rng, seed=seed)
        if id_dict is None or generation_layers is None:
            id_dict, generation_layers = build_id_dict_and_layers(root)

    ordered_leaves, coords, sibling_pairs = _position_leaves(
        leaves=leaves,
        sibling_pairs=sibling_pairs,
        coords=coords,
        placement_dispersal=placement_dispersal,
        generation_layers=generation_layers,
        rng=rng,
    )

    # Package all simulation state into the Embryo dataclass
    return Embryo(
        root=root,
        leaves=ordered_leaves,
        sibling_pairs=sibling_pairs,
        coords=coords,
        placement_dispersal=placement_dispersal,
        generation_rng=generation_rng,
        mutated_cells=mutated_cells,
        error_progenitors=error_progenitors,
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
