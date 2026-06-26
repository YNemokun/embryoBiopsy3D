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
    """Return a NumPy random ``Generator``, constructing one if necessary.

    Args:
        rng: Existing generator to reuse.  Returned unchanged when *seed* is
            also ``None``.
        seed: Integer seed used to create a fresh generator.  Takes precedence
            over *rng* when both are supplied.

    Returns:
        A :class:`numpy.random.Generator` instance.
    """
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
    """Container for all simulated embryo state after tree generation and placement.

    Attributes:
        root: Root cell of the lineage tree (generation 0).
        leaves: Ordered list of final-generation leaf cells.
        sibling_pairs: Pairs of sibling leaf cells at the last generation.
        coords: ``(N, 3)`` array of Cartesian unit-vector positions for leaves,
            or ``None`` before placement.
        placement_dispersal: Dispersal parameter used during sphere placement.
        generation_rng: Random generator that was active during tree generation;
            preserved for reproducible downstream draws.
        mutated_cells: Cells whose ``is_aneuploid`` flag was set during error-rate
            application.  Used for efficient state reset between trials.
        id_dict: UUID → Cell lookup table for O(1) node retrieval.
        generation_layers: ``generation_layers[g]`` lists all cells at generation *g*.
    """

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
    id_dict: Optional[dict[str, "Cell"]] = None  # fast lookup by UUID
    generation_layers: Optional[list[list["Cell"]]] = None  # index leaves by generation

    def _record_affected_cells(
        self, affected: list["Cell"], is_aneuploid: bool
    ) -> list["Cell"]:
        """Synchronize ``mutated_cells`` after a manual flag update.

        Args:
            affected: Cells whose ``is_aneuploid`` flag was just changed.
            is_aneuploid: The new flag value applied to *affected*.

        Returns:
            The *affected* list (pass-through for call-site convenience).
        """
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

    def get_node_by_id(self, node_id: str) -> "Cell":
        """Return the cell whose UUID matches *node_id*.

        Args:
            node_id: UUID string of the target cell.

        Returns:
            The matching :class:`Cell`, or ``None`` if the id is not present.

        Raises:
            ValueError: If ``id_dict`` has not been initialized.
        """
        if not self.id_dict:
            raise ValueError("Embryo id_dict is not initialized.")
        return self.id_dict.get(node_id)

    def set_aneuploid_by_id(
        self,
        node_id: str,
        is_aneuploid: bool = True,
        include_subtree: bool = True,
    ) -> list["Cell"]:
        """Set the aneuploid flag for the cell identified by *node_id*.

        Args:
            node_id: UUID of the target cell.
            is_aneuploid: Flag value to apply.
            include_subtree: When ``True``, propagate the flag to all
                descendants via BFS.

        Returns:
            List of :class:`Cell` objects that were modified.

        Raises:
            ValueError: If *node_id* is not found in ``id_dict``.
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            raise ValueError(f"Node id not found: {node_id!r}")
        affected = node.set_aneuploid(is_aneuploid, include_subtree)
        return self._record_affected_cells(affected, is_aneuploid)

    def get_node_by_generation_index(self, generation: int, index: int) -> "Cell":
        """Return the cell at ``generation_layers[generation][index]``.

        Args:
            generation: Generation number (0 = root).
            index: Position within that generation's layer.

        Returns:
            The matching :class:`Cell`.

        Raises:
            ValueError: If ``generation_layers`` is not initialized, or if
                *generation* / *index* are out of range or negative.
        """
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
    ) -> list["Cell"]:
        """Set the aneuploid flag for the cell at *(generation, index)*.

        Args:
            generation: Generation number (0 = root).
            index: Position within that generation's layer.
            is_aneuploid: Flag value to apply.
            include_subtree: When ``True``, propagate to all descendants.

        Returns:
            List of :class:`Cell` objects that were modified.
        """
        node = self.get_node_by_generation_index(generation, index)
        affected = node.set_aneuploid(is_aneuploid, include_subtree)
        return self._record_affected_cells(affected, is_aneuploid)


class Cell:
    """Single cell in the embryo lineage tree.

    Attributes:
        id: Unique UUID string assigned at construction.
        parent: Parent :class:`Cell`, or ``None`` for the root.
        children: Ordered list of child cells produced by division.
        generation: Division count from the root (root = 0).
        position: Cartesian unit-vector ``[x, y, z]`` on the sphere, or
            ``None`` before placement.
        layer_position: Layered coordinate ``[radius, theta, phi]`` used
            during bottom-up sphere placement, or ``None`` before placement.
        is_dead: Reserved for future early-cell-death simulation.
        is_aneuploid: ``True`` when the cell carries a chromosomal error.
    """

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
        self.is_dead = False  # For early cell deaths (To be implemented)
        self.is_aneuploid = False  # default to euploid

    def __repr__(self):  # for debug, changing the print behavior
        return (
            f"\n Cell(id={self.id[:8]}, "
            f"gen={self.generation}, "
            f"aneuploid={self.is_aneuploid}, "
            f"dead = {self.is_dead},"
            f"pos={self.position}, "
            f"layer_pos={self.layer_position})"
        )

    def set_aneuploid(
        self, is_aneuploid: bool = True, include_subtree: bool = True
    ) -> list["Cell"]:
        """Apply the aneuploid flag to this cell, optionally propagating to descendants.

        Args:
            is_aneuploid: Flag value to set.
            include_subtree: When ``True``, traverse all descendants via BFS
                and apply the same flag.

        Returns:
            List of every :class:`Cell` that was modified (this cell first,
            then descendants in BFS order when *include_subtree* is ``True``).
        """
        affected = []
        if include_subtree:
            # Breadth-first aneuploidy change through descendants.
            queue = deque([self])
            while queue:
                node = queue.popleft()
                node.is_aneuploid = is_aneuploid
                affected.append(node)
                # Continue traversal into children.
                queue.extend(node.children)
        else:
            # Just update the current cell
            self.is_aneuploid = is_aneuploid
            affected.append(self)
        # return the list of mutated cells
        return affected


# -----------------------------------------------------------------------------
# Lineage tree generation
# -----------------------------------------------------------------------------


def build_id_dict_and_layers(
    root: "Cell",
) -> tuple[dict[str, "Cell"], list[list["Cell"]]]:
    """Build a UUID lookup table and generation-indexed layer list via BFS.

    Args:
        root: Root cell of the lineage tree, or ``None`` for an empty embryo.

    Returns:
        A tuple ``(id_dict, generation_layers)`` where *id_dict* maps each
        cell's UUID to the :class:`Cell` object and *generation_layers[g]*
        lists all cells at generation *g*.
    """
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
    """Expand a root cell into a binary lineage tree without applying aneuploidy.

    Args:
        root: Seed cell for division.
        generations: Number of division rounds to simulate.
        id_dict: Pre-existing UUID → Cell map to extend in place.  A new dict
            is created when ``None``.
        generation_layers: Pre-existing generation bucket list to extend in
            place.  A new list is created when ``None``.
        include_metadata: When ``True``, append ``id_dict`` and
            ``generation_layers`` to the return value.

    Returns:
        ``(root, leaves, sibling_pairs)`` when *include_metadata* is ``False``,
        or ``(root, leaves, sibling_pairs, id_dict, generation_layers)``
        when ``True``.
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
) -> list["Cell"]:
    """Stochastically assign meiotic and mitotic aneuploidy errors via BFS.

    Meiotic error (``meio_rate``) is applied once to the root cell; mitotic
    errors (``mito_rate``) are independently tested at each internal node.
    Aneuploid status is inherited by children.

    Args:
        root: Root of the lineage tree.
        meio_rate: Probability that the root cell is aneuploid (meiotic error).
        mito_rate: Per-division probability of a new mitotic error.
        rng: Optional random generator for reproducibility.

    Returns:
        List of aneuploid :class:`Cell` objects (leaves and internal nodes)
        produced by this call.  Pass to :func:`reset_flags` to undo.
    """
    rng = _ensure_rng(rng)
    # for easy reset
    mutated_cells = []

    # apply meiotic error rate
    if rng.random() < meio_rate:
        root.is_aneuploid = True

    # Traverse all nodes to propagate inherited and new mitotic errors.
    queue = deque([root])
    # breadth-first search through the tree
    while queue:
        cell = queue.popleft()
        # skip if this is a leaf node
        if not cell.children:
            # Keep track of mutated leaves for downstream reset.
            if cell.is_aneuploid:
                mutated_cells.append(cell)
            continue

        # apply mitotic error rate
        if rng.random() < mito_rate:
            cell.is_aneuploid = True
        # extend its aneuploid status to its children
        if cell.is_aneuploid:
            mutated_cells.append(cell)
            # Children inherit parent aneuploid status
            for child in cell.children:
                child.is_aneuploid = True

        # add its children to the queue
        queue.extend(cell.children)

    return mutated_cells


def reset_flags(mutated_cells: list["Cell"]) -> None:
    """Clear the ``is_aneuploid`` flag on every cell in *mutated_cells* and empty the list.

    Args:
        mutated_cells: List returned by a previous :func:`apply_error_rates`
            call.  The list is cleared in place; pass ``[]`` for a no-op.
    """
    if not mutated_cells:
        return None
    for cell in mutated_cells:
        cell.is_aneuploid = False
    mutated_cells.clear()
    return None


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
    """Create a fresh binary lineage tree from a single root cell.

    Args:
        generations: Number of division rounds.  Passing ``0`` returns just the
            root with empty leaves and sibling pairs.
        include_metadata: When ``True``, also return the UUID lookup dict and
            generation layer list.

    Returns:
        ``(root, leaves, sibling_pairs)`` when *include_metadata* is ``False``,
        or ``(root, leaves, sibling_pairs, id_dict, generation_layers)``
        when ``True``.
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
    """Generate near-uniform spherical coordinates using the Fibonacci / golden-angle method.

    Args:
        n: Number of points.  Returns an empty ``(0, 2)`` array for ``n ≤ 0``.

    Returns:
        Array of shape ``(n, 2)`` with columns ``[theta, phi]`` in radians,
        where *theta* ∈ ``[0, 2π)`` and *phi* ∈ ``[0, π]``.
    """
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
    """Build the angular-distance cost matrix for Hungarian slot assignment.

    Args:
        child_ideal_angles: Array of shape ``(M, 2)`` with ideal ``[theta, phi]``
            targets for each child.
        fibonacci_angles: Array of shape ``(K, 2)`` with available Fibonacci
            sphere slot angles.

    Returns:
        Matrix ``C`` of shape ``(M, K)`` where ``C[i, j]`` is the angular
        distance (radians) from child *i*'s ideal to slot *j*.
    """
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
    """Assemble a fully positioned :class:`Embryo` from a tree or simulation parameters.

    Two usage modes:

    * **From parameters** — supply *generations*, *meio_rate*, and *mito_rate*
      to generate a fresh tree with stochastic error rates applied.
    * **From an existing tree** — supply *root*, *leaves*, and *sibling_pairs*
      to reuse a tree (e.g., for multi-trial sweeps).

    When leaf positions are absent, bottom-up layered placement assigns each
    leaf a Cartesian unit-vector position stored on ``cell.position``.  The
    intermediate ``[radius, theta, phi]`` layered coordinates remain accessible
    on ``cell.layer_position`` for generation-based visualization.

    Args:
        root: Root cell of an existing tree, or ``None`` to generate one.
        leaves: Leaf cells of an existing tree.
        sibling_pairs: Sibling pairs from an existing tree.
        coords: Pre-computed ``(N, 3)`` position array to attach directly,
            skipping sphere placement.
        id_dict: Pre-built UUID → Cell lookup table.
        generation_layers: Pre-built generation bucket list.
        generations: Number of division rounds (required when building from scratch).
        meio_rate: Meiotic error probability (required when building from scratch).
        mito_rate: Mitotic error probability per division (required when building
            from scratch).
        seed: Integer seed for reproducible generation + placement.
        placement_dispersal: Dispersal parameter in ``[0, 1]`` controlling how
            far sibling cells are spread from their parent on the sphere.
        rng: Existing random generator; used when *seed* is ``None``.

    Returns:
        A fully initialized :class:`Embryo` with positioned leaves.

    Raises:
        ValueError: If neither a complete tree nor all required simulation
            parameters are provided.
    """
    # Determine whether caller provided a fully realized lineage tree
    have_tree = root is not None and leaves is not None and sibling_pairs is not None
    generation_rng = None
    mutated_cells = []

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
        mutated_cells = apply_error_rates(root, meio_rate, mito_rate, rng)
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
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
