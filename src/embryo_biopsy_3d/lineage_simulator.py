"""
Lineage and spatial placement utilities for embryo simulation.

This module builds a binary lineage tree simulating cell division,
annotates aneuploidy, and places leaf cells on a sphere for downstream
biopsy and visualization workflows.
"""

import random
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


def _ensure_rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
    """Return a numpy Generator, optionally seeded."""
    if rng is not None:
        return rng
    # If no seed is provided, fall back to an unseeded generator for randomness
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def angular_distance(point_a, point_b) -> float:
    """Angular distance between two 3D points on a sphere."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    # Normalize vectors to unit length before taking dot products.
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    # Clip against floating-point drift before arccos.
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class Embryo:
    """Embryo class for the simulated embryo state."""

    root: "Cell"
    leaves: list["Cell"]
    sibling_pairs: list[tuple["Cell", "Cell"]]
    coords: Optional[np.ndarray] = None
    placement_dispersal: Optional[float] = None
    generation_rng: Optional[np.random.Generator] = None
    mutated_cells: Optional[list["Cell"]] = None
    id_dict: Optional[dict[str, "Cell"]] = None
    generation_layers: Optional[list[list["Cell"]]] = None

    def get_node_by_id(self, node_id):
        """Return the node matching the given id."""
        if not self.id_dict:
            raise ValueError("Embryo id_dict is not initialized.")
        return self.id_dict.get(node_id)

    def set_aneuploid_by_id(
        self,
        node_id,
        is_aneuploid: bool = True,
        include_subtree: bool = True,
    ):
        """Set aneuploid status for a node id, optionally including its subtree."""
        node = self.get_node_by_id(node_id)
        if node is None:
            raise ValueError(f"Node id not found: {node_id!r}")
        return node.set_aneuploid(is_aneuploid, include_subtree)

    def get_node_by_generation_index(self, generation, index):
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
        generation,
        index,
        is_aneuploid: bool = True,
        include_subtree: bool = True,
    ):
        """Set aneuploid status for a (generation, index) node, optionally its subtree."""
        node = self.get_node_by_generation_index(generation, index)
        return node.set_aneuploid(is_aneuploid, include_subtree)


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
        self.is_dead = False  # For early cell deaths
        self.is_aneuploid = False  # default to euploid

    def __repr__(self):  # for debug, changing the print behavior
        return (
            f"\n Cell(id={self.id[:8]}, "
            f"gen={self.generation}, "
            f"aneuploid={self.is_aneuploid}, "
            f"dead = {self.is_dead},"
            f"children={self.children}, "
            f"pos={self.position}, "
            f"layer_pos={self.layer_position})"
        )

    def set_aneuploid(self, is_aneuploid: bool = True, include_subtree: bool = True):
        """Set aneuploid status on this cell (optionally its subtree)."""
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
        return affected


# -----------------------------------------------------------------------------
# Lineage tree generation
# -----------------------------------------------------------------------------


def build_id_dict_and_layers(root):
    """Return (id_dict, generation_layers) for the given root (BFS)."""
    # Empty root means no nodes and no generation buckets.
    if root is None:
        return {}, []
    # id_dict: fast lookup by UUID
    # generation_layers: index leaves by generation
    id_dict = {}
    generation_layers = []
    queue = deque([root])
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
    root,
    generations,
    *,
    id_dict=None,
    generation_layers=None,
):
    """Return id- and generation-indexed containers for the requested number of generations."""
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
    root,
    generations,
    id_dict=None,
    generation_layers=None,
    *,
    include_metadata: bool = False,
):
    """Generate cell divisions without applying aneuploidy.

    If include_metadata is True, returns id_dict and generation_layers as well.
    """
    id_dict, generation_layers = _initialize_generation_metadata(
        root,
        generations,
        id_dict=id_dict,
        generation_layers=generation_layers,
    )
    if generations <= 0:
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
            # TODO Later: Cell death
            # if parent.is_dead:
            #     # Dead leaves are carried forward unchanged.
            #     next_generation.append(parent)
            #     continue

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
    root, meio_rate, mito_rate, rng: Optional[np.random.Generator] = None
):
    """Assign meiotic/mitotic errors on an existing tree structure (BFS)."""
    rng = _ensure_rng(rng)
    # for easy reset
    mutated_cells = []

    # apply meiotic error rate
    if rng.random() < meio_rate:
        root.is_aneuploid = True

    # Traverse all nodes to propagate inherited and new mitotic errors.
    queue = deque([root])
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


def reset_flags(mutated_cells):
    """Reset aneuploid flags on the provided cells."""
    if not mutated_cells:
        return None
    for cell in mutated_cells:
        cell.is_aneuploid = False
    mutated_cells.clear()
    return None


def generate_tree(generations, *, include_metadata: bool = False):
    """Generate a binary lineage tree for a fixed number of generations.

    If include_metadata is True, returns id_dict and generation_layers as well.
    """
    # Initialize a single-cell embryo root.
    root = Cell(parent=None, generation=0)
    leaves = []
    sibling_pairs = []
    id_dict, generation_layers = _initialize_generation_metadata(root, generations)
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


def coordinates_generate(n):
    """Generate near-uniform Cartesian coordinates on the unit sphere."""
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    # Golden-angle spacing for near-uniform points on a sphere
    golden_ratio = (1 + 5**0.5) / 2
    i = arange(n)
    theta = 2 * pi * i / golden_ratio
    phi = arccos(1 - 2 * (i + 0.5) / n)
    # From spherical coordinates to unit vectors.
    x = cos(theta) * sin(phi)
    y = sin(theta) * sin(phi)
    z = cos(phi)
    return np.c_[x, y, z]


def coordinates_generate_radians(n):
    """Generate near-uniform spherical coordinates as (theta, phi) radians."""
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    golden_ratio = (1 + 5**0.5) / 2
    i = arange(n)
    theta = (2 * pi * i / golden_ratio) % (2 * pi)
    phi = arccos(1 - 2 * (i + 0.5) / n)
    return np.c_[theta, phi]


def _angles_to_cartesian(theta, phi, radius):
    """Convert spherical (theta, phi) to cartesian (x, y, z) at given radius."""
    x = radius * cos(theta) * sin(phi)
    y = radius * sin(theta) * sin(phi)
    z = radius * cos(phi)
    return np.asarray([x, y, z], dtype=float)


def _cartesian_to_angles(x, y, z):
    """Convert cartesian (x,y,z) to spherical (theta, phi) on unit sphere. Returns (theta, phi)."""
    r = np.sqrt(x * x + y * y + z * z)
    if r <= 0:
        return 0.0, 0.0
    phi = float(np.arccos(np.clip(z / r, -1.0, 1.0)))
    theta = float(np.arctan2(y, x))
    if theta < 0:
        theta += 2 * pi
    return theta, phi


# -----------------------------------------------------------------------------
# Lineage ordering and placement helpers
# -----------------------------------------------------------------------------


def _lineage_distance(cell_a, cell_b) -> int:
    """Tree distance between two cells using parent pointers."""
    if cell_a is cell_b:
        return 0
    a, b = cell_a, cell_b
    # number of edges needed to travel from a to b
    dist = 0
    # Bring both nodes to the same generation depth.
    while a.generation > b.generation:
        a = a.parent
        dist += 1
    while b.generation > a.generation:
        b = b.parent
        dist += 1
    # Then climb together until the pointers meet.
    while a is not b:
        if a is None or b is None:
            raise ValueError("Cells do not share a common ancestor.")
        a = a.parent
        b = b.parent
        dist += 2
    if a is None:
        raise ValueError("Cells do not share a common ancestor.")
    return dist


def _wrap_theta(theta):
    """Wrap azimuth  (x-axis) to [0, 2pi)."""
    return float(theta % (2 * pi))


def _reflect_phi(phi):
    """Reflect polar angle (z-axis) into [0, pi] while preserving continuity."""
    value = float(phi)
    two_pi = 2.0 * float(pi)
    value = value % two_pi
    # reflect the angle around the equator (z-axis)
    if value > float(pi):
        value = two_pi - value
    return value


def _perturb_angles_from_parent(theta, phi, alpha, beta):
    """Return a child ideal angle from parent by angle-space perturbation."""
    # Keep finite dtheta near poles where longitude is numerically unstable.
    sin_phi = float(np.sin(phi))
    denom = max(abs(sin_phi), 1e-3)
    # calculate perturbation for each angle
    dphi = alpha * float(np.cos(beta))
    dtheta = alpha * float(np.sin(beta)) / denom
    # keep the angles within the valid range
    child_theta = _wrap_theta(theta + dtheta)
    child_phi = _reflect_phi(phi + dphi)
    return np.asarray([child_theta, child_phi], dtype=float)


def build_cost_matrix(child_ideal_angles, fibonacci_angles):
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


def _convert_layered_positions_to_cartesian_unit_sphere(cells):
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


def _validate_positioning_inputs(generation_layers, dispersal):
    """Validate shared inputs for bottom-up placement."""
    if generation_layers is None:
        raise ValueError("generation_layers is required for bottom-up positioning")
    if not 0 <= dispersal <= 1:
        raise ValueError("dispersal must be between 0 and 1 (inclusive)")


def _clear_generation_positions(generation_layers):
    """Clear cached node positions before rebuilding a layered placement."""
    for layer in generation_layers:
        for node in layer:
            node.position = None
            node.layer_position = None


def _assign_direct_layer_positions(current_layer, angles, radius):
    """Assign Fibonacci slots directly for the first two generations."""
    for node, angle in zip(current_layer, angles):
        node.layer_position = [radius, angle[0], angle[1]]


def _generation_targets(
    previous_layer,
    generation,
    total_layers,
    alpha,
    rng: Optional[np.random.Generator] = None,
):
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
                # evenly space the children around the parent
                beta = beta0 + (2.0 * pi * (child_idx / max(1, child_count)))
            child_ideal_angles.append(
                _perturb_angles_from_parent(
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


def _assign_hungarian_slots(next_layer, child_ideal_angles, angles, radius):
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


def _assign_greedy_slots(next_layer, child_ideal_angles, angles, radius, randomizer):
    """Assign each child to the nearest, currently unoccupied, and ideal slot."""
    if len(next_layer) != len(angles):
        raise ValueError("Mismatch between child count and available angle slots.")

    child_order = list(range(len(next_layer)))
    randomizer.shuffle(child_order)

    # reuse the cost matrix instead of recalculating it for each child
    cost_matrix = build_cost_matrix(child_ideal_angles, angles)

    for child_idx in child_order:
        slot_idx = int(np.argmin(cost_matrix[child_idx]))
        if not np.isfinite(cost_matrix[child_idx, slot_idx]):
            # should not happen
            raise ValueError("No available slot found for child placement.")

        next_layer[child_idx].layer_position = [
            radius,
            angles[slot_idx][0],
            angles[slot_idx][1],
        ]

        # mark this slot unavailable for all remaining children.
        cost_matrix[:, slot_idx] = np.inf


def _run_bottom_up_positioning(
    generation_layers, dispersal, *, placement_strategy, rng=None
):
    """Shared implementation for layered bottom-up leaf placement."""
    _validate_positioning_inputs(generation_layers, dispersal)
    _clear_generation_positions(generation_layers)

    randomizer = None
    rng = _ensure_rng(rng)
    randomizer = random.Random(int(rng.integers(0, 2**31 - 1)))

    sibling_pairs = []
    total_layers = len(generation_layers)

    # add a bit of jitter to the dispersal direction to avoid perfect alignment
    alpha = float(dispersal) * (pi / 2) + rng.uniform(0, 0.05)

    # assign positions to each generation's leaves
    for generation, current_layer in enumerate(generation_layers):
        angles = coordinates_generate_radians(len(current_layer))
        radius = float(generation + 1)

        if generation <= 1:
            _assign_direct_layer_positions(current_layer, angles, radius)
            continue

        # generate child ideals for the current generation
        next_layer, child_ideal_angles, generation_sibling_pairs = _generation_targets(
            generation_layers[generation - 1],
            generation,
            total_layers,
            alpha,
            rng=rng,
        )

        if placement_strategy == "hungarian":
            _assign_hungarian_slots(next_layer, child_ideal_angles, angles, radius)
        elif placement_strategy == "greedy":
            _assign_greedy_slots(
                next_layer,
                child_ideal_angles,
                angles,
                radius,
                randomizer,
            )
        else:
            raise ValueError(
                "placement_strategy must be either 'hungarian' or 'greedy'."
            )

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
    leaves,
    sibling_pairs,
    coords,
    placement_dispersal,
    generation_layers,
    placement_strategy="hungarian",
    rng: Optional[np.random.Generator] = None,
):
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
        # Use layered bottom-up placement.
        if placement_strategy == "hungarian":
            ordered_leaves, layer_coords, sibling_pairs = _bottom_up_position_leaves(
                generation_layers=generation_layers,
                dispersal=placement_dispersal,
                rng=rng,
            )
        elif placement_strategy == "greedy":
            ordered_leaves, layer_coords, sibling_pairs = (
                _bottom_up_position_leaves_greedy(
                    generation_layers=generation_layers,
                    dispersal=placement_dispersal,
                    rng=rng,
                )
            )
        else:
            raise ValueError(
                "placement_strategy must be either 'hungarian' or 'greedy'."
            )
        _convert_layered_positions_to_cartesian_unit_sphere(ordered_leaves)
        coords_array = np.array([leaf.position for leaf in ordered_leaves], dtype=float)
    elif coords_array is None:
        coords_array = np.array([leaf.position for leaf in ordered_leaves], dtype=float)

    return ordered_leaves, coords_array, sibling_pairs


def _bottom_up_position_leaves(
    generation_layers, dispersal, rng: Optional[np.random.Generator] = None
):
    """Assign coordinates per generation using Hungarian algorithm for parent-local children.

    - Radius stays fixed per generation: radius = generation + 1.
    - Child ideals are generated by angle-space perturbations from each parent.
    - Hungarian assignment maps those ideals to available Fibonacci slots.
    """
    return _run_bottom_up_positioning(
        generation_layers,
        dispersal,
        placement_strategy="hungarian",
        rng=rng,
    )


def _bottom_up_position_leaves_greedy(
    generation_layers,
    dispersal,
    rng: Optional[np.random.Generator] = None,
):
    """Assign coordinates per generation using greedy nearest-slot assignment.

    This keeps the same child ideal construction used by Hungarian placement, but
    assigns each child to the currently nearest unoccupied Fibonacci slot in a
    randomized child order.
    """
    return _run_bottom_up_positioning(
        generation_layers,
        dispersal,
        placement_strategy="greedy",
        rng=rng,
    )


# -----------------------------------------------------------------------------
# Embryo construction
# -----------------------------------------------------------------------------


def build_embryo(
    root=None,
    leaves=None,
    sibling_pairs=None,
    coords=None,
    id_dict=None,
    generation_layers=None,
    *,
    generations=None,
    meio_rate=None,
    mito_rate=None,
    seed=None,
    placement_dispersal=0.0,
    placement_strategy="hungarian",
    rng: Optional[np.random.Generator] = None,
):
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
        rng = _ensure_rng(rng, seed=seed)
        if id_dict is None or generation_layers is None:
            id_dict, generation_layers = build_id_dict_and_layers(root)

    ordered_leaves, coords, sibling_pairs = _position_leaves(
        leaves=leaves,
        sibling_pairs=sibling_pairs,
        coords=coords,
        placement_dispersal=placement_dispersal,
        generation_layers=generation_layers,
        placement_strategy=placement_strategy,
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
