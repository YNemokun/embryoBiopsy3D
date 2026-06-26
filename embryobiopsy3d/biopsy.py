"""
Biopsy and sampling helpers for embryo surfaces.

Provides the :class:`Sampling` class, which wraps a positioned leaf population
and supports distance-based biopsy selection and categorization (euploid /
mosaic / aneuploid).
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .lineage_simulator import Cell, _ensure_rng


class Sampling:
    """Sampling helpers on the embryo surface.

    Attributes:
        leaves: Positioned leaf cells (cells with ``position is not None``).
        rng: NumPy random generator used for all stochastic draws.
    """

    def __init__(self, leaves: list["Cell"], rng: Optional[np.random.Generator] = None):
        """Initialize the sampler and precompute pairwise angular distances.

        Args:
            leaves: Leaf cells from a positioned embryo.  Cells without a
                ``position`` attribute are silently dropped.
            rng: Optional random generator.  A fresh default generator is
                created when *rng* is ``None``.

        Raises:
            ValueError: If no cell in *leaves* has a position set.
        """
        self.leaves = [cell for cell in leaves if cell.position is not None]
        if not self.leaves:
            raise ValueError(
                "Sampling: leaves must have .position set (use build_embryo first)."
            )
        self.rng = _ensure_rng(rng)
        coords = np.array([cell.position for cell in self.leaves])
        # cache distance by coordinates
        self._dist_matrix = self._pairwise_angular(coords)
        self._index_map = {cell: idx for idx, cell in enumerate(self.leaves)}
        self._sorted_cache = {}

    def dist_on_sphere(
        self, point_a: NDArray[np.float64], point_b: NDArray[np.float64]
    ) -> float:
        """Return the angular distance (radians) between two 3-D unit vectors.

        Args:
            point_a: First 3-D point (need not be unit length).
            point_b: Second 3-D point (need not be unit length).

        Returns:
            Angular distance in radians in the range ``[0, π]``.
        """
        # normalize the points to unit length
        a = point_a / np.linalg.norm(point_a)
        b = point_b / np.linalg.norm(point_b)
        return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))

    @staticmethod
    def _pairwise_angular(coords: np.ndarray) -> np.ndarray:
        """Return the full symmetric pairwise angular distance matrix.

        Args:
            coords: Array of shape ``(N, 3)`` — unit vectors on the sphere.

        Returns:
            Symmetric ``(N, N)`` array of pairwise angular distances in radians,
            with zeros on the diagonal.
        """
        dots = np.clip(coords @ coords.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, 0.0)
        return dists

    def _sorted_neighbors(self, center_idx: int) -> np.ndarray:
        """Return neighbor indices sorted by angular distance to *center_idx*.

        Results are memoized in ``_sorted_cache`` on the first call.

        Args:
            center_idx: Index into ``self.leaves`` for the center cell.

        Returns:
            1-D integer array of leaf indices ordered closest-to-farthest
            (the center cell appears first with distance 0).
        """
        cached = self._sorted_cache.get(center_idx)
        if cached is not None:
            return cached
        order = np.argsort(self._dist_matrix[center_idx])
        self._sorted_cache[center_idx] = order
        return order

    def current_biopsy(
        self, n_cells: int = 5, center_leaf: Optional["Cell"] = None
    ) -> dict[str, list["Cell"]]:
        """Select the *n_cells* leaves nearest to a center leaf on the sphere.

        Args:
            n_cells: Number of cells to collect (including the center).
            center_leaf: Leaf to use as the biopsy center.  A random leaf is
                chosen when ``None``.

        Returns:
            Dict with keys:

            * ``"center_leaf"`` — the chosen center :class:`~lineage_simulator.Cell`.
            * ``"selected"`` — list of the *n_cells* closest leaves.
        """
        # using closeness measurement of the leaves, pick five consecutive cells from the coordinates
        if center_leaf is None:
            center_leaf = self.rng.choice(self.leaves)
        # pull from the cache
        center_idx = self._index_map[center_leaf]
        order = self._sorted_neighbors(center_idx)
        # selected cells include the center leaf
        selected_cells = [self.leaves[i] for i in order[:n_cells]]

        return {"center_leaf": center_leaf, "selected": selected_cells}

    def categorize_biopsy(self, biopsy_leaves: list["Cell"]) -> tuple[str, int]:
        """Categorize a biopsy sample by its aneuploid cell count.

        Args:
            biopsy_leaves: Cells in the biopsy sample.

        Returns:
            A tuple ``(category, aneuploid_count)`` where *category* is one of
            ``"euploid"``, ``"mosaic"``, or ``"aneuploid"``.

        Raises:
            ValueError: If *biopsy_leaves* is empty.
        """
        if not biopsy_leaves:
            raise ValueError("Cannot categorize biopsy: no leaves provided.")
        aneuploid_count = 0
        for leaf in biopsy_leaves:
            if leaf.is_aneuploid:
                aneuploid_count += 1

        # categorize based on the number of aneuploid cells
        if aneuploid_count == 0:
            category = "euploid"
        elif aneuploid_count == len(biopsy_leaves):
            category = "aneuploid"
        else:
            category = "mosaic"

        return category, aneuploid_count
