"""
Biopsy / sampling helpers for embryo surfaces.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .lineage_simulator import _ensure_rng


class Sampling:
    """Sampling helpers on the embryo surface."""

    def __init__(self, leaves, rng: Optional[np.random.Generator] = None):
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
        """Angular distance between two 3D points on the unit sphere."""
        # normalize the points to unit length
        a = point_a / np.linalg.norm(point_a)
        b = point_b / np.linalg.norm(point_b)
        return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))

    @staticmethod
    def _pairwise_angular(coords: np.ndarray) -> np.ndarray:
        """Compute full pairwise angular distance matrix for given coords (N,3) in the array."""
        dots = np.clip(coords @ coords.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, 0.0)
        return dists

    def _sorted_neighbors(self, center_idx: int) -> np.ndarray:
        """Return cached sorted neighbor indices (including self) for a center index. Stored in sorted_cache."""
        cached = self._sorted_cache.get(center_idx)
        if cached is not None:
            return cached
        order = np.argsort(self._dist_matrix[center_idx])
        self._sorted_cache[center_idx] = order
        return order

    # randomly pick a cluster of 5 cells
    def current_biopsy(self, n_cells=5, center_leaf=None):
        # using closeness measurement of the leaves, pick five consecutive cells from the coordinates
        if center_leaf is None:
            center_leaf = self.rng.choice(self.leaves)
        # pull from the cache
        center_idx = self._index_map[center_leaf]
        order = self._sorted_neighbors(center_idx)
        selected_cells = [self.leaves[i] for i in order[:n_cells]]

        return {"center_leaf": center_leaf, "selected": selected_cells}

    def categorize_biopsy(self, biopsy_leaves):
        """
        Categorize the biopsy into euploid (all euploid cells), mosaic (mixture of euploid and aneuploid cells),
        or aneuploid (all aneuploid cells). Return the category and the number of aneuploid cells.
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
