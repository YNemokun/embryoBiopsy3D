"""
Biopsy / sampling helpers for embryo surfaces.
"""

from typing import Optional

import numpy as np

from lineage_simulator import _ensure_rng, angular_distance


class Sampling:
    """Sampling helpers on the embryo surface."""

    def __init__(self, leaves, rng: Optional[np.random.Generator] = None):
        self.leaves = [cell for cell in leaves if cell.position is not None]
        if not self.leaves:
            raise ValueError("Sampling: leaves must have .position set (use build_embryo first).")
        self.rng = _ensure_rng(rng)
        coords = np.array([cell.position for cell in self.leaves])
        # cache distance by coordinates
        self._dist_matrix = self._pairwise_angular(coords)
        self._index_map = {cell: idx for idx, cell in enumerate(self.leaves)}
        self._sorted_cache = {}

    # calculate distance between two coordinated on the sphere
    def dist_on_sphere(self, point_a, point_b):
        return angular_distance(point_a, point_b)

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
    def current_biopsy(self, n_cells = 5, center_leaf=None):
        # using closeness measurement of the leaves, pick five consecutive cells from the coordinates
        if center_leaf is None:
            center_leaf = self.rng.choice(self.leaves)
        # pull from the cache
        center_idx = self._index_map[center_leaf]
        order = self._sorted_neighbors(center_idx)
        selected_cells = [self.leaves[i] for i in order[:n_cells]]

        return {"center_leaf": center_leaf, "selected": selected_cells}

    # randomly pick 5 cells slightly apart from the center cell
    def biopsy_with_distance(self, n_cells = 5, center_leaf=None, distance = 0.2):
        """
        Distance fraction: percentage of the farthest possible distance from the center.
        The returned 'selected' list includes the center cell.
        """
        if center_leaf is None:
            center_leaf = self.rng.choice(self.leaves)

        total_leaves = len(self.leaves)
        if n_cells <= 0:
            return {"center_leaf": center_leaf, "selected": [], "threshold": 0.0, "relaxed_by": 0}

        if n_cells >= total_leaves:
            selected = list(self.leaves)
            return {"center_leaf": center_leaf, "selected": selected, "threshold": 0.0, "relaxed_by": max(0, n_cells - total_leaves)}

        center = np.asarray(center_leaf.position)

        # calculate other cell's distance to the center leaf
        pairs  = []
        for cell in self.leaves:
            if cell is center_leaf:
                continue
            point = np.asarray(cell.position)
            dist = self.dist_on_sphere(center, point)
            pairs.append((dist, cell))

        # sort by distance to center leaf
        pairs.sort(key=lambda t: t[0])  
        # establish the threshold (inclusive)
        threshold = pairs[-1][0] * distance

        # number of additional cells needed beyond the center
        n_needed = max(0, n_cells - 1)

        if n_needed == 0:
            return {"center_leaf": center_leaf, "selected": [center_leaf], "threshold": threshold, "relaxed_by": 0}

        # select n_needed cells only if they are just above the threshold 
        start_index = None
        for i, (d, _) in enumerate(pairs):
            if d >= threshold:
                start_index = i
                break

        if start_index is None:
            start_index = len(pairs)
        
        select_len = len(pairs) - start_index
        
        # if there aren't enough cells to satisfy this boundary
        # relax the threshold
        if select_len >= n_needed:
            relaxed_by = 0
        else:
            relaxed_by = n_needed - select_len
            start_index = max(0, start_index - relaxed_by)
            
        selected_cells = [center_leaf] + [cell for _, cell in pairs[start_index:start_index + n_needed]]

        return {"center_leaf": center_leaf, "selected": selected_cells, "threshold": threshold,
                "relaxed_by": relaxed_by}

    def categorize_biopsy(self, biopsy_leaves):

        '''
        Categorize the biopsy into euploid (all euploid cells), mosaic (mixture of euploid and aneuploid cells), 
        or aneuploid (all aneuploid cells). Return the category and the number of aneuploid cells.
        '''
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