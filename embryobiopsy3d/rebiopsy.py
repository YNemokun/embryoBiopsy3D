"""
Rebiopsy simulation utilities.

Provides functions to simulate a second biopsy on a positioned embryo at a
controlled angular distance from the first biopsy center, then compare the two
biopsy categories (euploid / mosaic / aneuploid) for concordance.

Key entry points:

* :func:`rebiopsy_at_error_rate` — run *n* trials at fixed error rates.
* :func:`rebiopsy_single_embryo` — single paired-biopsy draw.
* :func:`simulate_experiment` — full sweep over error-rate and placement grids.
"""

from typing import Callable, Optional
from .biopsy import Sampling
from .lineage_simulator import (
    _ensure_rng,
    Cell,
    Embryo,
    build_embryo,
    generate_tree,
    apply_error_rates,
    reset_flags,
)
import numpy as np

DEFAULT_RELAX_STEP_FRACTION = 0.02
DEFAULT_MAX_RELAX_ATTEMPTS = 20


def _distances_to_center(center_vec: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Return angular distances from each row of *coords* to *center_vec*.

    Args:
        center_vec: Unit vector ``(3,)`` representing the biopsy center.
        coords: Array of shape ``(N, 3)`` of unit vectors.

    Returns:
        1-D array of length *N* with angular distances in radians.
    """
    dots = np.clip(coords @ center_vec, -1.0, 1.0)
    return np.arccos(dots)


def rebiopsy_at_error_rate(
    p_mito: float,
    p_meio: float,
    dispersal: float,
    distance: float,
    root: "Cell" = None,
    leaves: list["Cell"] = None,
    sibling_pairs: list[tuple["Cell", "Cell"]] = None,
    generations: int = 8,
    n_trials: int = 100,
    exp_id: int = None,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    coords_cache: np.ndarray = None,
) -> list[dict]:
    """Run *n_trials* paired-biopsy simulations at fixed error rates.

    Args:
        p_mito: Per-division mitotic error probability.
        p_meio: Meiotic error probability applied to the root cell.
        dispersal: Placement dispersal parameter in ``[0, 1]``.
        distance: Target angular distance between biopsies as a fraction of π.
        root: Root cell of a pre-built tree.  A new tree is generated when
            ``None``.
        leaves: Leaf cells of the pre-built tree.
        sibling_pairs: Sibling pairs of the pre-built tree.
        generations: Number of division generations (used when building a
            fresh tree).
        n_trials: Number of trials to run.
        exp_id: Experiment identifier stored in each result row.
        rng: Shared random generator; pass for reproducible multi-experiment
            sweeps.
        seed: Integer seed to construct a fresh generator (takes precedence
            over *rng*).
        coords_cache: Pre-computed ``(N, 3)`` leaf position array.  When
            supplied, sphere placement is skipped for all trials.

    Returns:
        List of per-trial result dicts, each containing ``match``,
        ``concordance``, ``p_meio``, ``p_mito``, ``placement_dispersal``,
        ``rebiopsy_distance``, ``trial``, ``exp_id``, and all metadata keys
        returned by :func:`rebiopsy_single_embryo`.
    """
    rows = []

    rng = _ensure_rng(rng, seed)
    # simulate one embryo lineage tree
    if root is None or leaves is None or sibling_pairs is None:
        root, leaves, sibling_pairs = generate_tree(generations=generations)
    # lazily build the baseline if coords were not supplied
    if coords_cache is None:
        baseline_embryo = build_embryo(
            root=root,
            leaves=leaves,
            sibling_pairs=sibling_pairs,
            placement_dispersal=dispersal,
            rng=rng,
        )
        coords_cache = baseline_embryo.coords

    for i in range(n_trials):
        # generate aneuploidy
        mutated_cells = apply_error_rates(
            root, meio_rate=p_meio, mito_rate=p_mito, rng=rng
        )
        # generate the embryo with the same tree, just apply the error rates
        embryo = build_embryo(
            root=root,
            leaves=leaves,
            sibling_pairs=sibling_pairs,
            placement_dispersal=dispersal,
            coords=coords_cache,
            rng=rng,
        )
        # record biopsy results with metadata
        meta = rebiopsy_single_embryo(
            embryo,
            distance,
            return_metadata=True,
            rng=rng,
        )
        meta.update(
            {
                "p_meio": p_meio,
                "p_mito": p_mito,
                "placement_dispersal": dispersal,
                "rebiopsy_distance": distance,
                "requested_distance": distance * np.pi,
                "trial": i,
                "exp_id": exp_id,
            }
        )
        # equal to the match indicator
        meta["concordance"] = float(meta.get("match", False))
        rows.append(meta)
        # reset for next trial
        reset_flags(mutated_cells)

    return rows


def rebiopsy_single_embryo(
    embryo: "Embryo",
    distance: float,
    return_metadata: bool = False,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    relax_step: float = None,
    max_attempts: int = None,
) -> bool | dict:
    """Take two biopsies from an embryo separated by a target angular distance.

    The first biopsy center is chosen randomly.  The second center is the
    closest available leaf that is at least ``distance × π`` radians away.  If
    no leaf satisfies the threshold, it is relaxed by *relax_step* radians up to
    *max_attempts* times, then falls back to the farthest available leaf.

    Args:
        embryo: Positioned :class:`~lineage_simulator.Embryo` to sample.
        distance: Target angular separation as a fraction of π (``0.0``–``1.0``).
        return_metadata: When ``True``, return a detailed result dict instead of
            the bare boolean.
        rng: Shared random generator for reproducibility.
        seed: Integer seed to construct a fresh generator.
        relax_step: Angular step (radians) by which the distance threshold is
            relaxed on each failed attempt.  Defaults to
            ``DEFAULT_RELAX_STEP_FRACTION × π``.
        max_attempts: Maximum number of relaxation iterations before falling
            back to the farthest leaf.  Defaults to
            ``DEFAULT_MAX_RELAX_ATTEMPTS``.

    Returns:
        ``True``/``False`` concordance flag when *return_metadata* is ``False``.
        When ``True``, a dict with keys ``match``, ``standard_category``,
        ``second_category``, ``standard_aneuploid_count``,
        ``second_aneuploid_count``, ``standard_center``, ``second_center``,
        ``standard_leaves``, ``second_leaves``, ``requested_distance``,
        ``actual_distance``, and optionally ``error``.
    """
    rng = _ensure_rng(rng, seed)

    # do one biopsy
    sampling_scheme = Sampling(embryo.leaves, rng=rng)
    standard_biopsy = sampling_scheme.current_biopsy()
    standard_biopsy_leaves = standard_biopsy["selected"]
    standard_biopsy_center_leaf = standard_biopsy["center_leaf"]
    # categorize it first
    standard_category, standard_aneuploid_count = sampling_scheme.categorize_biopsy(
        standard_biopsy_leaves
    )

    # in a new list of leaves, remove leaves associated with that biopsy
    new_leaves = [leaf for leaf in embryo.leaves if leaf not in standard_biopsy_leaves]
    if not new_leaves:  # should not end up here
        # No cells left to sample; abort this trial
        if return_metadata:
            return {
                "match": False,
                "standard_category": standard_category,
                "aneuploid_count": standard_aneuploid_count,
                "standard_aneuploid_count": standard_aneuploid_count,
                "second_category": None,
                "second_aneuploid_count": None,
                "standard_center": standard_biopsy_center_leaf,
                "second_center": None,
                "standard_leaves": standard_biopsy_leaves,
                "second_leaves": [],
                "requested_distance": distance * np.pi,
                "actual_distance": None,
                "error": "no remaining cells for rebiopsy",
            }
        return False

    # controlled randomness
    sampling_scheme = Sampling(new_leaves, rng=rng)

    # do another biopsy at a given distance
    center_leaf = None
    chosen_idx = None
    threshold = distance * np.pi  # target angular distance
    relax_step = (
        relax_step if relax_step is not None else DEFAULT_RELAX_STEP_FRACTION * np.pi
    )
    attempts = 0
    max_attempts = (
        max_attempts if max_attempts is not None else DEFAULT_MAX_RELAX_ATTEMPTS
    )

    coords = np.array([leaf.position for leaf in new_leaves])
    center_vec = np.asarray(standard_biopsy_center_leaf.position)
    # an array of distances to the center leaf for all leaves
    dists = _distances_to_center(center_vec, coords)

    while center_leaf is None and attempts < max_attempts:
        # find leaves that are at least at the target distance
        eligible_idx = np.nonzero(dists >= threshold)[0]
        if eligible_idx.size:
            eligible_dists = dists[eligible_idx]
            # pick the leaf with the smallest distance that still satisfies the threshold
            local_min_idx = int(np.argmin(eligible_dists))
            chosen_idx = int(eligible_idx[local_min_idx])
            center_leaf = new_leaves[chosen_idx]
            break
        threshold = max(0.0, threshold - relax_step)
        attempts += 1

    if center_leaf is None:
        # fallback to farthest leaf if nothing met the relaxed threshold
        chosen_idx = int(np.argmax(dists))
        center_leaf = new_leaves[chosen_idx]

    second_biopsy = sampling_scheme.current_biopsy(center_leaf=center_leaf)
    second_biopsy_leaves = second_biopsy["selected"]
    second_category, second_aneuploid_count = sampling_scheme.categorize_biopsy(
        second_biopsy_leaves
    )

    match = standard_category == second_category

    if return_metadata:
        actual_distance = (
            float(dists[chosen_idx])
            if chosen_idx is not None
            else sampling_scheme.dist_on_sphere(
                np.asarray(standard_biopsy_center_leaf.position),
                np.asarray(center_leaf.position),
            )
        )
        return {
            "match": match,
            "standard_category": standard_category,
            "aneuploid_count": standard_aneuploid_count,
            "standard_aneuploid_count": standard_aneuploid_count,
            "second_category": second_category,
            "second_aneuploid_count": second_aneuploid_count,
            "standard_center": standard_biopsy_center_leaf,
            "second_center": center_leaf,
            "standard_leaves": standard_biopsy_leaves,
            "second_leaves": second_biopsy_leaves,
            "requested_distance": distance * np.pi,
            "actual_distance": actual_distance,
        }

    # return whether the two biopsies are a match
    return match


def simulate_experiment(
    meio_range: list[float] = [0.0, 1.0],
    mito_range: list[float] = [0.0, 1.0],
    dispersal_range: list[float] = [0.0, 0.5, 1.0],
    distance_range: list[float] = [0.0, 0.5, 1.0],
    e: int = 100,
    n_trials: int = 100,
    generations: int = 8,
    verbose: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Run a full sweep of rebiopsy simulations over error-rate and placement grids.

    Samples *e* random (meio_rate, mito_rate) pairs uniformly from
    *meio_range* × *mito_range*, then for each pair runs *n_trials* biopsies
    at every combination of *dispersal_range* × *distance_range*.

    Args:
        meio_range: ``[low, high]`` uniform sampling bounds for meiotic error rate.
        mito_range: ``[low, high]`` uniform sampling bounds for mitotic error rate.
        dispersal_range: List of placement dispersal values to sweep.
        distance_range: List of biopsy separation distances (fraction of π) to sweep.
        e: Number of (meio_rate, mito_rate) pairs to draw.
        n_trials: Number of paired-biopsy trials per parameter combination.
        generations: Lineage tree depth.
        verbose: When ``True`` and *progress_callback* is ``None``, print a
            progress line to stdout.  A trailing newline is emitted on completion.
        progress_callback: Optional callable ``(completed, total, pct)`` invoked
            after each distance-block completes.  Suppresses verbose output when
            provided.
        rng: Shared random generator.  Construct one from *seed* when ``None``.
        seed: Integer seed for reproducible experiments.

    Returns:
        List of per-trial result dicts from :func:`rebiopsy_at_error_rate`,
        one entry per trial across all parameter combinations.
    """
    rng = _ensure_rng(rng, seed)
    results = []
    total_trials = len(dispersal_range) * len(distance_range) * e * n_trials
    completed_trials = 0

    def _report():
        """
        To track the progress of the simulation
        """
        pct = (completed_trials / total_trials * 100) if total_trials else 100.0
        if progress_callback:
            progress_callback(completed_trials, total_trials, pct)
        elif verbose:
            print(
                f"\rProgress: {completed_trials}/{total_trials} trials ({pct:.1f}%)",
                end="",
                flush=True,
            )

    _report()
    # randomly generate E error rates for meiosis and mitosis
    meio_rates = rng.uniform(meio_range[0], meio_range[1], e)
    mito_rates = rng.uniform(mito_range[0], mito_range[1], e)

    # create one tree for all embryo structures
    root, leaves, siblings = generate_tree(generations=generations)

    for i in range(e):
        for dispersal in dispersal_range:
            for leaf in leaves:
                leaf.position = None
            # cache the embryo structure to avoid rebuilding it for each trial
            baseline_embryo = build_embryo(
                root=root,
                leaves=leaves,
                sibling_pairs=siblings,
                placement_dispersal=dispersal,
                rng=rng,
            )
            coords_cache = baseline_embryo.coords

            for distance in distance_range:
                rows = rebiopsy_at_error_rate(
                    p_mito=mito_rates[i],
                    p_meio=meio_rates[i],
                    dispersal=dispersal,
                    distance=distance,
                    root=root,
                    leaves=leaves,
                    sibling_pairs=siblings,
                    n_trials=n_trials,
                    generations=generations,
                    exp_id=i,
                    coords_cache=coords_cache,
                    rng=rng,
                )
                results.extend(rows)
                completed_trials += n_trials
                _report()

    if verbose:
        print()  # newline after progress
    return results
