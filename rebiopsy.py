# This file contains the rebiopsy simulation

from biopsy import Sampling
from lineage_simulator import build_embryo, generate_tree, apply_error_rates, reset_flags
import numpy as np

DEFAULT_RELAX_STEP_FRACTION = 0.02
DEFAULT_MAX_RELAX_ATTEMPTS = 20


def _distances_to_center(center_vec, coords):
    """Vectorized angular distance to center for coords (N,3)."""
    dots = np.clip(coords @ center_vec, -1.0, 1.0)
    return np.arccos(dots)

def rebiopsy_at_error_rate(
    p_mito,
    p_meio,
    dispersal,
    distance,
    root=None,
    leaves=None,
    sibling_pairs=None,
    generations=8,
    n_trials=100,
    exp_id=None,
    *,
    coords_cache=None,
    placement_strategy="hungarian",
):
    """
    Simulate embryos at a given error rate for rebiopsy.

    Returns a list of per-trial dicts with match and metadata.
    """
    rows = []

    # TODO: add seed to make it reproducible
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
            placement_strategy=placement_strategy,
        )
        coords_cache = baseline_embryo.coords

    for i in range(n_trials):
        # TODO: add seed to the build_embryo function to make it reproducible
        # generate aneuploidy
        mutated_cells = apply_error_rates(root, meio_rate=p_meio, mito_rate=p_mito)
        # generate the embryo with the same tree, just apply the error rates
        embryo = build_embryo(
            root=root,
            leaves=leaves,
            sibling_pairs=sibling_pairs,
            placement_dispersal=dispersal,
            coords=coords_cache,
        )
        # record biopsy results with metadata
        meta = rebiopsy_single_embryo(
            embryo,
            distance,
            return_metadata=True,
        )
        meta.update({
            "p_meio": p_meio,
            "p_mito": p_mito,
            "placement_dispersal": dispersal,
            "rebiopsy_distance": distance,
            "requested_distance": distance * np.pi,
            "trial": i,
            "exp_id": exp_id,
        })
        # equal to the match indicator
        meta["concordance"] = float(meta.get("match", False))
        rows.append(meta)
        # reset for next trial
        reset_flags(mutated_cells)

    return rows

def rebiopsy_single_embryo(
    embryo,
    distance,
    return_metadata: bool = False,
    *,
    rng=None,
    seed=None,
    relax_step=None,
    max_attempts=None,
):
    '''
    Take two biopsies from the same embryo with a given distance between them

    distance: fraction of pi (0.0 to 1.0) for the target angular distance
    Return whether the two biopsies are a match (True/False).
    Optionally pass rng or seed to make sampling deterministic.
    relax_step: optional angular step in radians for relaxing distance threshold
    max_attempts: optional cap for relaxation iterations
    '''
    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)

    # do one biopsy 
    sampling_scheme = Sampling(embryo.leaves, rng=rng)
    standard_biopsy = sampling_scheme.current_biopsy()
    standard_biopsy_leaves = standard_biopsy["selected"]
    standard_biopsy_center_leaf = standard_biopsy["center_leaf"]
    # categorize it first
    standard_category, standard_aneuploid_count = sampling_scheme.categorize_biopsy(standard_biopsy_leaves)

    # in a new list of leaves, remove leaves associated with that biopsy
    new_leaves = [leaf for leaf in embryo.leaves if leaf not in standard_biopsy_leaves]
    if not new_leaves: # should not end up here
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

    sampling_scheme = Sampling(new_leaves, rng=rng)

    # do another biopsy at a given distance
    center_leaf = None
    chosen_idx = None
    threshold = distance * np.pi  # target angular distance
    relax_step = relax_step if relax_step is not None else DEFAULT_RELAX_STEP_FRACTION * np.pi
    attempts = 0
    max_attempts = max_attempts if max_attempts is not None else DEFAULT_MAX_RELAX_ATTEMPTS

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
    second_category, second_aneuploid_count = sampling_scheme.categorize_biopsy(second_biopsy_leaves)
    
    match = standard_category == second_category

    if return_metadata:
        actual_distance = float(dists[chosen_idx]) if chosen_idx is not None else sampling_scheme.dist_on_sphere(
            np.asarray(standard_biopsy_center_leaf.position),
            np.asarray(center_leaf.position),
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
    meio_range = [0.0, 1.0],
    mito_range = [0.0, 1.0],
    dispersal_range = [0.0, 0.5, 1.0],
    distance_range = [0.0, 0.5, 1.0],
    e=100,
    n_trials=100,
    generations=8,
    verbose=True,
    progress_callback=None,
    seed=None,
    placement_strategy="hungarian",
):
    '''
    Simulate rebiopsy at given error rate
    Return a dictionary of percentage of matches for each error rate and dispersal
    
    progress_callback: optional function(completed_trials, total_trials, percentage) for custom progress reporting
    seed: optional RNG seed for error rate generation
    '''
    results = []
    total_trials = len(dispersal_range) * len(distance_range) * e * n_trials
    completed_trials = 0

    def _report():
        '''
        To track the progress of the simulation
        '''
        pct = (completed_trials / total_trials * 100) if total_trials else 100.0
        if progress_callback:
            progress_callback(completed_trials, total_trials, pct)
        elif verbose:
            print(f"\rProgress: {completed_trials}/{total_trials} trials ({pct:.1f}%)", end="", flush=True)

    _report()
    # randomly generate E error rates for meiosis and mitosis
    rng = np.random.default_rng(seed)
    meio_rates = rng.uniform(meio_range[0], meio_range[1], e)
    mito_rates = rng.uniform(mito_range[0], mito_range[1], e)

    # create one tree for all embryo structures
    root, leaves, siblings = generate_tree(generations=generations)
    
    for i in range(e):
        for dispersal in dispersal_range:
            for leaf in leaves:
                leaf.position = None
            baseline_embryo = build_embryo(
                root=root,
                leaves=leaves,
                sibling_pairs=siblings,
                placement_dispersal=dispersal,
                placement_strategy=placement_strategy,
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
                    placement_strategy=placement_strategy,
                )
                results.extend(rows)
                completed_trials += n_trials
                _report()

    if verbose:
        print()  # newline after progress
    return results