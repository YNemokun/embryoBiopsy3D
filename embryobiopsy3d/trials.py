"""
Generate fixed-subtree rebiopsy trial data for selected divisions.

This module runs one full per-trial table and one compact 3x3 transition-probability
summary, sweeping over (generation index, dispersal, distance) with
a configurable number of trials per cell.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import os
import time

import numpy as np

from .lineage_simulator import build_embryo, generate_tree
from .rebiopsy import rebiopsy_single_embryo

# Default sweep grids and run settings (used when callers pass None to run_analysis).
DEFAULT_DISTANCE_VALUES = [0.0, 0.5, 1.0]
DEFAULT_GENERATION_INDEX_VALUES = list(range(8))
DEFAULT_DISPERSAL_VALUES = [0.0, 1.0]
DEFAULT_GENERATIONS = 8
DEFAULT_N_TRIALS = 1000
DEFAULT_OUT_DIR = "data/all_generations"
DEFAULT_BASE_SEED = 7
DEFAULT_CELL_INDEX = 0  # cell_index picks which leaf marks error.

# Ordered biopsy categorization
BIOPSY_CATEGORIES = ["euploid", "mosaic", "aneuploid"]
# Per-trial CSV schema. One row per (division, dispersal, distance) replicate.
TRIAL_FIELDNAMES = [
    "seed_index",
    "seed",
    "trial_within_group",
    "division",
    "cell_index",
    "aneuploid_leaf_count",
    "dispersal",
    "distance",
    "match",
    "standard_category",
    "standard_aneuploid_count",
    "second_category",
    "second_aneuploid_count",
    "actual_distance",
]


def _clear_tree_state(generation_layers: list[list]) -> None:
    """Clear aneuploid flags and cached positions before a new trial."""
    for layer in generation_layers:
        for node in layer:
            node.is_aneuploid = False
            node.position = None
            node.layer_position = None


def _aneuploid_leaf_count(total_generations: int, generation_index: int) -> int:
    """Return the number of aneuploid leaves in the marked subtree."""
    # error at generation g produces 2^(8-g) aneuploid leaves
    return 2 ** (total_generations - generation_index)


def _save_csv(path: str, rows: list[dict]) -> None:
    """Save rows to CSV."""
    if not rows:
        # write file so consumers see a real path.
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_summary_rows_from_counts(
    transition_counts: dict,
    standard_totals: dict,
    group_totals: dict,
) -> list[dict]:
    """
    Build the grouped 3x3 conditional probability summary from running counts.

    For each (division, dispersal, distance), this returns the full 3x3 table over:
    - standard biopsy category
    - second biopsy category

    P(second_category | standard_category, division, dispersal, distance)
    """
    summary_rows = []
    # group_totals: one count per (division, cell, leaf count, dispersal, distance).
    for group_key in sorted(group_totals.keys()):
        division, cell_index, aneuploid_leaf_count, dispersal, distance = group_key
        n_trials = group_totals[group_key]

        # Condition on the first-biopsy category
        for standard_category in BIOPSY_CATEGORIES:
            standard_key = group_key + (standard_category,)
            standard_total = standard_totals.get(standard_key, 0)

            for second_category in BIOPSY_CATEGORIES:
                transition_key = standard_key + (second_category,)
                transition_count = transition_counts.get(transition_key, 0)
                # P(second | first) = cell count / total trials with that first category.
                conditional_probability = (
                    transition_count / standard_total
                    if standard_total
                    else float("nan")
                )
                # P(first, second) in this (division, dispersal, distance) group.
                joint_probability = transition_count / n_trials

                summary_rows.append(
                    {
                        "division": division,
                        "cell_index": cell_index,
                        "aneuploid_leaf_count": aneuploid_leaf_count,
                        "dispersal": dispersal,
                        "distance": distance,
                        "n_trials": n_trials,
                        "standard_category": standard_category,
                        "second_category": second_category,
                        "transition_count": transition_count,
                        "standard_total": standard_total,
                        "conditional_probability": float(conditional_probability),
                        "joint_probability": float(joint_probability),
                    }
                )

    return summary_rows


def run_analysis(
    *,
    generations: int = DEFAULT_GENERATIONS,
    n_trials: int = DEFAULT_N_TRIALS,
    generation_index_values: list[int] = None,
    dispersal_values: list[float] = None,
    distance_values: list[float] = None,
    base_seed: int = DEFAULT_BASE_SEED,
    cell_index: int = DEFAULT_CELL_INDEX,
    out_dir: str = DEFAULT_OUT_DIR,
) -> tuple[list[dict], list[dict]]:
    """Generate all trial rows and the grouped transition summary."""
    if generation_index_values is None:
        generation_index_values = list(DEFAULT_GENERATION_INDEX_VALUES)
    if dispersal_values is None:
        dispersal_values = list(DEFAULT_DISPERSAL_VALUES)
    if distance_values is None:
        distance_values = list(DEFAULT_DISTANCE_VALUES)

    if generations < 1:
        raise ValueError("generations must be positive.")
    if n_trials < 1:
        raise ValueError("n-trials must be positive.")
    if cell_index < 0:
        raise ValueError("cell-index must be non-negative.")
    if len(generation_index_values) == 0:
        raise ValueError("generation-index-values must contain at least one value.")
    if len(dispersal_values) == 0:
        raise ValueError("dispersal-values must contain at least one value.")
    if len(distance_values) == 0:
        raise ValueError("distance-values must contain at least one value.")

    for generation_index in generation_index_values:
        if generation_index < 0:
            raise ValueError("generation indices must be non-negative.")
        if generation_index > generations:
            raise ValueError("generation indices cannot exceed total generations.")
    for dispersal in dispersal_values:
        if not 0.0 <= dispersal <= 1.0:
            raise ValueError("dispersal values must be between 0 and 1.")
    for distance in distance_values:
        if not 0.0 <= distance <= 1.0:
            raise ValueError("distance values must be between 0 and 1.")

    os.makedirs(out_dir, exist_ok=True)

    # Derives a long stream of trial seeds to reproduce same trial CSVs.
    seed_sequence = np.random.default_rng(base_seed)
    # placement and rebiopsy use the same tree across all sweeps
    root, leaves, siblings, id_dict, generation_layers = generate_tree(
        generations=generations,
        include_metadata=True,
    )
    transition_counts: dict = defaultdict(int)
    standard_totals: dict = defaultdict(int)
    group_totals: dict = defaultdict(int)
    trial_csv_path = os.path.join(out_dir, "rebiopsy_trials.csv")
    summary_csv_path = os.path.join(out_dir, "rebiopsy_transition_summary.csv")

    # Cartesian product of sweeps times n_trials replicates per cell
    total_trials = (
        len(generation_index_values)
        * len(dispersal_values)
        * len(distance_values)
        * n_trials
    )
    # sanity check: must match total_trials at end.
    trial_counter = 0

    with open(trial_csv_path, "w", newline="", encoding="utf-8") as handle:
        trial_writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDNAMES)
        trial_writer.writeheader()

        for generation_index in generation_index_values:
            aneuploid_leaf_count = _aneuploid_leaf_count(generations, generation_index)
            for dispersal in dispersal_values:
                for distance in distance_values:
                    for trial in range(n_trials):
                        # New seed for placement + error pattern + biopsy draws this trial
                        seed = int(seed_sequence.integers(0, 2**31 - 1))

                        _clear_tree_state(generation_layers)
                        # Hungarian sphere placement with dispersal
                        embryo = build_embryo(
                            root=root,
                            leaves=leaves,
                            sibling_pairs=siblings,
                            id_dict=id_dict,
                            generation_layers=generation_layers,
                            placement_dispersal=dispersal,
                            seed=seed,
                        )
                        # Mark the subtree at (generation_index, cell_index) as aneuploid.
                        embryo.set_aneuploid_by_generation_index(
                            generation_index,
                            cell_index,
                            is_aneuploid=True,
                            include_subtree=True,
                        )

                        # First biopsy then second at ~distance*pi
                        meta = rebiopsy_single_embryo(
                            embryo,
                            distance,
                            return_metadata=True,
                            seed=seed,
                        )

                        row = {
                            "seed_index": trial_counter,
                            "seed": seed,
                            "trial_within_group": trial,
                            "division": generation_index,
                            "cell_index": cell_index,
                            "aneuploid_leaf_count": aneuploid_leaf_count,
                            "dispersal": dispersal,
                            "distance": distance,
                            "match": meta.get("match"),
                            "standard_category": meta.get("standard_category"),
                            "standard_aneuploid_count": meta.get(
                                "standard_aneuploid_count"
                            ),
                            "second_category": meta.get("second_category"),
                            "second_aneuploid_count": meta.get(
                                "second_aneuploid_count"
                            ),
                            "actual_distance": meta.get("actual_distance"),
                        }
                        trial_writer.writerow(row)

                        group_key = (
                            generation_index,
                            cell_index,
                            aneuploid_leaf_count,
                            dispersal,
                            distance,
                        )
                        standard_key = group_key + (row["standard_category"],)
                        transition_key = standard_key + (row["second_category"],)

                        group_totals[group_key] += 1
                        standard_totals[standard_key] += 1
                        transition_counts[transition_key] += 1
                        trial_counter += 1

    if trial_counter != total_trials:
        raise ValueError("Generated trial count does not match the expected total.")

    # Expand counts into a long table of probabilities
    summary_rows = _build_summary_rows_from_counts(
        transition_counts,
        standard_totals,
        group_totals,
    )
    _save_csv(summary_csv_path, summary_rows)

    print("Trial data generation complete.")
    print(f"Output directory: {out_dir}")
    print(f"Trial CSV: {trial_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Total trials: {trial_counter}")

    return [], summary_rows


def run_with_defaults() -> None:
    """Run ``run_analysis`` with the module's default parameter grid."""
    start = time.perf_counter()
    run_analysis()
    elapsed_seconds = time.perf_counter() - start
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    run_with_defaults()
