"""
Command-line interface for embryobiopsy3d.

Installed entry point: ``embryobiopsy3d``.

Subcommands
-----------
  demo    Build one embryo and run a single biopsy.
  sweep   Run a rebiopsy parameter sweep and write CSV outputs.

Run ``embryobiopsy3d --help`` or ``embryobiopsy3d <subcommand> --help`` for
details.
"""

from __future__ import annotations

import argparse
import sys
import time

from embryobiopsy3d.lineage_simulator import build_embryo
from embryobiopsy3d.rebiopsy import rebiopsy_single_embryo
from embryobiopsy3d.trials import (
    DEFAULT_BASE_SEED,
    DEFAULT_CELL_INDEX,
    DEFAULT_DISPERSAL_VALUES,
    DEFAULT_DISTANCE_VALUES,
    DEFAULT_GENERATION_INDEX_VALUES,
    DEFAULT_GENERATIONS,
    DEFAULT_N_TRIALS,
    DEFAULT_OUT_DIR,
    run_analysis,
)


def _cmd_demo(args: argparse.Namespace) -> int:
    """Build a single embryo, run one biopsy, print a compact summary."""
    print(
        f"Building embryo: generations={args.generations}, "
        f"dispersal={args.dispersal}, seed={args.seed}"
    )
    embryo = build_embryo(
        generations=args.generations,
        meio_rate=0.0,
        mito_rate=0.0,
        placement_dispersal=args.dispersal,
        seed=args.seed,
    )

    if args.aneuploid_generation is not None:
        embryo.set_aneuploid_by_generation_index(
            args.aneuploid_generation,
            args.aneuploid_cell_index,
            is_aneuploid=True,
            include_subtree=True,
        )
        print(
            f"  Marked aneuploid subtree: generation={args.aneuploid_generation}, "
            f"index={args.aneuploid_cell_index}"
        )

    result = rebiopsy_single_embryo(
        embryo,
        args.distance,
        return_metadata=True,
        seed=args.seed,
    )

    print("\nRebiopsy result")
    print("---------------")
    print(f"  requested distance  : {args.distance} * pi")
    print(f"  actual distance     : {result.get('actual_distance')}")
    print(f"  standard category   : {result.get('standard_category')}")
    print(f"  second category     : {result.get('second_category')}")
    print(f"  standard aneuploid count: {result.get('standard_aneuploid_count')}")
    print(f"  second aneuploid count  : {result.get('second_aneuploid_count')}")
    print(f"  match               : {result.get('match')}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Wrap :func:`run_analysis` with timing and argument forwarding."""
    start = time.perf_counter()
    run_analysis(
        generations=args.generations,
        n_trials=args.n_trials,
        generation_index_values=args.generation_idx,
        dispersal_values=args.dispersal,
        distance_values=args.distance,
        base_seed=args.seed,
        cell_index=args.cell_index,
        out_dir=args.out_dir,
    )
    elapsed_seconds = time.perf_counter() - start
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embryobiopsy3d",
        description=(
            "Command-line interface for the embryobiopsy3d simulator. "
            "Use 'demo' for a quick sanity check, 'sweep' to reproduce "
            "parameter-sweep CSV outputs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    demo = subparsers.add_parser(
        "demo",
        help="Build one embryo and run a single rebiopsy (fast sanity check).",
        description=(
            "Build a single embryo and run one rebiopsy, then print a compact "
            "summary. Intended as a zero-setup sanity check that the package "
            "installed correctly."
        ),
    )
    demo.add_argument(
        "--generations", type=int, default=6, help="Number of cell generations."
    )
    demo.add_argument(
        "--dispersal",
        type=float,
        default=0.25,
        help="Placement dispersal in [0, 1].",
    )
    demo.add_argument(
        "--distance",
        type=float,
        default=0.5,
        help="Rebiopsy target distance as a fraction of pi in [0, 1].",
    )
    demo.add_argument("--seed", type=int, default=7, help="Random seed.")
    demo.add_argument(
        "--aneuploid-generation",
        type=int,
        default=None,
        help=(
            "If given, mark the subtree rooted at (generation, index) aneuploid "
            "before biopsy."
        ),
    )
    demo.add_argument(
        "--aneuploid-cell-index",
        type=int,
        default=0,
        help="Index within the aneuploid generation (used only with --aneuploid-generation).",
    )
    demo.set_defaults(func=_cmd_demo)

    sweep = subparsers.add_parser(
        "sweep",
        help="Run a rebiopsy parameter sweep and write CSV outputs.",
        description=(
            "Sweep over (generation index, dispersal, distance) with a configurable "
            "number of trials per cell. Writes 'rebiopsy_trials.csv' and "
            "'rebiopsy_transition_summary.csv' into --out-dir."
        ),
    )
    sweep.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help="Total cell generations in the lineage tree.",
    )
    sweep.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Trials per (generation_index, dispersal, distance) combination.",
    )
    sweep.add_argument(
        "--generation-idx",
        type=int,
        nargs="+",
        default=list(DEFAULT_GENERATION_INDEX_VALUES),
        metavar="G",
        help="Generation indices at which to place the fixed aneuploid subtree.",
    )
    sweep.add_argument(
        "--dispersal",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISPERSAL_VALUES),
        metavar="D",
        help="Dispersal values in [0, 1] to sweep.",
    )
    sweep.add_argument(
        "--distance",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISTANCE_VALUES),
        metavar="X",
        help="Rebiopsy distances (fractions of pi) in [0, 1] to sweep.",
    )
    sweep.add_argument(
        "--cell-index",
        type=int,
        default=DEFAULT_CELL_INDEX,
        help="Index of the cell whose subtree is marked aneuploid.",
    )
    sweep.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Base seed for the per-trial RNG sequence.",
    )
    sweep.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Directory to write trial and summary CSVs into.",
    )
    sweep.set_defaults(func=_cmd_sweep)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the subcommand's exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
