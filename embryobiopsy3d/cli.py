"""
Command-line interface for embryobiopsy3d.

Installed entry point: ``embryobiopsy3d``.

Subcommands
-----------
- ``demo``  — Build one embryo and run a single biopsy.
- ``sweep`` — Run a rebiopsy parameter sweep and write CSV outputs.

Run ``embryobiopsy3d --help`` or ``embryobiopsy3d <subcommand> --help`` for
details.
"""

from __future__ import annotations

import time

import rich_click as click

from .lineage_simulator import build_embryo
from .rebiopsy import rebiopsy_single_embryo
from .trials import (
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

click.rich_click.TEXT_MARKUP = "markdown"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_OPTION_DEFAULT = "dim"
click.rich_click.MAX_WIDTH = 100

_UNIT_INTERVAL = click.FloatRange(0.0, 1.0)
_POSITIVE_INT = click.IntRange(min=1)
_NON_NEGATIVE_INT = click.IntRange(min=0)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="embryo-biopsy-3d", prog_name="embryobiopsy3d")
def main() -> None:
    """
    Command-line interface for the **embryobiopsy3d** simulator.

    Use `demo` for a quick sanity check or `sweep` to reproduce
    parameter-sweep CSV outputs.
    """


############################################################### DEMO ################################################################


@main.command()
@click.option(
    "--generations",
    type=_POSITIVE_INT,
    default=6,
    show_default=True,
    help="Number of cell generations.",
)
@click.option(
    "--dispersal",
    type=_UNIT_INTERVAL,
    default=0.25,
    show_default=True,
    help="Placement dispersal in [0, 1].",
)
@click.option(
    "--distance",
    type=_UNIT_INTERVAL,
    default=0.5,
    show_default=True,
    help="Rebiopsy target distance as a fraction of pi in [0, 1].",
)
@click.option(
    "--seed",
    type=int,
    default=7,
    show_default=True,
    help="Random seed.",
)
@click.option(
    "--meio-rate",
    type=_UNIT_INTERVAL,
    default=0.0,
    show_default=True,
    help=(
        "Per-division meiotic error rate in [0, 1]. Randomly marks cells "
        "aneuploid during tree construction."
    ),
)
@click.option(
    "--mito-rate",
    type=_UNIT_INTERVAL,
    default=0.0,
    show_default=True,
    help=(
        "Per-division mitotic error rate in [0, 1]. Randomly marks cells "
        "aneuploid during tree construction."
    ),
)
@click.option(
    "--aneuploid-generation",
    type=_NON_NEGATIVE_INT,
    default=None,
    help=(
        "If given, mark the subtree rooted at (generation, index) aneuploid "
        "before biopsy. Can be combined with --meio-rate/--mito-rate for a "
        "hybrid deterministic + random aneuploidy."
    ),
)
@click.option(
    "--aneuploid-cell-index",
    type=_NON_NEGATIVE_INT,
    default=0,
    show_default=True,
    help="Index within the aneuploid generation (used only with --aneuploid-generation).",
)
def demo(
    generations: int,
    dispersal: float,
    distance: float,
    seed: int,
    meio_rate: float,
    mito_rate: float,
    aneuploid_generation: int | None,
    aneuploid_cell_index: int,
) -> None:
    """
    Build one embryo and run a single rebiopsy — a fast sanity check.

    Intended as a zero-setup confirmation that the package installed correctly.
    """
    click.echo(
        f"Building embryo: generations={generations}, "
        f"dispersal={dispersal}, seed={seed}"
    )
    embryo = build_embryo(
        generations=generations,
        meio_rate=meio_rate,
        mito_rate=mito_rate,
        placement_dispersal=dispersal,
        seed=seed,
    )

    if meio_rate > 0.0 or mito_rate > 0.0:
        n_random = len(embryo.mutated_cells or [])
        click.echo(
            f"  Random errors: meio_rate={meio_rate}, "
            f"mito_rate={mito_rate} -> {n_random} aneuploid cells"
        )

    if aneuploid_generation is not None:
        embryo.set_aneuploid_by_generation_index(
            aneuploid_generation,
            aneuploid_cell_index,
            is_aneuploid=True,
            include_subtree=True,
        )
        click.echo(
            f"  Marked aneuploid subtree: generation={aneuploid_generation}, "
            f"index={aneuploid_cell_index}"
        )

    result = rebiopsy_single_embryo(
        embryo,
        distance,
        return_metadata=True,
        seed=seed,
    )

    click.echo("\nRebiopsy result")
    click.echo("---------------")
    click.echo(f"  requested distance  : {distance} * pi")
    click.echo(f"  actual distance     : {result.get('actual_distance')}")
    click.echo(f"  standard category   : {result.get('standard_category')}")
    click.echo(f"  second category     : {result.get('second_category')}")
    click.echo(f"  standard aneuploid count: {result.get('standard_aneuploid_count')}")
    click.echo(f"  second aneuploid count  : {result.get('second_aneuploid_count')}")
    click.echo(f"  match               : {result.get('match')}")


############################################################### SWEEP ################################################################


@main.command()
@click.option(
    "--generations",
    type=_POSITIVE_INT,
    default=DEFAULT_GENERATIONS,
    show_default=True,
    help="Total cell generations in the lineage tree.",
)
@click.option(
    "--n-trials",
    type=_POSITIVE_INT,
    default=DEFAULT_N_TRIALS,
    show_default=True,
    help="Trials per (generation_index, dispersal, distance) combination.",
)
@click.option(
    "--generation-idx",
    "generation_idx",
    type=_NON_NEGATIVE_INT,
    multiple=True,
    default=tuple(DEFAULT_GENERATION_INDEX_VALUES),
    show_default=True,
    help=(
        "Generation index to place the fixed aneuploid subtree. "
        "Pass the flag multiple times to sweep several values, e.g. "
        "`--generation-idx 1 --generation-idx 2`."
    ),
)
@click.option(
    "--dispersal",
    type=_UNIT_INTERVAL,
    multiple=True,
    default=tuple(DEFAULT_DISPERSAL_VALUES),
    show_default=True,
    help=(
        "Dispersal value in [0, 1] to sweep. Pass the flag multiple times, "
        "e.g. `--dispersal 0.0 --dispersal 1.0`."
    ),
)
@click.option(
    "--distance",
    type=_UNIT_INTERVAL,
    multiple=True,
    default=tuple(DEFAULT_DISTANCE_VALUES),
    show_default=True,
    help=(
        "Rebiopsy distance (fraction of pi) in [0, 1] to sweep. "
        "Pass the flag multiple times."
    ),
)
@click.option(
    "--cell-index",
    type=_NON_NEGATIVE_INT,
    default=DEFAULT_CELL_INDEX,
    show_default=True,
    help="Index of the cell whose subtree is marked aneuploid.",
)
@click.option(
    "--seed",
    type=int,
    default=DEFAULT_BASE_SEED,
    show_default=True,
    help="Base seed for the per-trial RNG sequence.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    default=DEFAULT_OUT_DIR,
    show_default=True,
    help="Directory to write trial and summary CSVs into.",
)
def sweep(
    generations: int,
    n_trials: int,
    generation_idx: tuple[int, ...],
    dispersal: tuple[float, ...],
    distance: tuple[float, ...],
    cell_index: int,
    seed: int,
    out_dir: str,
) -> None:
    """
    Run a rebiopsy parameter sweep and write CSV outputs.

    Sweeps over (generation index, dispersal, distance) with a configurable
    number of trials per cell. Writes `rebiopsy_trials.csv` and
    `rebiopsy_transition_summary.csv` into `--out-dir`.
    """
    start = time.perf_counter()
    run_analysis(
        generations=generations,
        n_trials=n_trials,
        generation_index_values=list(generation_idx),
        dispersal_values=list(dispersal),
        distance_values=list(distance),
        base_seed=seed,
        cell_index=cell_index,
        out_dir=out_dir,
    )
    elapsed_seconds = time.perf_counter() - start
    click.echo(f"Elapsed time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
