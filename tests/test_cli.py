"""
Tests for the embryobiopsy3d command-line interface.

Covers:
- Argument-parser behaviour (help, missing/unknown subcommands).
- ``demo`` subcommand output and determinism.
- ``sweep`` subcommand CSV schema, row counts, probability correctness,
  determinism, validation, and custom grid handling.
- Installed entry-point smoke test via subprocess.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from embryobiopsy3d import cli
from embryobiopsy3d.trials import BIOPSY_CATEGORIES, TRIAL_FIELDNAMES


SUMMARY_FIELDNAMES = [
    "division",
    "cell_index",
    "aneuploid_leaf_count",
    "dispersal",
    "distance",
    "n_trials",
    "standard_category",
    "second_category",
    "transition_count",
    "standard_total",
    "conditional_probability",
    "joint_probability",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sweep_argv(out_dir: Path, **overrides) -> list[str]:
    """Build a minimal ``sweep`` argv with sane small defaults for fast tests."""
    args = {
        "generations": 4,
        "n_trials": 2,
        "generation_idx": [1, 2],
        "dispersal": [0.0, 1.0],
        "distance": [0.0, 0.5],
        "seed": 7,
        "cell_index": 0,
        "out_dir": str(out_dir),
    }
    args.update(overrides)

    argv = [
        "sweep",
        "--generations",
        str(args["generations"]),
        "--n-trials",
        str(args["n_trials"]),
        "--generation-idx",
        *[str(v) for v in args["generation_idx"]],
        "--dispersal",
        *[str(v) for v in args["dispersal"]],
        "--distance",
        *[str(v) for v in args["distance"]],
        "--seed",
        str(args["seed"]),
        "--cell-index",
        str(args["cell_index"]),
        "--out-dir",
        args["out_dir"],
    ]
    return argv


# ---------------------------------------------------------------------------
# Parser-level behaviour
# ---------------------------------------------------------------------------


def test_help_exits_zero(capsys):
    """The root ``--help`` flag exits with code 0 and prints usage."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "embryobiopsy3d" in out
    assert "demo" in out
    assert "sweep" in out


def test_demo_help_exits_zero(capsys):
    """``demo --help`` shows demo-specific flags and exits 0."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["demo", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--generations" in out
    assert "--aneuploid-generation" in out


def test_sweep_help_exits_zero(capsys):
    """``sweep --help`` shows sweep-specific flags and exits 0."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["sweep", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--n-trials" in out
    assert "--out-dir" in out


def test_missing_subcommand_errors():
    """Calling with no subcommand is an argparse error (exit code 2)."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_unknown_subcommand_errors():
    """Unknown subcommands are rejected by argparse with exit code 2."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["nosuchthing"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# ``demo`` subcommand
# ---------------------------------------------------------------------------


def test_demo_default_runs(capsys):
    """Default ``demo`` invocation succeeds and prints the expected sections."""
    rc = cli.main(["demo", "--generations", "4"])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Building embryo" in out
    assert "Rebiopsy result" in out
    assert "standard category" in out
    assert "second category" in out
    assert "match" in out
    # No aneuploid subtree was requested, so that line must not appear
    assert "Marked aneuploid subtree" not in out


def test_demo_with_aneuploid_subtree(capsys):
    """``--aneuploid-generation`` marks a subtree and that shows in the output."""
    rc = cli.main(
        [
            "demo",
            "--generations",
            "5",
            "--aneuploid-generation",
            "2",
            "--aneuploid-cell-index",
            "0",
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "Marked aneuploid subtree: generation=2, index=0" in out
    assert "standard category" in out


def test_demo_deterministic_for_same_seed(capsys):
    """Same CLI flags (including seed) produce identical demo output."""
    cli.main(["demo", "--generations", "4", "--seed", "123"])
    first = capsys.readouterr().out

    cli.main(["demo", "--generations", "4", "--seed", "123"])
    second = capsys.readouterr().out

    assert first == second


def test_demo_different_seed_changes_output(capsys):
    """Different seeds usually change something observable in the output."""
    cli.main(["demo", "--generations", "6", "--seed", "1"])
    first = capsys.readouterr().out

    cli.main(["demo", "--generations", "6", "--seed", "999"])
    second = capsys.readouterr().out

    assert first != second


def test_demo_custom_distance_appears_in_output(capsys):
    """The ``--distance`` value is echoed in the summary."""
    cli.main(["demo", "--generations", "4", "--distance", "0.3"])
    out = capsys.readouterr().out
    assert "requested distance  : 0.3" in out


def test_demo_default_does_not_print_random_errors_line(capsys):
    """With default rates of 0, the random-errors info line is absent."""
    cli.main(["demo", "--generations", "4"])
    out = capsys.readouterr().out
    assert "Random errors" not in out


def test_demo_with_meio_rate_prints_random_errors_line(capsys):
    """A nonzero --meio-rate causes the random-errors summary line to appear."""
    cli.main(
        [
            "demo",
            "--generations",
            "5",
            "--meio-rate",
            "0.5",
            "--seed",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert "Random errors:" in out
    assert "meio_rate=0.5" in out
    assert "mito_rate=0.0" in out
    assert "aneuploid cells" in out


def test_demo_with_mito_rate_prints_random_errors_line(capsys):
    """A nonzero --mito-rate alone also triggers the random-errors line."""
    cli.main(
        [
            "demo",
            "--generations",
            "5",
            "--mito-rate",
            "0.3",
            "--seed",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert "Random errors:" in out
    assert "mito_rate=0.3" in out


def test_demo_random_errors_are_deterministic_given_seed(capsys):
    """Same seed + same rates => same aneuploid cell count."""
    argv = [
        "demo",
        "--generations",
        "6",
        "--meio-rate",
        "0.2",
        "--mito-rate",
        "0.1",
        "--seed",
        "42",
    ]
    cli.main(argv)
    first = capsys.readouterr().out
    cli.main(argv)
    second = capsys.readouterr().out
    assert first == second


def test_demo_combines_random_and_targeted_aneuploidy(capsys):
    """--meio-rate together with --aneuploid-generation prints both lines."""
    cli.main(
        [
            "demo",
            "--generations",
            "5",
            "--meio-rate",
            "0.4",
            "--aneuploid-generation",
            "2",
            "--aneuploid-cell-index",
            "0",
            "--seed",
            "5",
        ]
    )
    out = capsys.readouterr().out
    assert "Random errors:" in out
    assert "Marked aneuploid subtree: generation=2, index=0" in out


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: file creation and schema
# ---------------------------------------------------------------------------


def test_sweep_creates_both_csvs(tmp_path: Path):
    """Sweep writes the trial and summary CSVs inside --out-dir."""
    out_dir = tmp_path / "sweep"
    rc = cli.main(_sweep_argv(out_dir))
    assert rc == 0

    trials_csv = out_dir / "rebiopsy_trials.csv"
    summary_csv = out_dir / "rebiopsy_transition_summary.csv"
    assert trials_csv.is_file()
    assert summary_csv.is_file()


def test_sweep_creates_out_dir_if_missing(tmp_path: Path):
    """--out-dir is created even if a parent and the dir itself do not exist."""
    out_dir = tmp_path / "nested" / "does_not_exist"
    assert not out_dir.exists()

    rc = cli.main(_sweep_argv(out_dir))
    assert rc == 0
    assert out_dir.is_dir()


def test_sweep_trial_csv_header_matches_spec(tmp_path: Path):
    """Trial CSV columns exactly match ``TRIAL_FIELDNAMES`` from trials.py."""
    out_dir = tmp_path / "sweep"
    cli.main(_sweep_argv(out_dir))

    rows = _read_csv(out_dir / "rebiopsy_trials.csv")
    assert rows, "trials CSV should not be empty"
    assert list(rows[0].keys()) == TRIAL_FIELDNAMES


def test_sweep_summary_csv_header_matches_spec(tmp_path: Path):
    """Summary CSV has the expected column order."""
    out_dir = tmp_path / "sweep"
    cli.main(_sweep_argv(out_dir))

    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")
    assert rows, "summary CSV should not be empty"
    assert list(rows[0].keys()) == SUMMARY_FIELDNAMES


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: content correctness
# ---------------------------------------------------------------------------


def test_sweep_trial_row_count_matches_grid(tmp_path: Path):
    """
    Total rows = |generation_idx| * |dispersal| * |distance| * n_trials.
    """
    out_dir = tmp_path / "sweep"
    argv = _sweep_argv(
        out_dir,
        n_trials=3,
        generation_idx=[1, 2, 3],
        dispersal=[0.0, 0.5, 1.0],
        distance=[0.0, 1.0],
    )
    cli.main(argv)

    rows = _read_csv(out_dir / "rebiopsy_trials.csv")
    assert len(rows) == 3 * 3 * 2 * 3


def test_sweep_trial_values_respect_custom_grids(tmp_path: Path):
    """Unique values in the trial CSV match the grids passed via flags."""
    out_dir = tmp_path / "sweep"
    argv = _sweep_argv(
        out_dir,
        generation_idx=[1, 3],
        dispersal=[0.0, 1.0],
        distance=[0.5],
    )
    cli.main(argv)

    rows = _read_csv(out_dir / "rebiopsy_trials.csv")
    assert {int(r["division"]) for r in rows} == {1, 3}
    assert {float(r["dispersal"]) for r in rows} == {0.0, 1.0}
    assert {float(r["distance"]) for r in rows} == {0.5}


def test_sweep_summary_has_nine_rows_per_group(tmp_path: Path):
    """Summary has a 3x3 block per (division, dispersal, distance) group."""
    out_dir = tmp_path / "sweep"
    argv = _sweep_argv(
        out_dir,
        n_trials=2,
        generation_idx=[1, 2],
        dispersal=[0.0, 1.0],
        distance=[0.0, 0.5],
    )
    cli.main(argv)

    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")
    n_groups = 2 * 2 * 2
    assert len(rows) == 9 * n_groups

    for row in rows:
        assert row["standard_category"] in BIOPSY_CATEGORIES
        assert row["second_category"] in BIOPSY_CATEGORIES


def test_sweep_summary_joint_probabilities_sum_to_one(tmp_path: Path):
    """Within each (division, dispersal, distance) group, joint probs sum to 1."""
    out_dir = tmp_path / "sweep"
    cli.main(_sweep_argv(out_dir, n_trials=5))
    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")

    groups: dict[tuple, float] = {}
    for row in rows:
        key = (row["division"], row["dispersal"], row["distance"])
        groups.setdefault(key, 0.0)
        groups[key] += float(row["joint_probability"])

    for key, total in groups.items():
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"joint probs do not sum to 1 for {key}"
        )


def test_sweep_summary_transition_counts_sum_to_n_trials(tmp_path: Path):
    """transition_count within a group sums to n_trials for that group."""
    out_dir = tmp_path / "sweep"
    n_trials = 4
    cli.main(_sweep_argv(out_dir, n_trials=n_trials))
    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")

    group_totals: dict[tuple, int] = {}
    for row in rows:
        key = (row["division"], row["dispersal"], row["distance"])
        group_totals.setdefault(key, 0)
        group_totals[key] += int(row["transition_count"])

    for key, total in group_totals.items():
        assert total == n_trials, f"transition_count does not sum to n_trials for {key}"


def test_sweep_summary_conditional_probabilities_sum_to_one_or_nan(tmp_path: Path):
    """
    For each (group, standard_category), conditional probs either sum to 1.0
    (standard_total > 0) or are all NaN (standard_total == 0).
    """
    out_dir = tmp_path / "sweep"
    cli.main(_sweep_argv(out_dir, n_trials=10))
    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")

    buckets: dict[tuple, list[float]] = {}
    for row in rows:
        key = (
            row["division"],
            row["dispersal"],
            row["distance"],
            row["standard_category"],
        )
        buckets.setdefault(key, []).append(float(row["conditional_probability"]))

    for key, probs in buckets.items():
        any_nan = any(p != p for p in probs)  # NaN != NaN
        if any_nan:
            assert all(p != p for p in probs), f"mixed NaN/finite probs for {key}"
        else:
            assert sum(probs) == pytest.approx(1.0, abs=1e-9), (
                f"cond probs != 1 for {key}"
            )


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: determinism
# ---------------------------------------------------------------------------


def test_sweep_deterministic_for_same_seed(tmp_path: Path):
    """Two sweeps with identical flags produce byte-identical trial CSVs."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cli.main(_sweep_argv(out_a, seed=42))
    cli.main(_sweep_argv(out_b, seed=42))

    assert (out_a / "rebiopsy_trials.csv").read_bytes() == (
        out_b / "rebiopsy_trials.csv"
    ).read_bytes()
    assert (out_a / "rebiopsy_transition_summary.csv").read_bytes() == (
        out_b / "rebiopsy_transition_summary.csv"
    ).read_bytes()


def test_sweep_different_seed_changes_trials(tmp_path: Path):
    """Changing the seed changes the set of per-trial seeds in the CSV."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    cli.main(_sweep_argv(out_a, seed=1))
    cli.main(_sweep_argv(out_b, seed=2))

    seeds_a = [r["seed"] for r in _read_csv(out_a / "rebiopsy_trials.csv")]
    seeds_b = [r["seed"] for r in _read_csv(out_b / "rebiopsy_trials.csv")]
    assert seeds_a != seeds_b


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: validation
# ---------------------------------------------------------------------------


def test_sweep_rejects_dispersal_out_of_range(tmp_path: Path):
    """Dispersal > 1.0 is rejected by run_analysis with ValueError."""
    out_dir = tmp_path / "sweep"
    with pytest.raises(ValueError, match="dispersal"):
        cli.main(_sweep_argv(out_dir, dispersal=[2.0]))


def test_sweep_rejects_distance_out_of_range(tmp_path: Path):
    """Distance outside [0, 1] is rejected with ValueError."""
    out_dir = tmp_path / "sweep"
    with pytest.raises(ValueError, match="distance"):
        cli.main(_sweep_argv(out_dir, distance=[-0.1]))


def test_sweep_rejects_nonpositive_generations(tmp_path: Path):
    """``--generations 0`` is rejected."""
    out_dir = tmp_path / "sweep"
    with pytest.raises(ValueError, match="generations"):
        cli.main(_sweep_argv(out_dir, generations=0))


def test_sweep_rejects_nonpositive_n_trials(tmp_path: Path):
    """``--n-trials 0`` is rejected."""
    out_dir = tmp_path / "sweep"
    with pytest.raises(ValueError, match="n-trials"):
        cli.main(_sweep_argv(out_dir, n_trials=0))


def test_sweep_rejects_generation_idx_exceeding_generations(tmp_path: Path):
    """Generation indices > total generations are rejected."""
    out_dir = tmp_path / "sweep"
    with pytest.raises(ValueError, match="generation indices"):
        cli.main(_sweep_argv(out_dir, generations=3, generation_idx=[5]))


def test_sweep_rejects_bad_int_flag_value(tmp_path: Path):
    """Non-integer value for an int flag is rejected by argparse (exit 2)."""
    out_dir = tmp_path / "sweep"
    argv = _sweep_argv(out_dir)
    n_trials_pos = argv.index("--n-trials") + 1
    argv[n_trials_pos] = "not-an-int"
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Installed entry-point smoke test
# ---------------------------------------------------------------------------


def test_installed_entry_point_help_runs():
    """
    ``embryobiopsy3d --help`` works when invoked as an installed console script.

    Skips gracefully if the environment running the tests does not have the
    entry point on PATH (e.g. the user ran ``pytest`` before ``pip install -e``).
    """
    exe = shutil.which("embryobiopsy3d")
    if exe is None:
        pytest.skip("embryobiopsy3d entry point not on PATH (package not installed)")

    result = subprocess.run(
        [exe, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "demo" in result.stdout
    assert "sweep" in result.stdout


def test_module_invocation_help_runs():
    """
    ``python -m embryobiopsy3d.cli --help`` also works.

    This covers the case where the entry point isn't installed but the package
    importer is available.
    """
    result = subprocess.run(
        [sys.executable, "-m", "embryobiopsy3d.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "demo" in result.stdout
    assert "sweep" in result.stdout
