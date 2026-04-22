"""
Tests for the embryobiopsy3d command-line interface (rich-click based).

Covers:
- Argument-parser behaviour (help, missing/unknown subcommands).
- ``demo`` subcommand output, determinism, and error-rate flags.
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
from click.testing import CliRunner

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


@pytest.fixture()
def runner() -> CliRunner:
    """Provide a fresh CliRunner per test.

    Click 8.2 removed ``mix_stderr``; stdout and stderr are combined into
    ``result.output`` by default, which is what our assertions use.
    """
    return CliRunner()


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
        "--seed",
        str(args["seed"]),
        "--cell-index",
        str(args["cell_index"]),
        "--out-dir",
        args["out_dir"],
    ]
    for value in args["generation_idx"]:
        argv += ["--generation-idx", str(value)]
    for value in args["dispersal"]:
        argv += ["--dispersal", str(value)]
    for value in args["distance"]:
        argv += ["--distance", str(value)]
    return argv


def _assert_ok(result) -> None:
    """Helper: fail with a readable message on non-zero exit codes."""
    assert result.exit_code == 0, (
        f"exit_code={result.exit_code}\n"
        f"output:\n{result.output}\n"
        f"exception: {result.exception!r}"
    )


# ---------------------------------------------------------------------------
# Parser-level behaviour
# ---------------------------------------------------------------------------


def test_help_exits_zero(runner):
    """The root ``--help`` flag exits 0 and advertises both subcommands."""
    result = runner.invoke(cli.main, ["--help"])
    _assert_ok(result)
    assert "embryobiopsy3d" in result.output
    assert "demo" in result.output
    assert "sweep" in result.output


def test_demo_help_exits_zero(runner):
    """``demo --help`` shows demo-specific flags and exits 0."""
    result = runner.invoke(cli.main, ["demo", "--help"])
    _assert_ok(result)
    assert "--generations" in result.output
    assert "--aneuploid-generation" in result.output
    assert "--meio-rate" in result.output
    assert "--mito-rate" in result.output


def test_sweep_help_exits_zero(runner):
    """``sweep --help`` shows sweep-specific flags and exits 0."""
    result = runner.invoke(cli.main, ["sweep", "--help"])
    _assert_ok(result)
    assert "--n-trials" in result.output
    assert "--out-dir" in result.output


def test_short_help_flag_works(runner):
    """``-h`` is also wired up (help_option_names=['-h','--help'])."""
    result = runner.invoke(cli.main, ["-h"])
    _assert_ok(result)
    assert "demo" in result.output


def test_missing_subcommand_shows_help(runner):
    """
    Calling the group with no subcommand is a no-op (exit 0) and prints help,
    which is standard Click group behaviour.
    """
    result = runner.invoke(cli.main, [])
    assert result.exit_code in (0, 2)
    assert "demo" in result.output or "Usage" in result.output


def test_unknown_subcommand_errors(runner):
    """Unknown subcommands are rejected by Click with exit code 2."""
    result = runner.invoke(cli.main, ["nosuchthing"])
    assert result.exit_code == 2
    assert "No such command" in result.output or "Usage" in result.output


# ---------------------------------------------------------------------------
# ``demo`` subcommand
# ---------------------------------------------------------------------------


def test_demo_default_runs(runner):
    """Default ``demo`` invocation succeeds and prints the expected sections."""
    result = runner.invoke(cli.main, ["demo", "--generations", "4"])
    _assert_ok(result)

    out = result.output
    assert "Building embryo" in out
    assert "Rebiopsy result" in out
    assert "standard category" in out
    assert "second category" in out
    assert "match" in out
    assert "Marked aneuploid subtree" not in out
    assert "Random errors" not in out


def test_demo_with_aneuploid_subtree(runner):
    """``--aneuploid-generation`` marks a subtree and shows it in the output."""
    result = runner.invoke(
        cli.main,
        [
            "demo",
            "--generations",
            "5",
            "--aneuploid-generation",
            "2",
            "--aneuploid-cell-index",
            "0",
        ],
    )
    _assert_ok(result)
    assert "Marked aneuploid subtree: generation=2, index=0" in result.output


def test_demo_deterministic_for_same_seed(runner):
    """Same CLI flags (including seed) produce identical demo output."""
    argv = ["demo", "--generations", "4", "--seed", "123"]
    first = runner.invoke(cli.main, argv)
    second = runner.invoke(cli.main, argv)
    _assert_ok(first)
    _assert_ok(second)
    assert first.output == second.output


def test_demo_different_seed_changes_output(runner):
    """Different seeds usually change something observable in the output."""
    r1 = runner.invoke(cli.main, ["demo", "--generations", "6", "--seed", "1"])
    r2 = runner.invoke(cli.main, ["demo", "--generations", "6", "--seed", "999"])
    _assert_ok(r1)
    _assert_ok(r2)
    assert r1.output != r2.output


def test_demo_custom_distance_appears_in_output(runner):
    """The ``--distance`` value is echoed in the summary."""
    result = runner.invoke(
        cli.main, ["demo", "--generations", "4", "--distance", "0.3"]
    )
    _assert_ok(result)
    assert "requested distance  : 0.3" in result.output


def test_demo_with_meio_rate_prints_random_errors_line(runner):
    """A nonzero --meio-rate triggers the random-errors summary line."""
    result = runner.invoke(
        cli.main,
        ["demo", "--generations", "5", "--meio-rate", "0.5", "--seed", "3"],
    )
    _assert_ok(result)
    assert "Random errors:" in result.output
    assert "meio_rate=0.5" in result.output
    assert "mito_rate=0.0" in result.output


def test_demo_with_mito_rate_prints_random_errors_line(runner):
    """A nonzero --mito-rate alone also triggers the random-errors line."""
    result = runner.invoke(
        cli.main,
        ["demo", "--generations", "5", "--mito-rate", "0.3", "--seed", "3"],
    )
    _assert_ok(result)
    assert "Random errors:" in result.output
    assert "mito_rate=0.3" in result.output


def test_demo_random_errors_are_deterministic_given_seed(runner):
    """Same seed + same rates => same output byte-for-byte."""
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
    r1 = runner.invoke(cli.main, argv)
    r2 = runner.invoke(cli.main, argv)
    _assert_ok(r1)
    _assert_ok(r2)
    assert r1.output == r2.output


def test_demo_combines_random_and_targeted_aneuploidy(runner):
    """--meio-rate with --aneuploid-generation prints both info lines."""
    result = runner.invoke(
        cli.main,
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
        ],
    )
    _assert_ok(result)
    assert "Random errors:" in result.output
    assert "Marked aneuploid subtree: generation=2, index=0" in result.output


# ---------------------------------------------------------------------------
# ``demo`` subcommand: Click-level validation
# ---------------------------------------------------------------------------


def test_demo_rejects_dispersal_out_of_range(runner):
    """--dispersal > 1 is rejected at Click level with exit code 2."""
    result = runner.invoke(cli.main, ["demo", "--dispersal", "2.0"])
    assert result.exit_code == 2
    assert "dispersal" in result.output.lower()


def test_demo_rejects_meio_rate_out_of_range(runner):
    """--meio-rate < 0 is rejected at Click level."""
    result = runner.invoke(cli.main, ["demo", "--meio-rate", "-0.1"])
    assert result.exit_code == 2


def test_demo_rejects_nonpositive_generations(runner):
    """--generations 0 is rejected (IntRange min=1)."""
    result = runner.invoke(cli.main, ["demo", "--generations", "0"])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: file creation and schema
# ---------------------------------------------------------------------------


def test_sweep_creates_both_csvs(runner, tmp_path: Path):
    """Sweep writes the trial and summary CSVs inside --out-dir."""
    out_dir = tmp_path / "sweep"
    result = runner.invoke(cli.main, _sweep_argv(out_dir))
    _assert_ok(result)

    assert (out_dir / "rebiopsy_trials.csv").is_file()
    assert (out_dir / "rebiopsy_transition_summary.csv").is_file()


def test_sweep_creates_out_dir_if_missing(runner, tmp_path: Path):
    """--out-dir is created even if a parent and the dir itself do not exist."""
    out_dir = tmp_path / "nested" / "does_not_exist"
    assert not out_dir.exists()

    result = runner.invoke(cli.main, _sweep_argv(out_dir))
    _assert_ok(result)
    assert out_dir.is_dir()


def test_sweep_trial_csv_header_matches_spec(runner, tmp_path: Path):
    """Trial CSV columns exactly match ``TRIAL_FIELDNAMES`` from trials.py."""
    out_dir = tmp_path / "sweep"
    result = runner.invoke(cli.main, _sweep_argv(out_dir))
    _assert_ok(result)

    rows = _read_csv(out_dir / "rebiopsy_trials.csv")
    assert rows, "trials CSV should not be empty"
    assert list(rows[0].keys()) == TRIAL_FIELDNAMES


def test_sweep_summary_csv_header_matches_spec(runner, tmp_path: Path):
    """Summary CSV has the expected column order."""
    out_dir = tmp_path / "sweep"
    result = runner.invoke(cli.main, _sweep_argv(out_dir))
    _assert_ok(result)

    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")
    assert rows, "summary CSV should not be empty"
    assert list(rows[0].keys()) == SUMMARY_FIELDNAMES


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: content correctness
# ---------------------------------------------------------------------------


def test_sweep_trial_row_count_matches_grid(runner, tmp_path: Path):
    """Total rows = |generation_idx| * |dispersal| * |distance| * n_trials."""
    out_dir = tmp_path / "sweep"
    argv = _sweep_argv(
        out_dir,
        n_trials=3,
        generation_idx=[1, 2, 3],
        dispersal=[0.0, 0.5, 1.0],
        distance=[0.0, 1.0],
    )
    result = runner.invoke(cli.main, argv)
    _assert_ok(result)

    rows = _read_csv(out_dir / "rebiopsy_trials.csv")
    assert len(rows) == 3 * 3 * 2 * 3


def test_sweep_trial_values_respect_custom_grids(runner, tmp_path: Path):
    """Unique values in the trial CSV match the grids passed via flags."""
    out_dir = tmp_path / "sweep"
    argv = _sweep_argv(
        out_dir,
        generation_idx=[1, 3],
        dispersal=[0.0, 1.0],
        distance=[0.5],
    )
    result = runner.invoke(cli.main, argv)
    _assert_ok(result)

    rows = _read_csv(out_dir / "rebiopsy_trials.csv")
    assert {int(r["division"]) for r in rows} == {1, 3}
    assert {float(r["dispersal"]) for r in rows} == {0.0, 1.0}
    assert {float(r["distance"]) for r in rows} == {0.5}


def test_sweep_summary_has_nine_rows_per_group(runner, tmp_path: Path):
    """Summary has a 3x3 block per (division, dispersal, distance) group."""
    out_dir = tmp_path / "sweep"
    argv = _sweep_argv(
        out_dir,
        n_trials=2,
        generation_idx=[1, 2],
        dispersal=[0.0, 1.0],
        distance=[0.0, 0.5],
    )
    result = runner.invoke(cli.main, argv)
    _assert_ok(result)

    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")
    n_groups = 2 * 2 * 2
    assert len(rows) == 9 * n_groups
    for row in rows:
        assert row["standard_category"] in BIOPSY_CATEGORIES
        assert row["second_category"] in BIOPSY_CATEGORIES


def test_sweep_summary_joint_probabilities_sum_to_one(runner, tmp_path: Path):
    """Within each group, joint probabilities sum to 1.0."""
    out_dir = tmp_path / "sweep"
    result = runner.invoke(cli.main, _sweep_argv(out_dir, n_trials=5))
    _assert_ok(result)

    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")
    groups: dict[tuple, float] = {}
    for row in rows:
        key = (row["division"], row["dispersal"], row["distance"])
        groups.setdefault(key, 0.0)
        groups[key] += float(row["joint_probability"])

    for key, total in groups.items():
        assert total == pytest.approx(1.0, abs=1e-9), f"joint probs != 1 for {key}"


def test_sweep_summary_transition_counts_sum_to_n_trials(runner, tmp_path: Path):
    """transition_count within a group sums to n_trials."""
    out_dir = tmp_path / "sweep"
    n_trials = 4
    result = runner.invoke(cli.main, _sweep_argv(out_dir, n_trials=n_trials))
    _assert_ok(result)

    rows = _read_csv(out_dir / "rebiopsy_transition_summary.csv")
    totals: dict[tuple, int] = {}
    for row in rows:
        key = (row["division"], row["dispersal"], row["distance"])
        totals.setdefault(key, 0)
        totals[key] += int(row["transition_count"])
    for key, total in totals.items():
        assert total == n_trials, f"transition_count != n_trials for {key}"


def test_sweep_summary_conditional_probabilities_sum_to_one_or_nan(
    runner, tmp_path: Path
):
    """
    For each (group, standard_category), conditional probs either sum to 1.0
    (standard_total > 0) or are all NaN (standard_total == 0).
    """
    out_dir = tmp_path / "sweep"
    result = runner.invoke(cli.main, _sweep_argv(out_dir, n_trials=10))
    _assert_ok(result)

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
        any_nan = any(p != p for p in probs)
        if any_nan:
            assert all(p != p for p in probs), f"mixed NaN/finite probs for {key}"
        else:
            assert sum(probs) == pytest.approx(1.0, abs=1e-9), (
                f"cond probs != 1 for {key}"
            )


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: determinism
# ---------------------------------------------------------------------------


def test_sweep_deterministic_for_same_seed(runner, tmp_path: Path):
    """Two sweeps with identical flags produce byte-identical trial CSVs."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    _assert_ok(runner.invoke(cli.main, _sweep_argv(out_a, seed=42)))
    _assert_ok(runner.invoke(cli.main, _sweep_argv(out_b, seed=42)))

    assert (out_a / "rebiopsy_trials.csv").read_bytes() == (
        out_b / "rebiopsy_trials.csv"
    ).read_bytes()
    assert (out_a / "rebiopsy_transition_summary.csv").read_bytes() == (
        out_b / "rebiopsy_transition_summary.csv"
    ).read_bytes()


def test_sweep_different_seed_changes_trials(runner, tmp_path: Path):
    """Changing the seed changes the set of per-trial seeds in the CSV."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    _assert_ok(runner.invoke(cli.main, _sweep_argv(out_a, seed=1)))
    _assert_ok(runner.invoke(cli.main, _sweep_argv(out_b, seed=2)))

    seeds_a = [r["seed"] for r in _read_csv(out_a / "rebiopsy_trials.csv")]
    seeds_b = [r["seed"] for r in _read_csv(out_b / "rebiopsy_trials.csv")]
    assert seeds_a != seeds_b


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: Click-level validation (exit code 2)
# ---------------------------------------------------------------------------


def test_sweep_rejects_dispersal_out_of_range(runner, tmp_path: Path):
    """Dispersal > 1.0 is rejected by Click's FloatRange."""
    result = runner.invoke(cli.main, _sweep_argv(tmp_path / "s", dispersal=[2.0]))
    assert result.exit_code == 2


def test_sweep_rejects_distance_out_of_range(runner, tmp_path: Path):
    """Distance outside [0, 1] is rejected by Click's FloatRange."""
    result = runner.invoke(cli.main, _sweep_argv(tmp_path / "s", distance=[-0.1]))
    assert result.exit_code == 2


def test_sweep_rejects_nonpositive_generations(runner, tmp_path: Path):
    """--generations 0 is rejected by Click's IntRange(min=1)."""
    result = runner.invoke(cli.main, _sweep_argv(tmp_path / "s", generations=0))
    assert result.exit_code == 2


def test_sweep_rejects_nonpositive_n_trials(runner, tmp_path: Path):
    """--n-trials 0 is rejected by Click's IntRange(min=1)."""
    result = runner.invoke(cli.main, _sweep_argv(tmp_path / "s", n_trials=0))
    assert result.exit_code == 2


def test_sweep_rejects_bad_int_flag_value(runner, tmp_path: Path):
    """Non-integer value for an int flag is rejected by Click (exit 2)."""
    argv = _sweep_argv(tmp_path / "s")
    n_trials_pos = argv.index("--n-trials") + 1
    argv[n_trials_pos] = "not-an-int"
    result = runner.invoke(cli.main, argv)
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# ``sweep`` subcommand: cross-field validation (raised by run_analysis)
# ---------------------------------------------------------------------------


def test_sweep_rejects_generation_idx_exceeding_generations(runner, tmp_path: Path):
    """
    Cross-field validation (generation_idx > generations) is enforced by
    ``run_analysis`` and surfaces as a non-zero exit code with an exception.
    """
    argv = _sweep_argv(tmp_path / "s", generations=3, generation_idx=[5])
    result = runner.invoke(cli.main, argv, catch_exceptions=True)
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "generation indices" in str(result.exception)


# ---------------------------------------------------------------------------
# Installed entry-point smoke test
# ---------------------------------------------------------------------------


def test_installed_entry_point_help_runs():
    """``embryobiopsy3d --help`` works as an installed console script."""
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
    """``python -m embryobiopsy3d.cli --help`` also works."""
    result = subprocess.run(
        [sys.executable, "-m", "embryobiopsy3d.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "demo" in result.stdout
    assert "sweep" in result.stdout
