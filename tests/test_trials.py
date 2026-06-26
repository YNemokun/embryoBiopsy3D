"""Tests for embryobiopsy3d/trials.py — validation, CSV helpers, and run_with_defaults."""

import os

import pytest

from embryobiopsy3d import trials
from embryobiopsy3d.trials import _save_csv, run_analysis


# ---------------------------------------------------------------------------
# _save_csv
# ---------------------------------------------------------------------------


def test_save_csv_empty_rows_creates_empty_file(tmp_path):
    path = str(tmp_path / "out.csv")
    _save_csv(path, [])
    assert os.path.exists(path)
    with open(path) as fh:
        assert fh.read() == ""


def test_save_csv_writes_header_and_rows(tmp_path):
    path = str(tmp_path / "out.csv")
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    _save_csv(path, rows)
    content = open(path).read()
    assert "a,b" in content
    assert "1,2" in content
    assert "3,4" in content


# ---------------------------------------------------------------------------
# run_analysis — parameter validation
# ---------------------------------------------------------------------------


def test_run_analysis_raises_for_zero_generations(tmp_path):
    with pytest.raises(ValueError, match="generations must be positive"):
        run_analysis(generations=0, out_dir=str(tmp_path))


def test_run_analysis_raises_for_zero_n_trials(tmp_path):
    with pytest.raises(ValueError, match="n-trials must be positive"):
        run_analysis(n_trials=0, out_dir=str(tmp_path))


def test_run_analysis_raises_for_negative_cell_index(tmp_path):
    with pytest.raises(ValueError, match="cell-index must be non-negative"):
        run_analysis(cell_index=-1, out_dir=str(tmp_path))


def test_run_analysis_raises_for_empty_generation_index_values(tmp_path):
    with pytest.raises(ValueError, match="generation-index-values must contain"):
        run_analysis(generation_index_values=[], out_dir=str(tmp_path))


def test_run_analysis_raises_for_empty_dispersal_values(tmp_path):
    with pytest.raises(ValueError, match="dispersal-values must contain"):
        run_analysis(
            generation_index_values=[0],
            dispersal_values=[],
            out_dir=str(tmp_path),
        )


def test_run_analysis_raises_for_empty_distance_values(tmp_path):
    with pytest.raises(ValueError, match="distance-values must contain"):
        run_analysis(
            generation_index_values=[0],
            dispersal_values=[0.0],
            distance_values=[],
            out_dir=str(tmp_path),
        )


def test_run_analysis_raises_for_negative_generation_index(tmp_path):
    with pytest.raises(ValueError, match="generation indices must be non-negative"):
        run_analysis(generation_index_values=[-1], out_dir=str(tmp_path))


def test_run_analysis_raises_when_generation_index_exceeds_generations(tmp_path):
    with pytest.raises(ValueError, match="generation indices cannot exceed total generations"):
        run_analysis(
            generations=4,
            generation_index_values=[5],
            out_dir=str(tmp_path),
        )


def test_run_analysis_raises_for_dispersal_above_one(tmp_path):
    with pytest.raises(ValueError, match="dispersal values must be between 0 and 1"):
        run_analysis(
            generation_index_values=[0],
            dispersal_values=[1.5],
            out_dir=str(tmp_path),
        )


def test_run_analysis_raises_for_distance_above_one(tmp_path):
    with pytest.raises(ValueError, match="distance values must be between 0 and 1"):
        run_analysis(
            generation_index_values=[0],
            dispersal_values=[0.0],
            distance_values=[2.0],
            out_dir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# run_with_defaults
# ---------------------------------------------------------------------------


def test_run_with_defaults_calls_run_analysis_and_prints_elapsed(monkeypatch, capsys):
    calls = []

    def fast_run_analysis(**kwargs):
        calls.append(kwargs)
        return [], []

    monkeypatch.setattr(trials, "run_analysis", fast_run_analysis)
    trials.run_with_defaults()
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "Elapsed" in out
