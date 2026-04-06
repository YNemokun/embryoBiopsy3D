"""Tests for visualization scene helpers."""

import math

from embryobiopsy3d.visualization.scene import (
    build_demo_scene,
    scene_leaf_rows,
    scene_summary_rows,
    serialize_scene,
)


def test_build_demo_scene_random_rebiopsy_serializes_plain_data():
    """Random-error scene builds with serializable biopsy metadata."""
    scene = build_demo_scene(
        generations=4,
        dispersal=0.3,
        error_mode="random",
        meio_rate=0.2,
        mito_rate=0.1,
        error_seed=5,
        placement_seed=7,
        rebiopsy_distance=0.5,
        biopsy_seed=3,
    )

    payload = serialize_scene(scene)
    assert payload["biopsy"] is not None
    assert payload["metadata"]["error_mode"] == "random"
    assert payload["biopsy"]["requested_distance"] == math.pi * 0.5
    assert all(isinstance(node["id"], str) for node in payload["nodes"])


def test_build_demo_scene_fixed_generation_marks_targeted_subtree():
    """Fixed-generation mode records aneuploid descendants in the scene."""
    scene = build_demo_scene(
        generations=4,
        dispersal=0.0,
        error_mode="fixed_generation",
        target_generation=2,
        target_index=1,
        placement_seed=9,
        include_subtree=True,
    )

    rows = scene_leaf_rows(scene)
    assert rows
    assert any(row["is_aneuploid"] for row in rows)
    assert all(row["generation"] == 4 for row in rows)


def test_scene_summary_rows_include_biopsy_metrics():
    """Summary rows expose both simulation and biopsy values."""
    scene = build_demo_scene(
        generations=3,
        dispersal=0.1,
        error_mode="none",
        placement_seed=2,
        rebiopsy_distance=0.25,
        biopsy_seed=8,
    )

    metrics = {row["metric"]: row["value"] for row in scene_summary_rows(scene)}
    assert metrics["error_mode"] == "none"
    assert metrics["placement_seed"] == 2
    assert metrics["biopsy_seed"] == 8
    assert "first_biopsy_category" in metrics
