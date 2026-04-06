"""
Streamlit demo for embryo construction and rebiopsy visualization.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from embryobiopsy3d.visualization.plotly_views import (
    make_embryo_figure,
    make_lineage_figure,
)
from embryobiopsy3d.visualization.scene import (
    build_demo_scene,
    scene_leaf_rows,
    scene_summary_rows,
    serialize_scene,
)


st.set_page_config(page_title="embryoBiopsy3D demo", layout="wide")


def _format_summary_value(value):
    """Return a stable display string for mixed-type summary values."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _summary_dataframe(scene):
    """Build an Arrow-friendly summary table for Streamlit."""
    frame = pd.DataFrame(scene_summary_rows(scene))
    frame["value"] = frame["value"].map(_format_summary_value)
    return frame


def _sidebar_controls() -> dict:
    st.sidebar.header("Simulation controls")
    generations = st.sidebar.slider("Generations", min_value=1, max_value=9, value=8)
    dispersal = st.sidebar.slider(
        "Dispersal", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )
    error_mode = st.sidebar.radio(
        "Error mode",
        options=["none", "random", "fixed_generation"],
        format_func=lambda value: {
            "none": "No error",
            "random": "Random meiotic + mitotic",
            "fixed_generation": "Set aneuploidy at one generation",
        }[value],
    )

    meio_rate = 0.1
    mito_rate = 0.05
    error_seed = 42
    target_generation = None
    target_index = 0
    include_subtree = True

    if error_mode == "random":
        meio_rate = st.sidebar.slider("Meiotic error rate", 0.0, 1.0, 0.1, 0.01)
        mito_rate = st.sidebar.slider("Mitotic error rate", 0.0, 1.0, 0.05, 0.01)
        error_seed = st.sidebar.number_input("Error seed", value=42, step=1)
    elif error_mode == "fixed_generation":
        target_generation = st.sidebar.slider(
            "Error generation",
            min_value=0,
            max_value=generations,
            value=min(3, generations),
        )
        max_index = (2**target_generation) - 1 if target_generation else 0
        target_index = st.sidebar.slider(
            "Error index within generation",
            min_value=0,
            max_value=max_index,
            value=0,
        )
        include_subtree = st.sidebar.checkbox("Propagate through subtree", value=True)

    st.sidebar.header("Placement and sampling")
    placement_seed = st.sidebar.number_input("Placement seed", value=7, step=1)
    rebiopsy_distance = st.sidebar.slider(
        "Rebiopsy distance (fraction of pi)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    biopsy_seed = st.sidebar.number_input("Biopsy seed", value=11, step=1)

    return {
        "generations": generations,
        "dispersal": dispersal,
        "error_mode": error_mode,
        "meio_rate": meio_rate,
        "mito_rate": mito_rate,
        "error_seed": int(error_seed),
        "placement_seed": int(placement_seed),
        "target_generation": target_generation,
        "target_index": target_index,
        "include_subtree": include_subtree,
        "rebiopsy_distance": rebiopsy_distance,
        "biopsy_seed": int(biopsy_seed),
    }


def _show_scene(scene, *, title_prefix: str):
    left, right = st.columns([1.0, 1.2])
    with left:
        st.plotly_chart(
            make_embryo_figure(scene, title=f"{title_prefix}: embryo"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            make_lineage_figure(scene, title=f"{title_prefix}: lineage tree"),
            width="stretch",
        )


def _show_tables(scene):
    summary_col, leaves_col = st.columns([0.7, 1.3])
    with summary_col:
        st.subheader("Summary")
        st.dataframe(_summary_dataframe(scene), hide_index=True, width="stretch")
    with leaves_col:
        st.subheader("Leaf table")
        st.dataframe(
            pd.DataFrame(scene_leaf_rows(scene)), hide_index=True, width="stretch"
        )


def main():
    st.title("embryoBiopsy3D visualization demo")
    st.write(
        "Use the controls to compare embryo placement, lineage structure, and "
        "rebiopsy sampling under random or generation-targeted aneuploidy."
    )

    controls = _sidebar_controls()
    construction_scene = build_demo_scene(
        generations=controls["generations"],
        dispersal=controls["dispersal"],
        error_mode=controls["error_mode"],
        meio_rate=controls["meio_rate"],
        mito_rate=controls["mito_rate"],
        error_seed=controls["error_seed"],
        placement_seed=controls["placement_seed"],
        target_generation=controls["target_generation"],
        target_index=controls["target_index"],
        include_subtree=controls["include_subtree"],
    )
    rebiopsy_scene = build_demo_scene(**controls)

    construction_tab, rebiopsy_tab, raw_tab = st.tabs(
        ["Embryo construction", "Rebiopsy", "Scene data"]
    )

    with construction_tab:
        _show_scene(construction_scene, title_prefix="Construction")
        _show_tables(construction_scene)

    with rebiopsy_tab:
        _show_scene(rebiopsy_scene, title_prefix="Rebiopsy")
        _show_tables(rebiopsy_scene)
        if rebiopsy_scene.biopsy is not None:
            st.caption(
                "Biopsy centers are labeled 1 and 2. Blue and orange rings show the "
                "cells sampled in the first and second biopsy."
            )

    with raw_tab:
        st.subheader("Serialized scene contract")
        st.json(serialize_scene(rebiopsy_scene), expanded=False)


if __name__ == "__main__":
    main()
