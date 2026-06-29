"""Tests for embryobiopsy3d/visualization/plotly_views.py."""

import plotly.graph_objects as go
import pytest

from embryobiopsy3d.visualization.plotly_views import (
    _edge_trace_2d,
    _edge_trace_3d,
    _format_node_hover,
    _labels_3d,
    _node_lookup,
    _points_3d,
    _rings_3d,
    _sphere_wireframe_traces,
    make_embryo_figure,
    make_lineage_figure,
)
from embryobiopsy3d.visualization.scene import (
    EmbryoScene,
    SceneEdge,
    SceneNode,
    build_demo_scene,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _node(
    nid,
    *,
    x=1.0,
    y=0.0,
    z=0.0,
    is_aneuploid=False,
    is_leaf=True,
    in_first_biopsy=False,
    in_second_biopsy=False,
    is_first_center=False,
    is_second_center=False,
):
    return SceneNode(
        id=nid,
        generation=1,
        generation_index=0,
        is_leaf=is_leaf,
        is_aneuploid=is_aneuploid,
        lineage_x=0.5,
        lineage_y=1.0,
        x=x,
        y=y,
        z=z,
        in_first_biopsy=in_first_biopsy,
        in_second_biopsy=in_second_biopsy,
        is_first_center=is_first_center,
        is_second_center=is_second_center,
    )


def _two_node_scene(n1_x=1.0, n1_y=0.0, n1_z=0.0, n2_x=0.0, n2_y=1.0, n2_z=0.0):
    """Minimal two-node scene with one lineage edge and one sibling edge."""
    n1 = _node("nodeid01", x=n1_x, y=n1_y, z=n1_z)
    n2 = _node("nodeid02", x=n2_x, y=n2_y, z=n2_z)
    edge = SceneEdge(source_id="nodeid01", target_id="nodeid02")
    return EmbryoScene(nodes=[n1, n2], lineage_edges=[edge], sibling_edges=[edge])


@pytest.fixture(scope="module")
def biopsy_scene():
    """Full scene with rebiopsy — covers first/second biopsy and center flags."""
    return build_demo_scene(
        generations=3,
        dispersal=0.3,
        error_mode="none",
        placement_seed=42,
        rebiopsy_distance=0.5,
        biopsy_seed=7,
    )


# ---------------------------------------------------------------------------
# _node_lookup
# ---------------------------------------------------------------------------


def test_node_lookup_indexes_every_node_by_id():
    scene = _two_node_scene()
    lookup = _node_lookup(scene)
    assert set(lookup.keys()) == {n.id for n in scene.nodes}
    for node in scene.nodes:
        assert lookup[node.id] is node


# ---------------------------------------------------------------------------
# _edge_trace_3d
# ---------------------------------------------------------------------------


def test_edge_trace_3d_connects_two_nodes_with_valid_coords():
    scene = _two_node_scene()
    trace = _edge_trace_3d(scene, scene.sibling_edges, color="#ff0000", width=2, opacity=0.5)
    assert isinstance(trace, go.Scatter3d)
    assert trace.mode == "lines"
    # Two nodes produce [x1, x2, None] — three coordinate values per dimension.
    assert len(trace.x) == 3


def test_edge_trace_3d_skips_edge_when_source_coordinate_is_none():
    scene = _two_node_scene(n1_x=None, n1_y=None, n1_z=None)
    trace = _edge_trace_3d(scene, scene.sibling_edges, color="#ff0000", width=2, opacity=0.5)
    # The None-coord source means the segment is skipped entirely.
    assert len(trace.x) == 0


# ---------------------------------------------------------------------------
# _edge_trace_2d
# ---------------------------------------------------------------------------


def test_edge_trace_2d_returns_scatter_with_line_mode():
    scene = _two_node_scene()
    trace = _edge_trace_2d(scene, scene.lineage_edges, color="#aaaaaa", width=1.5, opacity=1.0)
    assert isinstance(trace, go.Scatter)
    assert trace.mode == "lines"
    assert len(trace.x) == 3  # [x1, x2, None]


# ---------------------------------------------------------------------------
# _format_node_hover
# ---------------------------------------------------------------------------


def test_format_node_hover_euploid_with_no_biopsy_flags():
    node = _node("abcdef01")
    text = _format_node_hover(node)
    assert "euploid" in text
    assert "aneuploid" not in text
    assert "biopsy" not in text
    assert "center" not in text


def test_format_node_hover_aneuploid_with_all_biopsy_flags():
    node = _node(
        "abcdef02",
        is_aneuploid=True,
        in_first_biopsy=True,
        in_second_biopsy=True,
        is_first_center=True,
        is_second_center=True,
    )
    text = _format_node_hover(node)
    assert "aneuploid" in text
    assert "biopsy=first" in text
    assert "biopsy=second" in text
    assert "center=1" in text
    assert "center=2" in text


# ---------------------------------------------------------------------------
# _points_3d
# ---------------------------------------------------------------------------


def test_points_3d_uses_different_colors_for_euploid_and_aneuploid():
    euploid = _node("nodeid01", is_aneuploid=False)
    aneuploid = _node("nodeid02", is_aneuploid=True)
    trace = _points_3d([euploid, aneuploid], size=6, name="Cells")
    assert isinstance(trace, go.Scatter3d)
    assert len(trace.x) == 2
    assert trace.marker.color[0] != trace.marker.color[1]


# ---------------------------------------------------------------------------
# _rings_3d
# ---------------------------------------------------------------------------


def test_rings_3d_returns_hollow_marker_scatter():
    nodes = [_node("nodeid01"), _node("nodeid02")]
    trace = _rings_3d(nodes, color="#1f77b4", size=14, name="First biopsy")
    assert isinstance(trace, go.Scatter3d)
    assert trace.marker.color == "rgba(0,0,0,0)"
    assert trace.name == "First biopsy"


# ---------------------------------------------------------------------------
# _labels_3d
# ---------------------------------------------------------------------------


def test_labels_3d_places_text_at_each_node():
    nodes = [_node("nodeid01"), _node("nodeid02")]
    trace = _labels_3d(nodes, text="1", color="#1f77b4")
    assert isinstance(trace, go.Scatter3d)
    assert trace.mode == "text"
    assert list(trace.text) == ["1", "1"]


# ---------------------------------------------------------------------------
# _sphere_wireframe_traces
# ---------------------------------------------------------------------------


def test_sphere_wireframe_traces_returns_ten_scatter3d_traces():
    traces = _sphere_wireframe_traces()
    assert len(traces) == 10  # 5 longitude rings + 5 latitude rings
    assert all(isinstance(t, go.Scatter3d) for t in traces)
    assert all(t.mode == "lines" for t in traces)


# ---------------------------------------------------------------------------
# make_embryo_figure
# ---------------------------------------------------------------------------


def test_make_embryo_figure_empty_scene_returns_valid_figure():
    scene = EmbryoScene(nodes=[], lineage_edges=[], sibling_edges=[])
    fig = make_embryo_figure(scene, title="Empty test")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Empty test"
    # Only wireframe traces present (10 sphere traces, no node traces).
    assert len(fig.data) == 10


def test_make_embryo_figure_full_scene_adds_biopsy_ring_traces(biopsy_scene):
    fig = make_embryo_figure(biopsy_scene)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if getattr(t, "name", None)]
    assert any("First biopsy" in n for n in trace_names)
    assert any("Second biopsy" in n for n in trace_names)


def test_make_embryo_figure_full_scene_adds_sibling_edge_trace(biopsy_scene):
    # Sibling edges should produce one extra line trace beyond the wireframe.
    fig = make_embryo_figure(biopsy_scene)
    line_traces = [t for t in fig.data if getattr(t, "mode", None) == "lines"]
    # 10 wireframe + 1 sibling-edge trace
    assert len(line_traces) >= 11


# ---------------------------------------------------------------------------
# make_lineage_figure
# ---------------------------------------------------------------------------


def test_make_lineage_figure_empty_nodes_returns_valid_figure():
    scene = EmbryoScene(nodes=[], lineage_edges=[], sibling_edges=[])
    fig = make_lineage_figure(scene)
    assert isinstance(fig, go.Figure)
    # One Scatter trace for the (empty) node list is always added.
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) >= 1


def test_make_lineage_figure_full_scene_adds_biopsy_ring_traces(biopsy_scene):
    fig = make_lineage_figure(biopsy_scene)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if getattr(t, "name", None)]
    assert any("First biopsy" in n for n in trace_names)
    assert any("Second biopsy" in n for n in trace_names)
