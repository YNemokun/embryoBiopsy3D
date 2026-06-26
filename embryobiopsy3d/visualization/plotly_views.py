"""
Plotly figure builders for embryo visualization scenes.
"""

from __future__ import annotations

import math
from typing import Iterable

import plotly.graph_objects as go

from .scene import EmbryoScene, SceneNode

ANEUPLOID_COLOR = "#d62728"
EUPLOID_FACE = "#ffffff"
BASE_EDGE = "#222222"
BIOPSY_1_COLOR = "#1f77b4"
BIOPSY_2_COLOR = "#ff7f0e"
SPHERE_COLOR = "#dddddd"
EDGE_COLOR = "#c8c8c8"
SIBLING_COLOR = "#d9d9d9"


def _node_lookup(scene: EmbryoScene) -> dict[str, SceneNode]:
    """Build a UUID → SceneNode lookup dict from *scene*."""
    return {node.id: node for node in scene.nodes}


def _edge_trace_3d(
    scene: EmbryoScene, edge_ids, *, color: str, width: float, opacity: float
):
    """Return a 3-D line trace connecting the provided edges.

    Args:
        scene: Source scene containing all node positions.
        edge_ids: Iterable of :class:`~scene.SceneEdge` objects to draw.
        color: Line color string.
        width: Line width in pixels.
        opacity: Line opacity in ``[0, 1]``.

    Returns:
        A :class:`plotly.graph_objects.Scatter3d` trace with ``mode="lines"``.
    """
    nodes = _node_lookup(scene)
    xs, ys, zs = [], [], []
    for edge in edge_ids:
        source = nodes[edge.source_id]
        target = nodes[edge.target_id]
        if None in (source.x, source.y, source.z, target.x, target.y, target.z):
            continue
        xs.extend([source.x, target.x, None])
        ys.extend([source.y, target.y, None])
        zs.extend([source.z, target.z, None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line={"color": color, "width": width},
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


def _edge_trace_2d(
    scene: EmbryoScene, edge_ids, *, color: str, width: float, opacity: float
):
    """Return a 2-D line trace connecting the provided lineage edges.

    Args:
        scene: Source scene providing lineage layout coordinates.
        edge_ids: Iterable of :class:`~scene.SceneEdge` objects to draw.
        color: Line color string.
        width: Line width in pixels.
        opacity: Line opacity in ``[0, 1]``.

    Returns:
        A :class:`plotly.graph_objects.Scatter` trace with ``mode="lines"``.
    """
    nodes = _node_lookup(scene)
    xs, ys = [], []
    for edge in edge_ids:
        source = nodes[edge.source_id]
        target = nodes[edge.target_id]
        xs.extend([source.lineage_x, target.lineage_x, None])
        ys.extend([source.lineage_y, target.lineage_y, None])
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"color": color, "width": width},
        opacity=opacity,
        hoverinfo="skip",
        showlegend=False,
    )


def _format_node_hover(node: SceneNode) -> str:
    """Build an HTML hover string for a single scene node.

    Args:
        node: The :class:`~scene.SceneNode` to describe.

    Returns:
        Multi-line HTML string (``"<br>"``-joined) for use as a Plotly
        ``hovertemplate`` entry.
    """
    status = "aneuploid" if node.is_aneuploid else "euploid"
    bits = [
        f"id={node.id[:8]}",
        f"generation={node.generation}",
        f"index={node.generation_index}",
        status,
    ]
    if node.in_first_biopsy:
        bits.append("biopsy=first")
    if node.in_second_biopsy:
        bits.append("biopsy=second")
    if node.is_first_center:
        bits.append("center=1")
    if node.is_second_center:
        bits.append("center=2")
    return "<br>".join(bits)


def _points_3d(nodes: Iterable[SceneNode], *, size: float, name: str):
    """Return a 3-D scatter trace for a collection of nodes coloured by aneuploid status.

    Args:
        nodes: Nodes to render as sphere points.
        size: Marker size in pixels.
        name: Legend label for the trace.

    Returns:
        A :class:`plotly.graph_objects.Scatter3d` trace with ``mode="markers"``.
    """
    nodes = list(nodes)
    return go.Scatter3d(
        x=[node.x for node in nodes],
        y=[node.y for node in nodes],
        z=[node.z for node in nodes],
        mode="markers",
        name=name,
        hovertemplate=[_format_node_hover(node) for node in nodes],
        marker={
            "size": size,
            "color": [
                ANEUPLOID_COLOR if node.is_aneuploid else EUPLOID_FACE for node in nodes
            ],
            "line": {"color": BASE_EDGE, "width": 2},
        },
    )


def _rings_3d(
    nodes: Iterable[SceneNode],
    *,
    color: str,
    size: float,
    name: str,
    line_width: float = 12,
):
    """Hollow markers used as biopsy rings; size dominates visibility in 3D."""
    nodes = list(nodes)
    return go.Scatter3d(
        x=[node.x for node in nodes],
        y=[node.y for node in nodes],
        z=[node.z for node in nodes],
        mode="markers",
        name=name,
        hoverinfo="skip",
        marker={
            "size": size,
            "color": "rgba(0,0,0,0)",
            "line": {"color": color, "width": line_width},
        },
    )


def _labels_3d(nodes: Iterable[SceneNode], *, text: str, color: str):
    """Return a 3-D text trace labelling each node with *text*.

    Args:
        nodes: Nodes to label.
        text: Label string placed at each node position.
        color: Text color.

    Returns:
        A :class:`plotly.graph_objects.Scatter3d` trace with ``mode="text"``.
    """
    nodes = list(nodes)
    return go.Scatter3d(
        x=[node.x for node in nodes],
        y=[node.y for node in nodes],
        z=[node.z for node in nodes],
        mode="text",
        text=[text for _ in nodes],
        textfont={"color": color, "size": 12},
        hoverinfo="skip",
        showlegend=False,
    )


def _sphere_wireframe_traces():
    """Return a list of 3-D line traces forming a faint unit-sphere wireframe.

    Returns:
        List of :class:`plotly.graph_objects.Scatter3d` traces — one per
        longitude and latitude ring.
    """
    traces = []
    longitudes = [-120, -60, 0, 60, 120]
    latitudes = [-60, -30, 0, 30, 60]

    for lon_deg in longitudes:
        lon = math.radians(lon_deg)
        xs, ys, zs = [], [], []
        for step in range(61):
            theta = 2 * math.pi * step / 60
            xs.append(math.cos(theta) * math.cos(lon))
            ys.append(math.sin(theta) * math.cos(lon))
            zs.append(math.sin(lon))
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": SPHERE_COLOR, "width": 2},
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for lat_deg in latitudes:
        lat = math.radians(lat_deg)
        xs, ys, zs = [], [], []
        radius = math.cos(lat)
        z = math.sin(lat)
        for step in range(61):
            theta = 2 * math.pi * step / 60
            xs.append(radius * math.cos(theta))
            ys.append(radius * math.sin(theta))
            zs.append(z)
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": SPHERE_COLOR, "width": 2},
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return traces


_EYE_X0 = 1.5
_EYE_Y0 = 1.2
_EYE_Z = 0.9


def make_embryo_figure(scene: EmbryoScene, *, title: str = "Embryo") -> go.Figure:
    """Build an interactive 3-D embryo visualization from a scene.

    Renders leaf cells on the unit sphere with a faint wireframe, optional
    sibling edges, and biopsy ring markers when the scene contains a
    :class:`~scene.SceneBiopsy`.

    Args:
        scene: Fully built :class:`~scene.EmbryoScene`.
        title: Figure title shown above the plot.

    Returns:
        A :class:`plotly.graph_objects.Figure` ready for display or export.
    """
    figure = go.Figure()
    nodes = [node for node in scene.nodes if node.is_leaf and node.x is not None]
    first = [node for node in nodes if node.in_first_biopsy]
    second = [node for node in nodes if node.in_second_biopsy]
    first_center = [node for node in nodes if node.is_first_center]
    second_center = [node for node in nodes if node.is_second_center]

    for trace in _sphere_wireframe_traces():
        figure.add_trace(trace)
    if scene.sibling_edges:
        figure.add_trace(
            _edge_trace_3d(
                scene,
                scene.sibling_edges,
                color=SIBLING_COLOR,
                width=2,
                opacity=0.5,
            )
        )
    if nodes:
        figure.add_trace(_points_3d(nodes, size=6, name="Cells"))
    # Ring markers must be larger than leaf dots or the 3D outline reads as a hairline.
    _ring_size = 14
    _ring_line = 14
    if first:
        figure.add_trace(
            _rings_3d(
                first,
                color=BIOPSY_1_COLOR,
                size=_ring_size,
                name="First biopsy",
                line_width=_ring_line,
            )
        )
    if second:
        figure.add_trace(
            _rings_3d(
                second,
                color=BIOPSY_2_COLOR,
                size=_ring_size,
                name="Second biopsy",
                line_width=_ring_line,
            )
        )
    if first_center:
        figure.add_trace(_labels_3d(first_center, text="1", color=BIOPSY_1_COLOR))
    if second_center:
        figure.add_trace(_labels_3d(second_center, text="2", color=BIOPSY_2_COLOR))

    figure.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "b": 0, "t": 45},
        scene={
            "xaxis": {"visible": False, "range": [-1.15, 1.15]},
            "yaxis": {"visible": False, "range": [-1.15, 1.15]},
            "zaxis": {"visible": False, "range": [-1.15, 1.15]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": _EYE_X0, "y": _EYE_Y0, "z": _EYE_Z}},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01},
    )
    return figure


def make_lineage_figure(
    scene: EmbryoScene, *, title: str = "Lineage tree"
) -> go.Figure:
    """Build a 2-D lineage-tree visualization from a scene.

    Renders every cell as a marker at its ``(lineage_x, generation)`` position
    with biopsy ring overlays and center labels when the scene contains a
    :class:`~scene.SceneBiopsy`.

    Args:
        scene: Fully built :class:`~scene.EmbryoScene`.
        title: Figure title shown above the plot.

    Returns:
        A :class:`plotly.graph_objects.Figure` ready for display or export.
    """
    figure = go.Figure()
    nodes = scene.nodes
    first = [node for node in nodes if node.in_first_biopsy]
    second = [node for node in nodes if node.in_second_biopsy]
    first_center = [node for node in nodes if node.is_first_center]
    second_center = [node for node in nodes if node.is_second_center]

    if scene.lineage_edges:
        figure.add_trace(
            _edge_trace_2d(
                scene, scene.lineage_edges, color=EDGE_COLOR, width=1.5, opacity=1.0
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[node.lineage_x for node in nodes],
            y=[node.lineage_y for node in nodes],
            mode="markers",
            name="Cells",
            hovertemplate=[_format_node_hover(node) for node in nodes],
            marker={
                "size": [11 if node.is_leaf else 9 for node in nodes],
                "color": [
                    ANEUPLOID_COLOR if node.is_aneuploid else EUPLOID_FACE
                    for node in nodes
                ],
                "line": {"color": BASE_EDGE, "width": 1.5},
            },
        )
    )

    if first:
        figure.add_trace(
            go.Scatter(
                x=[node.lineage_x for node in first],
                y=[node.lineage_y for node in first],
                mode="markers",
                name="First biopsy",
                hoverinfo="skip",
                marker={
                    "size": [15 if node.is_leaf else 13 for node in first],
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": BIOPSY_1_COLOR, "width": 2.5},
                },
            )
        )
    if second:
        figure.add_trace(
            go.Scatter(
                x=[node.lineage_x for node in second],
                y=[node.lineage_y for node in second],
                mode="markers",
                name="Second biopsy",
                hoverinfo="skip",
                marker={
                    "size": [15 if node.is_leaf else 13 for node in second],
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": BIOPSY_2_COLOR, "width": 2.5},
                },
            )
        )
    if first_center:
        figure.add_trace(
            go.Scatter(
                x=[node.lineage_x for node in first_center],
                y=[node.lineage_y for node in first_center],
                mode="text",
                text=["1" for _ in first_center],
                textfont={"color": BIOPSY_1_COLOR, "size": 12},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if second_center:
        figure.add_trace(
            go.Scatter(
                x=[node.lineage_x for node in second_center],
                y=[node.lineage_y for node in second_center],
                mode="text",
                text=["2" for _ in second_center],
                textfont={"color": BIOPSY_2_COLOR, "size": 12},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    max_y = max(node.lineage_y for node in nodes) if nodes else 0.0
    max_level = int(round(max_y))
    figure.update_layout(
        title=title,
        margin={"l": 10, "r": 10, "b": 10, "t": 45},
        xaxis={"visible": False},
        yaxis={
            "title": "Generation",
            "tickmode": "array",
            "tickvals": list(range(max_level + 1)),
            "ticktext": [str(max_level - idx) for idx in range(max_level + 1)],
            "range": [-0.2, max_y + 0.2],
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        plot_bgcolor="white",
    )
    return figure
