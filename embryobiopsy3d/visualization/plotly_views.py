"""
Plotly figure builders for embryo visualization scenes.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import plotly.graph_objects as go

from .scene import EmbryoScene, SceneNode

ANEUPLOID_COLOR = "#d62728"
EUPLOID_FACE = "#ffffff"
BASE_EDGE = "#222222"
BIOPSY_1_COLOR = "#1f77b4"
BIOPSY_1_EUPLOID = "#aec7e8"
BIOPSY_2_COLOR = "#ff7f0e"
BIOPSY_2_EUPLOID = "#ffbb78"
SPHERE_COLOR = "#dddddd"
EDGE_COLOR = "#c8c8c8"
SIBLING_COLOR = "#d9d9d9"

# Palette for clade coloring — avoids the fixed colors already in use
# (red = aneuploid, white = euploid, blue = biopsy 1, orange = biopsy 2).
# Colors cycle if there are more progenitors than palette entries.
CLADE_PALETTE = [
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # teal
    "#bcbd22",  # yellow-green
    "#393b79",  # indigo
    "#d95f02",  # burnt orange
]

# Marker symbol used to flag a cell that divides erroneously (euploid progenitor).
_ERRONEOUS_SYMBOL_3D = "x"
_ERRONEOUS_SYMBOL_2D = "x"
_CIRCLE_3D = "circle"
_CIRCLE_2D = "circle"


def build_clade_colors(scene: EmbryoScene) -> dict[str, str]:
    """Return a stable progenitor-id → color mapping for clade visualization.

    Colors are assigned in BFS order (root first), so the same progenitor
    always gets the same color for a given scene regardless of how the
    function is called.
    """
    # scene.nodes is in BFS order; picking divides_erroneously in that order
    # gives a generation-stable assignment.
    progenitor_ids = [node.id for node in scene.nodes if node.divides_erroneously]
    return {
        pid: CLADE_PALETTE[i % len(CLADE_PALETTE)]
        for i, pid in enumerate(progenitor_ids)
    }


def _node_symbol_3d(node: SceneNode) -> str:
    return _ERRONEOUS_SYMBOL_3D if node.divides_erroneously else _CIRCLE_3D


def _node_symbol_2d(node: SceneNode) -> str:
    return _ERRONEOUS_SYMBOL_2D if node.divides_erroneously else _CIRCLE_2D


def _node_face_color(
    node: SceneNode,
    clade_colors: Optional[dict[str, str]] = None,
) -> str:
    """Resolve the fill color for a node.

    Default mode
    ------------
    Aneuploid → ANEUPLOID_COLOR (red); everything else → EUPLOID_FACE (white).

    Clade-color mode  (clade_colors is not None)
    --------------------------------------------
    - Erroneous progenitor (×) → its own clade color.
    - Aneuploid descendant with a known progenitor → that progenitor's clade color.
    - Aneuploid with no progenitor (meiotic / direct) → ANEUPLOID_COLOR.
    - Euploid → EUPLOID_FACE.
    """
    if clade_colors:
        if node.divides_erroneously:
            return clade_colors.get(node.id, ANEUPLOID_COLOR)
        if node.is_aneuploid and node.error_progenitor:
            return clade_colors.get(node.error_progenitor, ANEUPLOID_COLOR)
        if node.is_aneuploid:
            return ANEUPLOID_COLOR
    elif node.is_aneuploid:
        return ANEUPLOID_COLOR
    return EUPLOID_FACE


def _biopsy_face_color(
    node: SceneNode,
    euploid_color: str,
    aneuploid_color: str,
    clade_colors: Optional[dict[str, str]] = None,
) -> str:
    """Biopsy variant: euploid cells keep the biopsy hue; aneuploid cells
    use the clade color when clade mode is active."""
    if clade_colors:
        if node.divides_erroneously:
            return clade_colors.get(node.id, aneuploid_color)
        if node.is_aneuploid and node.error_progenitor:
            return clade_colors.get(node.error_progenitor, aneuploid_color)
        if node.is_aneuploid:
            return aneuploid_color
        return euploid_color
    return aneuploid_color if node.is_aneuploid else euploid_color


def _clade_legend_entries_2d(
    scene: EmbryoScene, clade_colors: dict[str, str]
) -> list[go.Scatter]:
    """Invisible scatter points that inject per-clade entries into the legend."""
    traces = []
    # Build a fast id→node lookup for label generation.
    node_map = {n.id: n for n in scene.nodes}
    for pid, color in clade_colors.items():
        prog = node_map.get(pid)
        label = (
            f"Clade: gen {prog.generation}, idx {prog.generation_index}"
            if prog
            else f"Clade: {pid[:8]}"
        )
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "color": color,
                    "size": 11,
                    "symbol": "circle",
                    "line": {"color": BASE_EDGE, "width": 1.5},
                },
                name=label,
                showlegend=True,
            )
        )
    return traces


def _clade_legend_entries_3d(
    scene: EmbryoScene, clade_colors: dict[str, str]
) -> list[go.Scatter3d]:
    """Invisible 3-D scatter points that inject per-clade legend entries."""
    traces = []
    node_map = {n.id: n for n in scene.nodes}
    for pid, color in clade_colors.items():
        prog = node_map.get(pid)
        label = (
            f"Clade: gen {prog.generation}, idx {prog.generation_index}"
            if prog
            else f"Clade: {pid[:8]}"
        )
        traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={"color": color, "size": 7, "symbol": "circle"},
                name=label,
                showlegend=True,
            )
        )
    return traces


def _node_lookup(scene: EmbryoScene) -> dict[str, SceneNode]:
    return {node.id: node for node in scene.nodes}


def _edge_trace_3d(
    scene: EmbryoScene, edge_ids, *, color: str, width: float, opacity: float
):
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
    if node.divides_erroneously:
        status = "euploid — divides erroneously"
    elif node.is_aneuploid:
        status = "aneuploid"
    else:
        status = "euploid"
    bits = [
        f"id={node.id[:8]}",
        f"generation={node.generation}",
        f"index={node.generation_index}",
        status,
    ]
    if node.error_progenitor:
        bits.append(f"progenitor={node.error_progenitor[:8]}")
    if node.in_first_biopsy:
        bits.append("biopsy=first")
    if node.in_second_biopsy:
        bits.append("biopsy=second")
    if node.is_first_center:
        bits.append("center=1")
    if node.is_second_center:
        bits.append("center=2")
    return "<br>".join(bits)


def _points_3d(
    nodes: Iterable[SceneNode],
    *,
    size: float,
    name: str,
    clade_colors: Optional[dict[str, str]] = None,
):
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
            "symbol": [_node_symbol_3d(node) for node in nodes],
            "color": [_node_face_color(node, clade_colors) for node in nodes],
            "line": {"color": BASE_EDGE, "width": 2},
        },
    )


def _points_3d_biopsy(
    nodes: Iterable[SceneNode],
    *,
    euploid_color: str,
    aneuploid_color: str,
    size: float,
    name: str,
    clade_colors: Optional[dict[str, str]] = None,
):
    """Cell markers recolored and enlarged to indicate biopsy membership.

    In default mode, aneuploid cells use *aneuploid_color* and euploid cells use
    *euploid_color*.  In clade mode, aneuploid cells are colored by their clade
    while euploid cells retain *euploid_color* (the biopsy hue).
    """
    nodes = list(nodes)
    colors = [
        _biopsy_face_color(node, euploid_color, aneuploid_color, clade_colors)
        for node in nodes
    ]
    return go.Scatter3d(
        x=[node.x for node in nodes],
        y=[node.y for node in nodes],
        z=[node.z for node in nodes],
        mode="markers",
        name=name,
        hovertemplate=[_format_node_hover(node) for node in nodes],
        marker={
            "size": size,
            "symbol": [_node_symbol_3d(node) for node in nodes],
            "color": colors,
            "line": {"color": BASE_EDGE, "width": 3},
        },
    )


def _labels_3d(nodes: Iterable[SceneNode], *, text: str, color: str):
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


def make_embryo_figure(
    scene: EmbryoScene,
    *,
    title: str = "Embryo",
    clade_colors: Optional[dict[str, str]] = None,
) -> go.Figure:
    """Build an interactive 3D embryo view.

    Parameters
    ----------
    clade_colors : dict or None
        When provided (use :func:`build_clade_colors` to create it), each
        aneuploid clade is rendered in a distinct color so independent mitotic
        errors can be counted visually.  Pass ``None`` for the default red/white
        scheme.
    """
    figure = go.Figure()
    nodes = [node for node in scene.nodes if node.is_leaf and node.x is not None]
    first = [node for node in nodes if node.in_first_biopsy]
    second = [node for node in nodes if node.in_second_biopsy]
    first_center = [node for node in nodes if node.is_first_center]
    second_center = [node for node in nodes if node.is_second_center]
    biopsy_ids = {n.id for n in first} | {n.id for n in second}
    plain = [node for node in nodes if node.id not in biopsy_ids]

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
    if plain:
        figure.add_trace(
            _points_3d(plain, size=6, name="Cells", clade_colors=clade_colors)
        )
    if first:
        figure.add_trace(
            _points_3d_biopsy(
                first,
                euploid_color=BIOPSY_1_EUPLOID,
                aneuploid_color=BIOPSY_1_COLOR,
                size=9,
                name="First biopsy",
                clade_colors=clade_colors,
            )
        )
    if second:
        figure.add_trace(
            _points_3d_biopsy(
                second,
                euploid_color=BIOPSY_2_EUPLOID,
                aneuploid_color=BIOPSY_2_COLOR,
                size=9,
                name="Second biopsy",
                clade_colors=clade_colors,
            )
        )
    if first_center:
        figure.add_trace(_labels_3d(first_center, text="1", color=BIOPSY_1_COLOR))
    if second_center:
        figure.add_trace(_labels_3d(second_center, text="2", color=BIOPSY_2_COLOR))
    if clade_colors:
        for trace in _clade_legend_entries_3d(scene, clade_colors):
            figure.add_trace(trace)

    figure.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "b": 120 if clade_colors else 0, "t": 45},
        scene={
            "xaxis": {"visible": False, "range": [-1.15, 1.15]},
            "yaxis": {"visible": False, "range": [-1.15, 1.15]},
            "zaxis": {"visible": False, "range": [-1.15, 1.15]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": _EYE_X0, "y": _EYE_Y0, "z": _EYE_Z}},
        },
        legend=(
            {
                "orientation": "h",
                "yanchor": "top",
                "y": -0.02,
                "xanchor": "center",
                "x": 0.5,
            }
            if clade_colors
            else {"orientation": "h", "yanchor": "bottom", "y": 0.01}
        ),
    )
    return figure


def make_lineage_figure(
    scene: EmbryoScene,
    *,
    title: str = "Lineage tree",
    clade_colors: Optional[dict[str, str]] = None,
) -> go.Figure:
    """Build a 2D lineage tree view.

    Parameters
    ----------
    clade_colors : dict or None
        When provided, each aneuploid clade is rendered in a distinct color.
        Build via :func:`build_clade_colors`.
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
                "symbol": [_node_symbol_2d(node) for node in nodes],
                "color": [_node_face_color(node, clade_colors) for node in nodes],
                "line": {"color": BASE_EDGE, "width": 1.5},
            },
        )
    )
    if clade_colors:
        for trace in _clade_legend_entries_2d(scene, clade_colors):
            figure.add_trace(trace)

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
        margin={"l": 10, "r": 10, "b": 120 if clade_colors else 10, "t": 45},
        xaxis={"visible": False},
        yaxis={
            "title": "Generation",
            "tickmode": "array",
            "tickvals": list(range(max_level + 1)),
            "ticktext": [str(max_level - idx) for idx in range(max_level + 1)],
            "range": [-0.2, max_y + 0.2],
        },
        legend=(
            {
                "orientation": "h",
                "yanchor": "top",
                "y": -0.05,
                "xanchor": "center",
                "x": 0.5,
            }
            if clade_colors
            else {"orientation": "h", "yanchor": "bottom", "y": 0.99}
        ),
        plot_bgcolor="white",
    )
    return figure
