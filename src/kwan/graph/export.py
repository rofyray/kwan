"""Export the curriculum graph as one interactive HTML page (`graph.html`), drawn with pyvis.

Each indicator is a node, coloured and shaped by strand. Grades are columns, from B1 on the left,
and the indicators of one content standard sit on one row, so an edge to a later grade runs
left to right. Rule edges are solid lines; augmented edges are dashed. Tick boxes above the picture
hide and show grades and strands. Hovering over a node shows its code and short label; the
curriculum description is shown only when the graph is in full-text mode.

`graph_html(G)` returns the page as a string (the demo embeds it); `write_html(G, path)` writes it.
pyvis is imported only when a page is made (`uv add 'kwan[demo]'`).
"""

from __future__ import annotations

import html
import json
import os
import re
import textwrap
from collections import Counter, defaultdict
from collections.abc import Collection, Hashable
from pathlib import Path
from typing import Any

import networkx as nx

from kwan import ids
from kwan.graph import schema

__all__ = ["EDGE_STYLE", "STRAND_STYLE", "graph_html", "layout", "node_tooltip", "write_html"]

# strand number -> (fill colour, node shape). The four colours were checked for colour-vision
# deficiency separation as a set; the shape repeats the strand so colour is never the only cue.
STRAND_STYLE: dict[int, tuple[str, str]] = {
    1: ("#2a78d6", "dot"),
    2: ("#eda100", "square"),
    3: ("#e87ba4", "triangle"),
    4: ("#008300", "diamond"),
}
OTHER_STYLE = ("#8a8984", "dot")
SHAPE_GLYPH = {"dot": "●", "square": "■", "triangle": "▲", "diamond": "◆"}
INK = "#52514e"
MUTED = "#b5b4ae"
SURFACE = "#fcfcfb"
# edge type -> vis-network edge options; only augmented edges are dashed
EDGE_STYLE: dict[str, dict[str, Any]] = {
    "sequence": {"color": INK, "width": 1.5, "dashes": False},
    "vertical": {"color": INK, "width": 3, "dashes": False},
    "bridge": {"color": "#8a8984", "width": 1, "dashes": False},
    "augmented": {"color": "#6250d6", "width": 2, "dashes": [8, 6]},
}
EDGE_MEANING = {
    "sequence": "next indicator in the same content standard",
    "vertical": "same content standard in a later grade",
    "bridge": "next content standard in the same sub-strand",
    "augmented": "suggested by the decision model and accepted by a reviewer",
}
DX, DY, STRAND_GAP, GRADE_GAP = 95, 70, 50, 140


def _cs_tuple(key: Any) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in str(key).split("."))
    except ValueError:
        return (10**6,)


def layout(G: nx.DiGraph) -> dict[Hashable, tuple[int, int]]:
    """A fixed position for every node: one column block per grade, one row per content standard."""
    by_cell: dict[tuple[int, tuple[int, ...]], list[Hashable]] = defaultdict(list)
    for n in sorted(G, key=schema.sort_key):
        d = G.nodes[n]
        by_cell[(int(d.get("grade") or 0), _cs_tuple(d.get("cs_key", "")))].append(n)
    rows = sorted({cs for _, cs in by_cell})
    row_y: dict[tuple[int, ...], int] = {}
    y = 0
    for i, cs in enumerate(rows):
        if i and cs[:1] != rows[i - 1][:1]:
            y += STRAND_GAP
        row_y[cs] = y
        y += DY
    width = Counter[int]()
    for (grade, _), members in by_cell.items():
        width[grade] = max(width[grade], len(members))
    col_x: dict[int, int] = {}
    x = 0
    for grade in sorted(width):
        col_x[grade] = x
        x += width[grade] * DX + GRADE_GAP
    pos: dict[Hashable, tuple[int, int]] = {}
    for (grade, cs), members in by_cell.items():
        for i, n in enumerate(members):
            pos[n] = (col_x[grade] + i * DX, row_y[cs])
    return pos


def node_tooltip(node_id: Hashable, data: dict[str, Any], text_mode: str) -> str:
    """The hover text of a node: code and short label, plus the description in full-text mode only."""
    code = data.get("code") or ids.code_of(str(node_id))
    label = data.get("short_label") or ""
    lines = [str(code)]
    if label:
        lines += textwrap.wrap(str(label), 64)
    place = [f"B{data['grade']}" if data.get("grade") is not None else "", str(data.get("strand_name") or "")]
    if any(place):
        lines.append(", ".join(p for p in place if p))
    description = data.get("description")
    if text_mode == "full" and description and description != label:
        lines += ["", *textwrap.wrap(str(description), 64)]
    return "\n".join(lines)


def _edge_tooltip(a: Hashable, b: Hashable, data: dict[str, Any]) -> str:
    kind = str(data.get("type"))
    lines = [
        f"{ids.code_of(str(a))} to {ids.code_of(str(b))}",
        f"{kind}: {EDGE_MEANING.get(kind, 'prerequisite link')}",
    ]
    if kind == "augmented":
        lines.append(f"probability {float(data.get('weight', 0)):.2f}")
    return "\n".join(lines)


def _controls(G: nx.DiGraph, grades: list[int], strands: dict[int, str], title: str) -> str:
    esc = html.escape
    grade_boxes = "".join(f'<label><input type="checkbox" data-kwan-grade="{g}" checked> B{g}</label>' for g in grades)
    strand_boxes = "".join(
        f'<label><input type="checkbox" data-kwan-strand="{s}" checked> '
        f'<span class="kwan-swatch" style="color:{STRAND_STYLE.get(s, OTHER_STYLE)[0]}">'
        f"{SHAPE_GLYPH[STRAND_STYLE.get(s, OTHER_STYLE)[1]]}</span> {esc(name or f'Strand {s}')}</label>"
        for s, name in strands.items()
    )
    kinds = [k for k in EDGE_STYLE if any(d.get("type") == k for *_, d in G.edges(data=True))]
    if "augmented" not in kinds:
        kinds.append("augmented")  # the legend always explains the dashed line

    def key(kind: str) -> str:
        style = EDGE_STYLE[kind]
        dash = ' stroke-dasharray="6 4"' if style["dashes"] else ""
        return (
            f'<span class="kwan-key" title="{esc(EDGE_MEANING[kind])}"><svg width="34" height="10" aria-hidden="true">'
            f'<line x1="1" y1="5" x2="33" y2="5" stroke="{style["color"]}" stroke-width="{style["width"]}"{dash}/>'
            f"</svg> {kind}</span>"
        )

    edge_keys = "".join(key(k) for k in kinds)
    mode = "full curriculum text" if G.graph.get("text_mode") == "full" else "short labels"
    return f"""
<style>
 body {{ margin: 0; background: {SURFACE}; color: #0b0b0b;
   font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
 #kwan-controls {{ padding: 10px 16px 6px; font-size: 14px; line-height: 1.5; }}
 #kwan-controls h1 {{ font-size: 18px; margin: 0 0 2px; font-weight: 600; }}
 #kwan-controls p {{ margin: 0 0 6px; color: {INK}; }}
 #kwan-controls .kwan-row {{ display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center; margin: 2px 0; }}
 #kwan-controls .kwan-name {{ color: {INK}; min-width: 4.5em; }}
 #kwan-controls label, #kwan-controls .kwan-key {{ white-space: nowrap; cursor: pointer; margin: 0; }}
 #kwan-controls .kwan-key {{ cursor: help; }}
 #kwan-controls .kwan-swatch {{ font-size: 15px; }}
 #kwan-controls button {{ font: inherit; padding: 1px 10px; border: 1px solid {MUTED}; border-radius: 4px;
   background: #fff; cursor: pointer; }}
 #mynetwork {{ border: 0 !important; border-top: 1px solid #e4e3df !important; background: {SURFACE}; }}
 div.vis-tooltip {{ white-space: pre-line !important; max-width: 480px; font-family: inherit !important;
   font-size: 13px !important; }}
</style>
<div id="kwan-controls">
 <h1>{esc(title)}</h1>
 <p>{G.number_of_nodes()} indicators, {G.number_of_edges()} prerequisite links, shown with {mode}.
 Grades run left to right. Scroll to zoom, drag to move, point at a node for its code and label.</p>
 <div class="kwan-row"><span class="kwan-name">Grades</span>{grade_boxes}
  <button type="button" id="kwan-all">Show all</button><button type="button" id="kwan-fit">Fit to window</button></div>
 <div class="kwan-row"><span class="kwan-name">Strands</span>{strand_boxes}</div>
 <div class="kwan-row"><span class="kwan-name">Links</span>{edge_keys}</div>
</div>
"""


_SCRIPT = """
<script>
(function () {
  var boxes = Array.prototype.slice.call(document.querySelectorAll("#kwan-controls input[type=checkbox]"));
  function apply() {
    var grade = {}, strand = {};
    boxes.forEach(function (b) {
      if (b.dataset.kwanGrade) { grade[b.dataset.kwanGrade] = b.checked; }
      if (b.dataset.kwanStrand) { strand[b.dataset.kwanStrand] = b.checked; }
    });
    nodes.update(nodes.get().map(function (n) {
      var show = grade[String(n.kwan_grade)] !== false;
      if (n.kwan_strand != null && strand[String(n.kwan_strand)] === false) { show = false; }
      return {id: n.id, hidden: !show};
    }));
  }
  boxes.forEach(function (b) { b.addEventListener("change", apply); });
  document.getElementById("kwan-all").addEventListener("click", function () {
    boxes.forEach(function (b) { b.checked = true; });
    apply();
    network.fit();
  });
  document.getElementById("kwan-fit").addEventListener("click", function () { network.fit(); });
})();
</script>
"""

_BOOTSTRAP = re.compile(r"<(link|script)\b[^>]*bootstrap[^>]*>(\s*</script>)?", re.S)
_EMPTY_HEADING = re.compile(r"<center>\s*<h1>\s*</h1>\s*</center>", re.S)


def graph_html(
    G: nx.DiGraph,
    grades: Collection[int] | None = None,
    height: str = "760px",
    inline: bool = False,
    title: str | None = None,
) -> str:
    """The interactive page of a graph, as one HTML string.

    `grades` keeps only those grades (the page also has grade tick boxes for the reader). With
    `inline=True` the drawing library is written into the page, so it works without internet access;
    otherwise the page loads it from a public CDN and stays small. Node hover text never shows the
    curriculum description unless `G.graph["text_mode"]` is `"full"`.
    """
    from pyvis.network import Network

    if grades is not None:
        keep = set(grades)
        G = G.subgraph([n for n, d in G.nodes(data=True) if d.get("grade") in keep])
    text_mode = G.graph.get("text_mode") or "labels"
    subject = G.graph.get("subject") or ids.DEFAULT_SUBJECT
    net = Network(
        height=height, width="100%", directed=True, bgcolor=SURFACE, cdn_resources="in_line" if inline else "remote"
    )
    pos = layout(G)
    strands: dict[int, str] = {}
    present_grades: set[int] = set()
    top = min((y for _, y in pos.values()), default=0)
    for n in sorted(G, key=schema.sort_key):
        d = G.nodes[n]
        strand = d.get("strand")
        colour, shape = STRAND_STYLE.get(strand, OTHER_STYLE) if isinstance(strand, int) else OTHER_STYLE
        if isinstance(strand, int):
            strands.setdefault(strand, str(d.get("strand_name") or ""))
        if d.get("grade") is not None:
            present_grades.add(int(d["grade"]))
        x, y = pos[n]
        net.add_node(
            str(n),
            label=str(d.get("code") or ids.code_of(str(n))),
            title=node_tooltip(n, d, text_mode),
            shape=shape,
            color={"background": colour, "border": INK, "highlight": {"background": colour, "border": "#0b0b0b"}},
            borderWidth=1,
            size=11,
            x=x,
            y=y,
            physics=False,
            font={"size": 12, "color": INK},
            kwan_grade=d.get("grade"),
            kwan_strand=strand,
        )
    column_x: dict[int, int] = {}
    for n, (x, _) in pos.items():
        grade = G.nodes[n].get("grade")
        if grade is not None:
            column_x[int(grade)] = min(column_x.get(int(grade), x), x)
    for grade, x in sorted(column_x.items()):
        net.add_node(
            f"heading:B{grade}",
            label=f"B{grade}",
            shape="text",
            x=x,
            y=top - 2 * DY,
            physics=False,
            fixed=True,
            font={"size": 44, "color": INK},
            kwan_grade=grade,
            kwan_strand=None,
        )
    for a, b, d in sorted(G.edges(data=True), key=lambda e: (schema.sort_key(e[0]), schema.sort_key(e[1]))):
        style = EDGE_STYLE.get(str(d.get("type")), EDGE_STYLE["sequence"])
        net.add_edge(
            str(a),
            str(b),
            title=_edge_tooltip(a, b, d),
            color=style["color"],
            width=style["width"],
            dashes=style["dashes"],
            kwan_type=d.get("type"),
        )
    net.set_options(
        json.dumps(
            {
                "physics": {"enabled": False},
                "interaction": {"hover": True, "tooltipDelay": 120, "dragNodes": False},
                "edges": {"smooth": False, "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}},
            }
        )
    )
    page = net.generate_html(notebook=False)
    page = _BOOTSTRAP.sub("", page)
    page = _EMPTY_HEADING.sub("", page)
    heading = title or f"Kwan curriculum graph: {subject}"
    page = page.replace("<head>", f"<head>\n<title>{html.escape(heading)}</title>", 1)
    page = page.replace(
        "<body>", "<body>" + _controls(G, sorted(present_grades), dict(sorted(strands.items())), heading), 1
    )
    return page.replace("</body>", _SCRIPT + "</body>", 1)


def write_html(G: nx.DiGraph, path: str | os.PathLike[str], **options: Any) -> Path:
    """Write the page of `graph_html(G, **options)` to `path` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_html(G, **options), encoding="utf-8")
    return path
