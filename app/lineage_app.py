"""dbt Column Lineage Explorer — Streamlit app.

Run with:
    streamlit run app/lineage_app.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# Allow running from repo root or from app/ directory
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dbt_column_lineage.api import ColumnLineageResult, get_column_lineage

# Suppress noisy library logs in the UI
logging.getLogger("dbt_column_lineage").setLevel(logging.ERROR)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="dbt Column Lineage Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_GRAPH_NODES = 120  # cap before warning
FLAG_ICONS = {
    "origin":   "🌱",
    "rename":   "🔄",
    "computed": "🧮",
    "direct":   "→",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _mtime(path: str) -> float:
    """Return file mtime for cache invalidation."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _node_color(name: str, resource_type: str) -> str:
    if resource_type == "source":
        return "#2ecc71"
    if resource_type == "seed":
        return "#f1c40f"
    n = name.lower()
    if n.startswith("stg_"):
        return "#3498db"
    if n.startswith("base_"):
        return "#85c1e9"
    if n.startswith("int_"):
        return "#e67e22"
    if n.startswith(("mart_", "fct_", "dim_", "rpt_")):
        return "#9b59b6"
    return "#e74c3c"


def _flag_label(r: ColumnLineageResult) -> str:
    parts = []
    if r.is_first_in_chain:
        parts.append("🌱 origin")
    if r.is_rename:
        parts.append("🔄 rename")
    if r.is_computed:
        parts.append("🧮 computed")
    if not parts:
        parts.append("→ direct")
    return "  ".join(parts)


# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="⏳ Loading column lineage…")
def load_lineage(
    manifest: str,
    catalog: str,
    sql_source: str,
    _mtime_manifest: float,
    _mtime_catalog: float,
) -> List[ColumnLineageResult]:
    return get_column_lineage(
        manifest_path=manifest,
        catalog_path=catalog,
        compiled_sql_source=sql_source,  # type: ignore[arg-type]
    )


@st.cache_data(show_spinner=False)
def load_manifest_json(manifest: str, _mtime: float) -> dict:
    with open(manifest, encoding="utf-8") as f:
        return json.load(f)


# ── Index building ────────────────────────────────────────────────────────────

def build_indices(
    results: List[ColumnLineageResult],
) -> Tuple[Dict[str, List[ColumnLineageResult]], Dict[Tuple[str, str], ColumnLineageResult]]:
    by_model: Dict[str, List[ColumnLineageResult]] = {}
    fwd: Dict[Tuple[str, str], ColumnLineageResult] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)
        key = (r.model, r.column)
        if key not in fwd:          # first entry wins (highest precedence)
            fwd[key] = r
    return by_model, fwd


def build_model_graph(
    manifest_data: dict,
    results: List[ColumnLineageResult],
) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """Build a DiGraph; nodes are model names, edges from depends_on."""
    G = nx.DiGraph()

    # resource_type lookup: unique_id -> (name, resource_type)
    rt_map: Dict[str, str] = {}
    for uid, node in manifest_data.get("nodes", {}).items():
        name = node.get("name", "").lower()
        rt_map[name] = node.get("resource_type", "model")
    for uid, node in manifest_data.get("sources", {}).items():
        name = node.get("name", "").lower()
        rt_map[name] = "source"

    # Seed resource type
    for uid, node in manifest_data.get("nodes", {}).items():
        if node.get("resource_type") == "seed":
            name = node.get("name", "").lower()
            rt_map[name] = "seed"

    # Add all result models + their progenitors (sources/seeds)
    all_names: set[str] = set()
    for r in results:
        all_names.add(r.model)
        if r.progenitor_model:
            all_names.add(r.progenitor_model)

    for name in all_names:
        G.add_node(name, resource_type=rt_map.get(name, "model"))

    # Edges from lineage: progenitor -> model
    seen_edges: set[Tuple[str, str]] = set()
    for r in results:
        if r.progenitor_model and r.progenitor_model != r.model:
            edge = (r.progenitor_model, r.model)
            if edge not in seen_edges:
                G.add_edge(*edge)
                seen_edges.add(edge)

    return G, rt_map


# ── PyVis graph renderer ──────────────────────────────────────────────────────

def render_neighborhood(
    G: nx.DiGraph,
    selected: str,
    hops: int,
) -> Tuple[str, bool]:
    """Return (html, was_truncated)."""
    # Collect neighborhood
    neighborhood: set[str] = {selected}
    frontier = {selected}
    for _ in range(hops):
        new: set[str] = set()
        for n in frontier:
            new.update(G.predecessors(n))
            new.update(G.successors(n))
        new -= neighborhood
        neighborhood |= new
        frontier = new

    truncated = len(neighborhood) > MAX_GRAPH_NODES
    if truncated:
        # Keep selected + immediate neighbors first, then fill up
        keep: set[str] = {selected}
        keep.update(G.predecessors(selected))
        keep.update(G.successors(selected))
        for n in list(neighborhood):
            if len(keep) >= MAX_GRAPH_NODES:
                break
            keep.add(n)
        neighborhood = keep

    sub = G.subgraph(neighborhood)

    net = Network(
        height="520px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="white",
    )
    net.barnes_hut(
        gravity=-4000,
        central_gravity=0.4,
        spring_length=130,
        spring_strength=0.04,
        damping=0.9,
    )

    for node in sub.nodes():
        rt = G.nodes[node].get("resource_type", "model")
        color = _node_color(node, rt)
        is_selected = node == selected
        net.add_node(
            node,
            label=node,
            color={
                "background": color,
                "border": "#ffffff" if is_selected else color,
                "highlight": {"background": color, "border": "#ffffff"},
            },
            size=22 if is_selected else 14,
            font={"size": 11, "color": "white"},
            borderWidth=3 if is_selected else 1,
            title=f"<b>{node}</b><br/>{rt}",
        )

    for u, v in sub.edges():
        highlight = u == selected or v == selected
        net.add_edge(
            u, v,
            color={"color": "#aaaaaa" if not highlight else "#ffffff", "opacity": 0.7},
            arrows="to",
            width=2 if highlight else 1,
        )

    html_path = tempfile.mktemp(suffix=".html")
    net.save_graph(html_path)
    html = Path(html_path).read_text(encoding="utf-8")
    return html, truncated


# ── Column chain tracer ───────────────────────────────────────────────────────

def trace_chain(
    model: str,
    column: str,
    fwd: Dict[Tuple[str, str], ColumnLineageResult],
    max_depth: int = 30,
) -> List[ColumnLineageResult]:
    chain: List[ColumnLineageResult] = []
    current: Optional[Tuple[str, str]] = (model, column)
    visited: set[Tuple[str, str]] = set()

    while current and current in fwd and current not in visited and len(chain) < max_depth:
        visited.add(current)
        r = fwd[current]
        chain.append(r)
        if r.progenitor_model and r.progenitor_column:
            current = (r.progenitor_model, r.progenitor_column)
        else:
            current = None

    return chain


def render_chain(chain: List[ColumnLineageResult]) -> None:
    if not chain:
        st.info("No lineage data for this column.")
        return

    for i, r in enumerate(chain):
        icon = FLAG_ICONS["origin"] if r.is_first_in_chain else (
            FLAG_ICONS["rename"] if r.is_rename else (
                FLAG_ICONS["computed"] if r.is_computed else FLAG_ICONS["direct"]
            )
        )
        col_a, col_b = st.columns([3, 4])
        with col_a:
            st.markdown(f"**{i + 1}.** {icon} `{r.model}` **·** `{r.column}`")
        with col_b:
            parts = []
            if r.is_rename and r.source_column:
                parts.append(f"renamed from `{r.source_column}`")
            if r.is_computed:
                parts.append("computed expression")
            if r.progenitor_model:
                parts.append(f"← `{r.progenitor_model}.{r.progenitor_column}`")
            if parts:
                st.caption("  |  ".join(parts))

        # Show terminal source/seed node after the last chain entry
        if r.is_first_in_chain and r.progenitor_model:
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;&nbsp;↳ 🗄️ **source:** `{r.progenitor_model}`"
                + (f" · `{r.progenitor_column}`" if r.progenitor_column else ""),
                unsafe_allow_html=True,
            )
            break

    if len(chain) >= 30:
        st.warning("Chain truncated at 30 hops.")


# ── Sidebar: legend ───────────────────────────────────────────────────────────

def render_legend() -> None:
    st.markdown(
        """
**Node colors**
🟢 source &nbsp; 🟡 seed &nbsp; 🔵 staging  
🩵 base &nbsp; 🟠 intermediate &nbsp; 🟣 mart/fct/dim  
🔴 other

**Column flags**
🌱 first in chain &nbsp; 🔄 rename &nbsp; 🧮 computed &nbsp; → direct
        """
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sidebar ──
    with st.sidebar:
        st.title("🔍 Column Lineage")
        st.subheader("Configuration")

        manifest_path = st.text_input(
            "manifest.json",
            value=st.session_state.get("manifest_path", ""),
            placeholder="/path/to/target/manifest.json",
        )
        catalog_path = st.text_input(
            "catalog.json",
            value=st.session_state.get("catalog_path", ""),
            placeholder="/path/to/target/catalog.json",
        )
        compiled_sql_source = st.selectbox(
            "Compiled SQL source",
            ["target_dir", "manifest", "auto_compile"],
            index=0,
            help=(
                "target_dir: read compiled SQL from disk (fastest)\n"
                "manifest: require inline compiled_code (dbt compile)\n"
                "auto_compile: run dbt compile first (slowest, most accurate)"
            ),
        )
        load_btn = st.button("🚀 Load Lineage", use_container_width=True)

        if load_btn:
            if not manifest_path or not catalog_path:
                st.error("Provide both manifest.json and catalog.json paths.")
                st.stop()
            st.session_state["manifest_path"] = manifest_path
            st.session_state["catalog_path"] = catalog_path
            st.session_state["sql_source"] = compiled_sql_source
            st.cache_data.clear()

    # ── Guard: not yet loaded ──
    if "manifest_path" not in st.session_state:
        _render_welcome()
        return

    mp: str = st.session_state["manifest_path"]
    cp: str = st.session_state["catalog_path"]
    ss: str = st.session_state.get("sql_source", "target_dir")

    # ── Load data ──
    results = load_lineage(mp, cp, ss, _mtime(mp), _mtime(cp))
    manifest_data = load_manifest_json(mp, _mtime(mp))
    by_model, fwd = build_indices(results)
    G, rt_map = build_model_graph(manifest_data, results)

    # ── Sidebar: model selector + stats ──
    with st.sidebar:
        st.divider()
        st.subheader("Model Explorer")

        c1, c2 = st.columns(2)
        c1.metric("Models", len(by_model))
        c2.metric("Columns", len(results))

        model_names = sorted(by_model.keys())
        selected_model = st.selectbox("Select model", model_names, key="selected_model")

        model_cols = by_model.get(selected_model, [])
        unique_cols = {r.column: r for r in model_cols}

        n_origin   = sum(1 for r in unique_cols.values() if r.is_first_in_chain)
        n_rename   = sum(1 for r in unique_cols.values() if r.is_rename)
        n_computed = sum(1 for r in unique_cols.values() if r.is_computed)
        n_direct   = sum(
            1 for r in unique_cols.values()
            if not r.is_first_in_chain and not r.is_rename and not r.is_computed
        )

        st.markdown(
            f"**{len(unique_cols)} columns**  \n"
            f"🌱 {n_origin} &nbsp; 🔄 {n_rename} &nbsp; 🧮 {n_computed} &nbsp; → {n_direct}"
        )
        st.divider()
        render_legend()

    # ── Main tabs ──
    tab_graph, tab_cols, tab_chain = st.tabs(
        ["📊 Model Graph", "📋 Columns", "🔗 Column Chain"]
    )

    with tab_graph:
        hops = st.slider("Neighborhood hops", min_value=1, max_value=4, value=2)
        html, truncated = render_neighborhood(G, selected_model, hops)
        if truncated:
            st.warning(
                f"Graph truncated to {MAX_GRAPH_NODES} nodes. "
                "Reduce hops or select a less-connected model to see full neighborhood."
            )
        components.html(html, height=540, scrolling=False)

    with tab_cols:
        if not model_cols:
            st.info("No column data for this model.")
        else:
            flag_filter = st.multiselect(
                "Filter by flag",
                ["🌱 origin", "🔄 rename", "🧮 computed", "→ direct"],
                default=[],
                key="flag_filter",
            )
            rows = []
            for r in sorted(unique_cols.values(), key=lambda x: x.column):
                flag = _flag_label(r)
                if flag_filter and not any(f in flag for f in flag_filter):
                    continue
                rows.append({
                    "Column":           r.column,
                    "Flag":             flag,
                    "Progenitor model": r.progenitor_model or "—",
                    "Progenitor col":   r.progenitor_column or "—",
                    "Source col":       r.source_column or "—",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)
            else:
                st.info("No columns match the selected flags.")

    with tab_chain:
        col_names = sorted(unique_cols.keys())
        if not col_names:
            st.info("No column data for this model.")
        else:
            selected_col = st.selectbox(
                "Select column to trace",
                col_names,
                key="selected_col",
            )
            if selected_col:
                r0 = unique_cols.get(selected_col)
                if r0 and r0.is_computed:
                    st.info(
                        "🧮 This column is a **computed expression** — "
                        "it may depend on multiple upstream columns. "
                        "Showing one upstream path."
                    )
                chain = trace_chain(selected_model, selected_col, fwd)
                if chain:
                    st.markdown(
                        f"#### Lineage chain: `{selected_model}` · `{selected_col}`"
                    )
                    render_chain(chain)
                else:
                    st.warning("No lineage found — column may not appear in compiled SQL.")


def _render_welcome() -> None:
    st.title("🔍 dbt Column Lineage Explorer")
    st.info("👈 Configure your dbt project paths in the sidebar and click **Load Lineage**.")
    st.markdown("""
---
### What this app does

Explore column-level lineage across your entire dbt project.

| Tab | What you see |
|-----|-------------|
| 📊 **Model Graph** | Interactive dependency graph — select a model, adjust hop radius |
| 📋 **Columns** | All columns for the selected model with lineage flags |
| 🔗 **Column Chain** | Trace any column back step-by-step to its source or seed origin |

---
### Column flags

| Flag | Meaning |
|------|---------|
| 🌱 **origin** | First in chain — value comes directly from a source or seed table |
| 🔄 **rename** | Aliased from an upstream column (pure pass-through, different name) |
| 🧮 **computed** | Derived expression — CASE, arithmetic, function call, aggregation |
| **→ direct** | Straight pass-through between dbt models, same name |

---
### Quick start

```
manifest.json  →  <dbt_project>/target/manifest.json
catalog.json   →  <dbt_project>/target/catalog.json
SQL source     →  target_dir  (fastest; reads compiled SQL from disk)
```
    """)


if __name__ == "__main__":
    main()
