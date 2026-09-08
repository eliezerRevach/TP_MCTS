"""Render the unweighted action-fact interaction graph for Stuck_Car_1o."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"


actions = {
    "rest",
    "search",
    "place_bad",
    "place_good",
    "push_gas",
    "push_car",
    "push_car_gas",
}

facts = {
    "tired",
    "free_hands",
    "free_legs",
    "got_bad",
    "got_good",
    "rock_bad",
    "rock_good",
    "car_out",
}

# One undirected edge relation: an action is adjacent to every fact that it
# reads (condition/probability context) or changes (start/end effect).
edges = {
    "rest": {"tired", "free_hands", "free_legs"},
    "search": {"tired", "free_hands", "got_bad", "got_good"},
    "place_bad": {"tired", "free_hands", "free_legs", "got_bad", "rock_bad"},
    "place_good": {"tired", "free_hands", "free_legs", "got_good", "rock_good"},
    "push_gas": {"tired", "free_legs", "rock_bad", "rock_good", "car_out"},
    "push_car": {"tired", "free_hands", "rock_bad", "rock_good", "car_out"},
    "push_car_gas": {
        "tired",
        "free_hands",
        "free_legs",
        "rock_bad",
        "rock_good",
        "car_out",
    },
}

# Directed view of the same interaction data.  Direction is the only semantic
# distinction in the graph: facts point to actions that read them, and actions
# point to facts that they may change.
reads = {
    "rest": {"free_hands", "free_legs"},
    "search": {"tired", "free_hands"},
    "place_bad": {"tired", "free_hands", "free_legs", "got_bad"},
    "place_good": {"tired", "free_hands", "free_legs", "got_good"},
    "push_gas": {"tired", "free_legs", "rock_bad", "rock_good", "car_out"},
    "push_car": {"tired", "free_hands", "rock_bad", "rock_good", "car_out"},
    "push_car_gas": {
        "tired", "free_hands", "free_legs", "rock_bad", "rock_good", "car_out"
    },
}

writes = {
    "rest": {"tired"},
    "search": {"free_hands", "got_bad", "got_good"},
    "place_bad": {"tired", "free_hands", "free_legs", "got_bad", "rock_bad"},
    "place_good": {"tired", "free_hands", "free_legs", "got_good", "rock_good"},
    "push_gas": {"free_legs", "car_out"},
    "push_car": {"tired", "free_hands", "car_out"},
    "push_car_gas": {"tired", "free_hands", "free_legs", "car_out"},
}

labels = {
    "rest": "rest",
    "search": "search",
    "place_bad": "place rock\n(bad)",
    "place_good": "place rock\n(good)",
    "push_gas": "push gas",
    "push_car": "push car",
    "push_car_gas": "push car\n+ gas",
    "tired": "tired",
    "free_hands": "free\n(hands)",
    "free_legs": "free\n(legs)",
    "got_bad": "got rock\n(bad)",
    "got_good": "got rock\n(good)",
    "rock_bad": "rock under car\n(bad)",
    "rock_good": "rock under car\n(good)",
    "car_out": "car out\nGOAL",
}

graph = nx.Graph()
graph.add_nodes_from(actions, kind="action")
graph.add_nodes_from(facts, kind="fact")
for action, mentioned_facts in edges.items():
    graph.add_edges_from((action, fact) for fact in mentioned_facts)

# Deterministic force-directed layout, matching the interaction-graph idea.
pos = nx.spring_layout(graph, seed=31, k=1.35, iterations=1200)

fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
ax.set_facecolor("white")

nx.draw_networkx_edges(
    graph,
    pos,
    ax=ax,
    edge_color="#64748b",
    width=1.7,
    alpha=0.42,
)

nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=sorted(actions),
    node_shape="s",
    node_size=3500,
    node_color="#dbeafe",
    edgecolors="#2563ad",
    linewidths=2.5,
    ax=ax,
)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=sorted(facts - {"car_out"}),
    node_shape="o",
    node_size=3100,
    node_color="#dcfce7",
    edgecolors="#15803d",
    linewidths=2.5,
    ax=ax,
)
nx.draw_networkx_nodes(
    graph,
    pos,
    nodelist=["car_out"],
    node_shape="o",
    node_size=3500,
    node_color="#fef3c7",
    edgecolors="#d97706",
    linewidths=3.2,
    ax=ax,
)

nx.draw_networkx_labels(
    graph,
    pos,
    labels=labels,
    font_family="DejaVu Sans",
    font_size=9.5,
    font_weight="bold",
    font_color="#111827",
    ax=ax,
)

fig.suptitle(
    "Stuck_Car_1o — Action–Fact Interaction Graph",
    fontsize=25,
    fontweight="bold",
    color="#111827",
    y=0.975,
)
ax.set_title(
    r"$G_I=(A\cup F,E)$    ·    $\{a,f\}\in E$ iff action $a$ reads or changes fact $f$",
    fontsize=15,
    color="#334155",
    pad=18,
)

legend = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#dbeafe",
           markeredgecolor="#2563ad", markeredgewidth=2, markersize=17,
           label="action vertex"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#dcfce7",
           markeredgecolor="#15803d", markeredgewidth=2, markersize=17,
           label="fact vertex"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#fef3c7",
           markeredgecolor="#d97706", markeredgewidth=2, markersize=17,
           label="goal fact"),
]
ax.legend(handles=legend, loc="lower left", frameon=True, framealpha=0.96,
          facecolor="white", edgecolor="#cbd5e1", fontsize=12)

ax.text(
    0.995,
    0.015,
    "Unweighted · undirected · all edges have the same meaning",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    color="#475569",
)
ax.margins(0.16)
ax.axis("off")
fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))

png = OUT / "stuck_car_interaction_graph.png"
svg = OUT / "stuck_car_interaction_graph.svg"
fig.savefig(png, dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(svg, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(png)
print(svg)


# ---------------------------------------------------------------- directed view

digraph = nx.DiGraph()
digraph.add_nodes_from(actions, kind="action")
digraph.add_nodes_from(facts, kind="fact")
for action, read_facts in reads.items():
    digraph.add_edges_from((fact, action) for fact in read_facts)
for action, written_facts in writes.items():
    digraph.add_edges_from((action, fact) for fact in written_facts)

directed_edges = set(digraph.edges())
reciprocal = [(u, v) for u, v in directed_edges if (v, u) in directed_edges]
single = [(u, v) for u, v in directed_edges if (v, u) not in directed_edges]

fig, ax = plt.subplots(figsize=(16, 10), facecolor="white")
ax.set_facecolor("white")

nx.draw_networkx_edges(
    digraph,
    pos,
    edgelist=single,
    ax=ax,
    edge_color="#475569",
    width=1.7,
    alpha=0.56,
    arrows=True,
    arrowsize=17,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.0",
    min_source_margin=25,
    min_target_margin=25,
)
nx.draw_networkx_edges(
    digraph,
    pos,
    edgelist=reciprocal,
    ax=ax,
    edge_color="#475569",
    width=1.7,
    alpha=0.56,
    arrows=True,
    arrowsize=17,
    arrowstyle="-|>",
    connectionstyle="arc3,rad=0.13",
    min_source_margin=25,
    min_target_margin=25,
)

nx.draw_networkx_nodes(
    digraph,
    pos,
    nodelist=sorted(actions),
    node_shape="s",
    node_size=3500,
    node_color="#dbeafe",
    edgecolors="#2563ad",
    linewidths=2.5,
    ax=ax,
)
nx.draw_networkx_nodes(
    digraph,
    pos,
    nodelist=sorted(facts - {"car_out"}),
    node_shape="o",
    node_size=3100,
    node_color="#dcfce7",
    edgecolors="#15803d",
    linewidths=2.5,
    ax=ax,
)
nx.draw_networkx_nodes(
    digraph,
    pos,
    nodelist=["car_out"],
    node_shape="o",
    node_size=3500,
    node_color="#fef3c7",
    edgecolors="#d97706",
    linewidths=3.2,
    ax=ax,
)
nx.draw_networkx_labels(
    digraph,
    pos,
    labels=labels,
    font_family="DejaVu Sans",
    font_size=9.5,
    font_weight="bold",
    font_color="#111827",
    ax=ax,
)

fig.suptitle(
    "Stuck_Car_1o — Directed Action–Fact Interaction Graph",
    fontsize=25,
    fontweight="bold",
    color="#111827",
    y=0.975,
)
ax.set_title(
    r"$f\rightarrow a$: action reads fact    ·    $a\rightarrow f$: action may change fact",
    fontsize=15,
    color="#334155",
    pad=18,
)
ax.legend(handles=legend, loc="lower left", frameon=True, framealpha=0.96,
          facecolor="white", edgecolor="#cbd5e1", fontsize=12)
ax.text(
    0.995,
    0.015,
    "Unweighted · direction represents causal influence",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    color="#475569",
)
ax.margins(0.16)
ax.axis("off")
fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.94))

directed_png = OUT / "stuck_car_directed_interaction_graph.png"
directed_svg = OUT / "stuck_car_directed_interaction_graph.svg"
fig.savefig(directed_png, dpi=180, bbox_inches="tight", facecolor="white")
fig.savefig(directed_svg, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(directed_png)
print(directed_svg)
