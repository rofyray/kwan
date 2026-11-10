"""Build the Kwan curriculum knowledge graph (DAG) from parsed NaCCA indicators."""
import json, networkx as nx
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PATH = DATA_DIR / "graph.json"          # works from any working directory

def build_graph(indicators):
    G = nx.DiGraph()
    for r in indicators:
        G.add_node(r["id"], **r)
    by_cs = defaultdict(list)                      # (grade, cs_key) -> ordered indicator ids
    for r in indicators:
        by_cs[(r["grade"], r["cs_key"])].append(r["id"])
    # 1) sequence edges inside a content standard: indicator i -> i+1
    for ids in by_cs.values():
        for a, b in zip(ids, ids[1:]):
            G.add_edge(a, b, type="sequence", weight=1.0)
    # 2) vertical edges: last indicator of cs in grade g -> first indicator of same cs in the next grade that has it
    for (g, key), ids in by_cs.items():
        for g2 in range(g + 1, 7):
            if (g2, key) in by_cs:
                G.add_edge(ids[-1], by_cs[(g2, key)][0], type="vertical", weight=1.0)
                break
    # 3) sub-strand bridge: same grade, same sub-strand, consecutive content standards (weaker)
    sub = defaultdict(list)
    for (g, key), ids in sorted(by_cs.items(), key=lambda kv: (kv[0][0], [int(x) for x in kv[0][1].split(".")])):
        s, ss, _ = key.split(".")
        sub[(g, s, ss)].append(ids)
    for groups in sub.values():
        for a, b in zip(groups, groups[1:]):
            if not G.has_edge(a[-1], b[0]):
                G.add_edge(a[-1], b[0], type="bridge", weight=0.5)
    assert nx.is_directed_acyclic_graph(G), "curriculum graph must be a DAG"
    return G

def save(G, path=DEFAULT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G, edges="edges"), f, ensure_ascii=False, indent=1)

def load(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        return nx.node_link_graph(json.load(f), edges="edges")

if __name__ == "__main__":
    import sys
    with open(sys.argv[1], encoding="utf-8") as f:
        G = build_graph(json.load(f))
    save(G, sys.argv[2])
    from collections import Counter
    print(G.number_of_nodes(), "nodes", G.number_of_edges(), "edges", Counter(d["type"] for *_, d in G.edges(data=True)),
          "roots", sum(1 for n in G if G.in_degree(n) == 0), "longest path", nx.dag_longest_path_length(G))
