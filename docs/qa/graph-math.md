# Graph validation: math

This page is written by `kwan graph validate`. Do not edit it by hand: rebuild the graph with
`kwan graph build` and run `kwan graph validate` again. It lists codes and numbers only.

Graph file: `data/graph/math.graph.json` (schema version 2, labels mode).

## Result

**All checks pass.**

| Check | Result | Detail |
|---|---|---|
| The graph has no cycle (it is a DAG) | pass | no cycle found |
| No edge points to a node that is not listed | pass | orphan edges: 0 |
| Every node can be reached from a root | pass | nodes not reached: 0 |

## Size

| Measure | Value |
|---|---:|
| Nodes (indicators) | 276 |
| Edges (prerequisite links) | 295 |
| Roots (nothing comes before) | 16 |
| Leaves (nothing comes after) | 20 |
| Isolated nodes (no edge at all) | 0 |
| Connected parts | 16 |
| Edges that point to an earlier grade | 0 |

Nodes by strand:

| Strand | Nodes |
|---|---:|
| 1: Number | 149 |
| 2: Algebra | 38 |
| 3: Geometry and Measurement | 61 |
| 4: Data | 28 |

## Longest path

The longest chain of prerequisites has **36 edges**. Counted by weight (a bridge edge counts 0.5 and an augmented edge counts its probability), the heaviest chain weighs **35.0**.

Longest chain by number of edges: `B1.1.1.1.1` -> `B1.1.1.1.2` -> `B1.1.1.1.3` -> `B1.1.1.1.4` -> `B1.1.1.1.5` -> `B1.1.1.1.6` -> `B2.1.1.1.1` -> `B2.1.1.1.2` -> `B2.1.1.1.3` -> `B2.1.1.1.4` -> `B2.1.1.1.5` -> `B2.1.1.1.6` -> `B3.1.1.1.1` -> `B3.1.1.1.2` -> `B3.1.1.1.3` -> `B3.1.1.1.4` -> `B4.1.1.1.1` -> `B4.1.1.1.2` -> `B4.1.1.1.3` -> `B4.1.1.1.4` -> `B4.1.1.1.5` -> `B4.1.1.1.6` -> `B5.1.1.1.1` -> `B5.1.1.1.2` -> `B5.1.1.1.3` -> `B5.1.1.1.4` -> `B5.1.1.1.5` -> `B5.1.1.1.6` -> `B6.1.1.1.1` -> `B6.1.1.1.2` -> `B6.1.1.1.3` -> `B6.1.1.1.4` -> `B6.1.1.1.5` -> `B6.1.1.1.6` -> `B6.1.1.2.1` -> `B6.1.1.2.2` -> `B6.1.1.3.1`

## Degree by grade

In-degree is the number of edges that arrive at a node (its direct prerequisites). Out-degree is the number of edges that leave it.

| Grade | Nodes | Roots | Leaves | Edges in | Mean in | Max in | Edges out | Mean out | Max out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 24 | 9 | 0 | 15 | 0.62 | 1 | 27 | 1.12 | 2 |
| B2 | 29 | 0 | 0 | 32 | 1.10 | 2 | 35 | 1.21 | 2 |
| B3 | 40 | 0 | 0 | 46 | 1.15 | 2 | 50 | 1.25 | 2 |
| B4 | 71 | 2 | 4 | 79 | 1.11 | 2 | 79 | 1.11 | 2 |
| B5 | 67 | 3 | 5 | 72 | 1.07 | 2 | 70 | 1.04 | 2 |
| B6 | 45 | 2 | 11 | 51 | 1.13 | 2 | 34 | 0.76 | 1 |

Where the edges of each grade go:

| Grade | Edges inside the grade | Edges to a later grade |
|---|---:|---:|
| B1 | 15 | 12 |
| B2 | 20 | 15 |
| B3 | 31 | 19 |
| B4 | 60 | 19 |
| B5 | 53 | 17 |
| B6 | 34 | 0 |

## Edges by type

| Type | Edges |
|---|---:|
| bridge | 54 |
| sequence | 159 |
| vertical | 82 |

## Edges by provenance

Provenance says what made an edge: a rule (`rule:<name>`) or the decision model, whose edges enter the graph only after a reviewer accepts them.

| Made by | Edges |
|---|---:|
| `rule:bridge` | 54 |
| `rule:sequence` | 159 |
| `rule:vertical` | 82 |

Edges without a provenance record: 0.

## Isolated nodes

An isolated node has no edge. The rules give it none when its content standard has one indicator, appears in one grade only and is the only content standard of its sub-strand in that grade.

Isolated nodes (0): none

## Roots

Roots (16): `B1.1.1.1.1`, `B1.1.2.1.1`, `B1.1.3.1.1`, `B1.1.4.1.1`, `B1.2.1.1.1`, `B1.3.1.1.1`, `B1.3.2.1.1`, `B1.3.3.1.1`, `B1.4.1.1.1`, `B4.1.5.1.1`, `B4.2.2.1.1`, `B5.2.3.1.1`, `B5.3.4.1.1`, `B5.4.2.1.1`, `B6.3.3.5.1`, `B6.4.2.2.1`
