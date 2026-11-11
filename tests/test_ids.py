"""Namespaced ids: make, split, and the Resolver on v2 graphs, v1 graphs and plain records."""

import networkx as nx
import pytest

from kwan import ids


def v2_graph():
    G = nx.DiGraph(subject="math", schema_version=2, text_mode="labels")
    G.add_node("math:B4.1.1.1.2", subject="math", code="B4.1.1.1.2", aliases=["B4.1.1.1.2"])
    # the PDF printed this indicator with a wrong code; the printed code is kept as an alias
    G.add_node("math:B6.1.2.6.1", subject="math", code="B6.1.2.6.1", aliases=["B6.1.2.6.1", "B6.1.1.6.1"])
    G.add_node("math:B6.3.3.5.2", subject="math", code="B6.3.3.5.2", aliases=["B6.3.3.5.2", "math:B6.3.3.5.1_dup"])
    return G


# ---------------------------------------------------------------- make / split / code_of


def test_make_and_split_round_trip():
    assert ids.make("math", "B4.1.1.1.2") == "math:B4.1.1.1.2"
    assert ids.split("math:B4.1.1.1.2") == ("math", "B4.1.1.1.2")
    for subject, code in [("math", "B1.1.1.1.1"), ("english", "B6.4.10.1.3"), ("math", "B6.3.3.5.1_dup")]:
        assert ids.split(ids.make(subject, code)) == (subject, code)
        assert ids.code_of(ids.make(subject, code)) == code


def test_make_is_safe_to_call_twice_but_refuses_another_subject():
    assert ids.make("math", "math:B4.1.1.1.2") == "math:B4.1.1.1.2"
    with pytest.raises(ValueError):
        ids.make("english", "math:B4.1.1.1.2")


@pytest.mark.parametrize("subject, code", [("", "B1.1.1.1.1"), ("ma:th", "B1.1.1.1.1"), ("math", ""), (None, "B1")])
def test_make_rejects_bad_parts(subject, code):
    with pytest.raises(ValueError):
        ids.make(subject, code)


def test_split_of_a_bare_code_gives_no_subject_or_the_default():
    assert ids.split("B4.1.1.1.2") == (None, "B4.1.1.1.2")
    assert ids.split("B4.1.1.1.2", "math") == ("math", "B4.1.1.1.2")
    assert ids.split("english:B4.1.1.1.2", "math") == ("english", "B4.1.1.1.2")


@pytest.mark.parametrize("bad", ["", ":B4.1.1.1.2", "math:", None, 7])
def test_split_rejects_malformed_ids(bad):
    with pytest.raises(ValueError):
        ids.split(bad)


def test_code_of_and_subject_of():
    assert ids.code_of("math:B4.1.1.1.2") == "B4.1.1.1.2"
    assert ids.code_of("B4.1.1.1.2") == "B4.1.1.1.2"
    assert ids.subject_of("math:B4.1.1.1.2") == "math"
    assert ids.subject_of("B4.1.1.1.2") is None and ids.subject_of("B4.1.1.1.2", "math") == "math"


def test_is_namespaced():
    assert ids.is_namespaced("math:B4.1.1.1.2")
    assert not ids.is_namespaced("B4.1.1.1.2")
    assert not ids.is_namespaced(":B4.1.1.1.2") and not ids.is_namespaced("math:") and not ids.is_namespaced(None)


# ---------------------------------------------------------------- Resolver


def test_resolver_accepts_canonical_ids_and_bare_codes():
    r = ids.Resolver(v2_graph())
    assert r.resolve("math:B4.1.1.1.2") == "math:B4.1.1.1.2"
    assert r.resolve("B4.1.1.1.2") == "math:B4.1.1.1.2"


def test_resolver_accepts_aliases_bare_and_namespaced():
    r = ids.Resolver(v2_graph())
    assert r.resolve("B6.1.1.6.1") == "math:B6.1.2.6.1"  # the misprinted code
    assert r.resolve("math:B6.1.1.6.1") == "math:B6.1.2.6.1"
    assert r.resolve("B6.3.3.5.1_dup") == "math:B6.3.3.5.2"
    assert r.resolve("math:B6.3.3.5.1_dup") == "math:B6.3.3.5.2"


def test_every_resolved_id_resolves_to_itself():
    r = ids.Resolver(v2_graph())
    for spelling in ["B4.1.1.1.2", "math:B4.1.1.1.2", "B6.1.1.6.1", "math:B6.3.3.5.1_dup"]:
        canonical = r.resolve(spelling)
        assert ids.is_namespaced(canonical) and r.resolve(canonical) == canonical


@pytest.mark.parametrize("unknown", ["B9.9.9.9.9", "math:B9.9.9.9.9", "english:B4.1.1.1.2", "", "nonsense", None, 4])
def test_resolver_raises_keyerror_for_unknown_ids(unknown):
    r = ids.Resolver(v2_graph())
    with pytest.raises(KeyError) as err:
        r.resolve(unknown)
    assert err.value.args[0] == f"unknown indicator '{unknown}'"
    assert unknown not in r and r.get(unknown) is None and r.get(unknown, "fallback") == "fallback"


def test_resolver_forgives_spaces_and_letter_case():
    r = ids.Resolver(v2_graph())
    assert r.resolve(" B4.1.1.1.2 ") == "math:B4.1.1.1.2"
    assert r.resolve("b4.1.1.1.2") == "math:B4.1.1.1.2"
    assert r.resolve("MATH:b4.1.1.1.2") == "math:B4.1.1.1.2"


def test_a_real_node_wins_over_an_alias_with_the_same_code():
    G = v2_graph()
    G.add_node("math:B6.1.1.6.1", subject="math", code="B6.1.1.6.1", aliases=["B6.1.1.6.1"])
    r = ids.Resolver(G)
    assert r.resolve("B6.1.1.6.1") == "math:B6.1.1.6.1" and r.resolve("math:B6.1.1.6.1") == "math:B6.1.1.6.1"
    assert r.resolve("B6.1.2.6.1") == "math:B6.1.2.6.1"


def test_resolver_on_a_v1_graph_returns_the_bare_node_keys():
    G = nx.DiGraph()
    G.add_node("B4.1.1.1.2", grade=4)
    r = ids.Resolver(G)
    assert r.subject == "math"
    assert r.resolve("B4.1.1.1.2") == "B4.1.1.1.2" and r.resolve("math:B4.1.1.1.2") == "B4.1.1.1.2"
    with pytest.raises(KeyError):
        r.resolve("B4.1.1.1.3")


def test_resolver_takes_the_subject_from_the_graph():
    G = nx.DiGraph(subject="english")
    G.add_node("english:B4.1.1.1.2", subject="english", code="B4.1.1.1.2")
    r = ids.Resolver(G)
    assert r.resolve("B4.1.1.1.2") == "english:B4.1.1.1.2" and "math:B4.1.1.1.2" not in r


def test_resolver_accepts_records_and_mappings():
    records = [{"id": "math:B1.1.1.1.1", "aliases": ["B1.1.1.1.1", "B1.1.1.1.9"]}]
    assert ids.Resolver(records).resolve("B1.1.1.1.9") == "math:B1.1.1.1.1"
    assert (
        ids.Resolver({"math:B1.1.1.1.1": {"aliases": ["B1.1.1.1.9"]}}).resolve("math:B1.1.1.1.9") == "math:B1.1.1.1.1"
    )
    assert len(ids.Resolver(records)) == 1 and "B1.1.1.1.1" in ids.Resolver(records)


def test_ids_module_needs_only_the_standard_library():
    import subprocess
    import sys

    code = "import sys, kwan.ids; assert 'networkx' not in sys.modules, 'kwan.ids must not import networkx'"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode == 0
