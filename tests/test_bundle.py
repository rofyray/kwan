"""The bundle (PRD 9.4): the shipped file, the builder, its licence checks and `kwan bundle`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import pytest

from kwan import _bundle_cli, bundle, cli
from kwan.data.registry import non_commercial
from kwan.graph import schema
from kwan.learner.item_bank import ItemBank
from kwan.learner.params import EloParams

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "src" / "kwan" / "bundles" / "kwan-bundle.math.json"
FULL_GRAPH = ROOT / "build" / "graph" / "math.graph.full.json"
EM_DASH = chr(0x2014)
FIXED_TIME = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def shipped() -> bundle.Bundle:
    return bundle.load_bundle()


# ----------------------------------------------------------------------------- the shipped bundle


def test_the_shipped_bundle_is_the_default_and_loads(shipped):
    assert bundle.default_bundle_path() == SHIPPED and SHIPPED.is_file()
    assert shipped.path == SHIPPED
    assert shipped.schema_version == 2 and shipped.subject == "math" and shipped.text_mode == "labels"
    assert shipped.bundle_version == f"math-{bundle.__version__}" == bundle.bundle_version_for()
    assert len(shipped.source_commit) == 40 or shipped.source_commit == "unknown"
    datetime.fromisoformat(shipped.created_at)


def test_the_shipped_bundle_has_every_field_of_the_contract(shipped):
    raw = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert list(raw) == [
        "schema_version",
        "bundle_version",
        "subject",
        "text_mode",
        "created_at",
        "source_commit",
        "graph",
        "params",
        "item_bank",
        "contexts",
        "context_bank",
        "i18n",
        "provenance",
    ]
    assert set(raw["graph"]) == {"nodes", "edges"}
    assert raw["params"] == EloParams().to_dict() or EloParams.from_dict(raw["params"])
    assert raw["item_bank"] is None or isinstance(raw["item_bank"], dict)
    assert "en" in raw["i18n"] and raw["i18n"]["en"]["builds_on"]
    assert set(raw["provenance"]) >= {"params", "item_bank", "graph", "contexts", "kwan_version"}
    assert isinstance(raw["provenance"]["params"]["datasets"], list)
    assert shipped.to_dict() == raw


def test_the_shipped_bundle_stays_small(shipped):
    size = SHIPPED.stat().st_size
    assert size < bundle.MAX_BYTES, f"{size} bytes is over the PRD 7 limit"
    assert size == len(bundle.dumps_bundle(shipped.to_dict()).encode("utf-8"))


def test_the_shipped_bundle_matches_the_committed_graph(shipped):
    G = shipped.load_graph()
    committed = schema.load_graph()
    assert set(G) == set(committed) and set(G.edges) == set(committed.edges)
    assert G.graph["text_mode"] == "labels" and G.number_of_nodes() == shipped.node_count
    assert nx.is_directed_acyclic_graph(G)
    for n, d in G.nodes(data=True):
        assert n.startswith("math:") and d["description"] == d["short_label"]
        assert not any(k in d for k in ("content_standard_text", "exemplars", "page"))


def test_the_shipped_bundle_holds_only_kwan_written_text(shipped):
    for entries in shipped.contexts.values():
        for e in entries:
            assert set(e) == {"text", "lang", "reviewed"} and e["text"] and EM_DASH not in e["text"]
    for table in shipped.i18n.values():
        assert all(EM_DASH not in t for t in table.values())
    assert shipped.context_bank["1:lower"] and all(":" in key for key in shipped.context_bank)


def test_drafts_in_the_bundle_are_marked_and_unused_by_default(shipped):
    from kwan.context.settings import ContextStore

    store = ContextStore.from_bundle(shipped.to_dict())
    drafts = shipped.provenance["contexts"]["drafts"]
    reviewed = shipped.provenance["contexts"]["reviewed"]
    assert drafts == sum(1 for es in shipped.contexts.values() for e in es if not e["reviewed"])
    assert reviewed == sum(1 for es in shipped.contexts.values() for e in es if e["reviewed"])
    first = next(iter(shipped.contexts))
    node = shipped.load_graph().nodes[first]
    choice = store.choose(first, node["strand"], node["grade"])
    assert choice["tier"] in ("reviewed", "bank")  # never a draft unless asked
    if reviewed == 0 and drafts:
        assert store.choose(first, node["strand"], node["grade"], include_drafts=True)["tier"] == "draft"


# ----------------------------------------------------------------------------- building


def test_build_bundle_writes_what_it_returns(tmp_path):
    doc = bundle.build_bundle(out=tmp_path, created_at=FIXED_TIME, commit="abc123")
    path = tmp_path / "kwan-bundle.math.json"
    assert path.is_file()
    loaded = bundle.load_bundle(path)
    assert loaded.to_dict() == doc
    assert doc["created_at"] == "2026-11-11T12:00:00+00:00" and doc["source_commit"] == "abc123"
    assert doc["provenance"]["params"]["source"] in ("defaults", str(ROOT / "data" / "params" / "math.params.json"))


def test_build_bundle_is_deterministic():
    a = bundle.build_bundle(created_at=FIXED_TIME, commit="x")
    b = bundle.build_bundle(created_at=FIXED_TIME, commit="x")
    assert bundle.dumps_bundle(a) == bundle.dumps_bundle(b)


def test_build_bundle_accepts_a_file_path_and_params(tmp_path):
    out = tmp_path / "custom.json"
    params = EloParams(k_s=0.25, tau_days=None)
    doc = bundle.build_bundle(params=params, out=out, include_drafts=False, bundle_version="math-9.9.9")
    assert out.is_file() and doc["params"] == params.to_dict() and doc["bundle_version"] == "math-9.9.9"
    assert all(e["reviewed"] for es in doc["contexts"].values() for e in es)
    assert bundle.load_bundle(out).elo_params() == params


def test_build_bundle_from_a_given_graph_and_contexts(tmp_path):
    G = nx.DiGraph(subject="math", text_mode="labels", schema_version=2)
    G.add_node("math:B1.1.1.1.1", grade=1, strand=1, strand_name="Number", short_label="Count to 20", code="B1.1.1.1.1")
    G.add_node("math:B1.1.1.1.2", grade=1, strand=1, strand_name="Number", short_label="Count to 50", code="B1.1.1.1.2")
    G.add_edge("math:B1.1.1.1.1", "math:B1.1.1.1.2", type="sequence", weight=1.0)
    contexts = {
        "contexts": {"B1.1.1.1.1": [{"text": "Count the eggs in a crate.", "lang": "en", "reviewed": True}]},
        "context_bank": {"1:lower": ["Count bottle tops."]},
    }
    doc = bundle.build_bundle(
        graph=G, contexts=contexts, i18n={"en": {"review": "Check again."}}, created_at=FIXED_TIME
    )
    assert [n["id"] for n in doc["graph"]["nodes"]] == ["math:B1.1.1.1.1", "math:B1.1.1.1.2"]
    assert doc["contexts"] == {
        "math:B1.1.1.1.1": [{"text": "Count the eggs in a crate.", "lang": "en", "reviewed": True}]
    }
    assert doc["context_bank"] == {"1:lower": ["Count bottle tops."]}
    assert doc["i18n"]["en"]["review"] == "Check again." and doc["i18n"]["en"]["builds_on"]
    loaded = bundle.Bundle.from_dict(doc)
    assert loaded.load_graph().nodes["math:B1.1.1.1.2"]["description"] == "Count to 50"


def test_a_full_text_graph_is_stripped_in_labels_mode():
    G = nx.DiGraph(subject="math", text_mode="full", schema_version=2)
    G.add_node(
        "math:B1.1.1.1.1", grade=1, strand=1, strand_name="Number", short_label="Count", description="SECRET TEXT"
    )
    doc = bundle.build_bundle(graph=G, contexts={"contexts": {}, "context_bank": {}}, created_at=FIXED_TIME)
    assert doc["text_mode"] == "labels"
    assert "SECRET TEXT" not in bundle.dumps_bundle(doc)


# ----------------------------------------------------------------------------- licence checks


def test_the_registry_knows_the_evaluation_only_datasets():
    assert set(non_commercial()) == {"ammore_2024", "junyi_2015"}


@pytest.mark.parametrize("dataset", ["ammore_2024", "ammore", "junyi_2015", "junyi", "ammore_2024:ghana"])
def test_params_fitted_on_non_commercial_data_are_refused(dataset):
    with pytest.raises(bundle.LicenceError, match="evaluation only"):
        bundle.build_bundle(params=EloParams(), params_provenance={"datasets": [dataset]})


def test_an_item_bank_fitted_on_non_commercial_data_is_refused():
    bank = ItemBank({"q1": 0.3}, provenance={"datasets": ["assistments_2009", "junyi_2015"]})
    with pytest.raises(bundle.LicenceError, match="item_bank"):
        bundle.build_bundle(item_bank=bank)


def test_an_unknown_dataset_never_passes_as_permission():
    with pytest.raises(bundle.LicenceError, match="unknown dataset"):
        bundle.build_bundle(params_provenance={"datasets": ["my_private_dump"]})


def test_commercial_ok_provenance_is_kept(tmp_path):
    bank = ItemBank({"q1": 0.3}, provenance={"datasets": ["assistments_2009"], "fitted_at": "2026-11-11"})
    doc = bundle.build_bundle(
        params=EloParams(),
        params_provenance={"datasets": ["eedi_2020", "assistments_2009"]},
        item_bank=bank,
        created_at=FIXED_TIME,
    )
    assert doc["provenance"]["params"]["datasets"] == ["eedi_2020", "assistments_2009"]
    assert doc["provenance"]["item_bank"]["datasets"] == ["assistments_2009"]
    assert doc["item_bank"] == bank.to_dict()
    assert bundle.Bundle.from_dict(doc).load_item_bank() == bank


def test_a_params_file_carries_its_provenance(tmp_path):
    path = tmp_path / "math.params.json"
    path.write_text(json.dumps({"params": {"k_s": 0.2}, "provenance": {"datasets": ["ammore_2024"]}}), encoding="utf-8")
    params, prov = bundle.load_params_file(path)
    assert params.k_s == 0.2 and prov["datasets"] == ["ammore_2024"] and prov["source"] == str(path)
    with pytest.raises(bundle.LicenceError):
        bundle.build_bundle(params=params, params_provenance=prov)
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"k_k": 0.5, "datasets": ["xes3g5m"]}), encoding="utf-8")
    params, prov = bundle.load_params_file(flat)
    assert params.k_k == 0.5 and prov["datasets"] == ["xes3g5m"]


# ----------------------------------------------------------------------------- full text mode


def test_full_mode_is_refused_without_recorded_permission(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "_permission_granted", lambda: (False, tmp_path / "permissions.json"))
    if not FULL_GRAPH.is_file():
        pytest.skip("no full-text graph on this machine")
    with pytest.raises(bundle.BundleError, match="written permission"):
        bundle.build_bundle(text_mode="full")


def test_full_mode_is_refused_without_the_full_graph(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "_permission_granted", lambda: (True, tmp_path / "permissions.json"))
    monkeypatch.setattr(bundle, "_full_graph_path", lambda subject: tmp_path / "missing.json")
    with pytest.raises(bundle.BundleError, match="is not there"):
        bundle.build_bundle(text_mode="full")


def test_full_mode_builds_when_allowed(monkeypatch, tmp_path):
    if not FULL_GRAPH.is_file():
        pytest.skip("no full-text graph on this machine")
    monkeypatch.setattr(bundle, "_permission_granted", lambda: (True, tmp_path / "permissions.json"))
    doc = bundle.build_bundle(text_mode="full", out=tmp_path, created_at=FIXED_TIME)  # tmp only: never committed
    assert doc["text_mode"] == "full"
    assert any(n.get("description") not in (None, "", n["short_label"]) for n in doc["graph"]["nodes"])
    loaded = bundle.load_bundle(tmp_path / "kwan-bundle.math.json")
    assert loaded.load_graph().graph["text_mode"] == "full"


def test_the_committed_permissions_file_still_says_no():
    doc = json.loads((ROOT / "data" / "curriculum" / "permissions.json").read_text(encoding="utf-8"))
    assert doc["nacca_text_commit_allowed"] is False
    assert bundle._permission_granted()[0] is False


# ----------------------------------------------------------------------------- loading and validation


def test_load_bundle_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        bundle.load_bundle(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="not valid JSON"):
        bundle.load_bundle(bad)
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="JSON object"):
        bundle.load_bundle(bad)


def test_validate_bundle_reports_what_is_missing_or_wrong(shipped):
    doc = shipped.to_dict()
    with pytest.raises(bundle.BundleError, match="missing graph, params"):
        bundle.validate_bundle({k: v for k, v in doc.items() if k not in ("graph", "params")})
    with pytest.raises(bundle.BundleError, match="schema_version 3"):
        bundle.validate_bundle({**doc, "schema_version": 3})
    with pytest.raises(bundle.BundleError, match="text_mode"):
        bundle.validate_bundle({**doc, "text_mode": "rich"})
    with pytest.raises(bundle.BundleError, match="no nodes"):
        bundle.validate_bundle({**doc, "graph": {"nodes": [], "edges": []}})
    with pytest.raises(bundle.BundleError, match="params"):
        bundle.validate_bundle({**doc, "params": {"k_s": -1}})
    with pytest.raises(bundle.BundleError, match="contexts"):
        bundle.validate_bundle({**doc, "contexts": {"math:B1.1.1.1.1": [{"lang": "en"}]}})
    bundle.validate_bundle(doc)


def test_the_bundle_env_variable_picks_another_file(monkeypatch, tmp_path):
    out = bundle.write_bundle(bundle.build_bundle(created_at=FIXED_TIME, bundle_version="math-0.0.1"), tmp_path)
    monkeypatch.setenv("KWAN_BUNDLE", str(out))
    assert bundle.default_bundle_path() == out.resolve()
    assert bundle.load_bundle().bundle_version == "math-0.0.1"


def test_source_commit_is_a_hash_or_unknown(tmp_path):
    text = bundle.source_commit()
    assert text == "unknown" or (len(text) == 40 and all(c in "0123456789abcdef" for c in text))
    assert bundle.source_commit(tmp_path) == "unknown"


# ----------------------------------------------------------------------------- the command


def test_kwan_bundle_writes_the_file(tmp_path, capsys):
    assert cli.main(["bundle", "--subject", "math", "--text", "labels", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    path = tmp_path / "kwan-bundle.math.json"
    assert path.is_file() and str(path) in out and "276 skills" in out and "bytes" in out
    assert bundle.load_bundle(path).text_mode == "labels"


def test_kwan_bundle_refuses_full_without_permission(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bundle, "_permission_granted", lambda: (False, tmp_path / "permissions.json"))
    if not FULL_GRAPH.is_file():
        monkeypatch.setattr(bundle, "_full_graph_path", lambda subject: FULL_GRAPH)
        monkeypatch.setattr(Path, "is_file", lambda self: True if self == FULL_GRAPH else Path.exists(self))
    assert cli.main(["bundle", "--text", "full", "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("kwan bundle:") and "permission" in err
    assert not (tmp_path / "kwan-bundle.math.json").exists()


def test_kwan_bundle_takes_params_and_an_item_bank(tmp_path, capsys):
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"params": {"k_s": 0.21}, "provenance": {"datasets": ["assistments_2009"]}}))
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(ItemBank({"q1": 0.1}, provenance={"datasets": ["eedi_2020"]}).to_dict()))
    code = cli.main(
        ["bundle", "--out", str(tmp_path), "--params", str(params), "--item-bank", str(bank), "--reviewed-only"]
    )
    assert code == 0
    loaded = bundle.load_bundle(tmp_path / "kwan-bundle.math.json")
    assert loaded.params["k_s"] == 0.21 and loaded.provenance["params"]["datasets"] == ["assistments_2009"]
    assert loaded.load_item_bank() is not None and loaded.provenance["contexts"]["drafts"] == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"params": {}, "provenance": {"datasets": ["junyi_2015"]}}))
    assert cli.main(["bundle", "--out", str(tmp_path), "--params", str(bad)]) == 1
    assert "evaluation only" in capsys.readouterr().err


def test_the_command_is_registered():
    assert "kwan._bundle_cli" in cli.REGISTRY and _bundle_cli.DEFAULT_OUT == "build/bundles"
