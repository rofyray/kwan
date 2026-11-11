"""The dataset registry: data/datasets.yaml and the licence table in kwan.data.registry."""

import subprocess
import sys
from pathlib import Path

import pytest

from kwan.data import registry

ROOT = Path(__file__).resolve().parent.parent
YAML = ROOT / "data" / "datasets.yaml"
EM_DASH = chr(0x2014)

# PRD Section 5.2: every learner dataset of the 1.0 scope, and whether fitted parameters may ship
EXPECTED = {
    "assistments_2009": True,
    "eedi_2020": True,
    "ammore_2024": False,
    "assistments_2012": True,
    "assistments_2015": True,
    "assistments_2017": True,
    "xes3g5m": True,
    "junyi_2015": False,
}
RESULT_KEYS = [
    "assistments_2009",
    "eedi_2020_task34",
    "eedi_2020_task12",
    "ammore_2024",
    "assistments_2012",
    "assistments_2015",
    "assistments_2017",
    "xes3g5m",
    "junyi_2015",
]


@pytest.fixture(scope="module")
def reg():
    return registry.load_registry(YAML)


def test_registry_lists_every_dataset_of_the_prd(reg):
    assert list(reg) == list(EXPECTED)
    assert list(registry.FACTS) == list(EXPECTED)


def test_every_entry_has_the_agreed_fields(reg):
    for key, entry in reg.items():
        for field in ("name", "url", "size", "licence", "citation", "role", "commercial_params_ok", "raw", "files"):
            assert entry.get(field) not in (None, "", []), f"{key}.{field}"
        assert entry["url"].startswith(("http://", "https://"))
        assert isinstance(entry["commercial_params_ok"], bool)
        assert isinstance(entry["role"], list) and entry["role"]
        assert len(entry["citation"]) > 30, key
        assert entry["fetch"]["method"] in ("zip", "git", "gdown", "rar")


def test_python_table_and_yaml_agree(reg):
    """The bundle builder reads the Python table; the YAML file is what people edit. They must not drift."""
    assert set(registry.FACTS) == set(reg)
    for key, entry in reg.items():
        licence, ok = registry.FACTS[key]
        assert entry["licence"] == licence, key
        assert entry["commercial_params_ok"] is ok, key
        assert registry.licence(key) == entry["licence"]
        assert registry.commercial_ok(key) is entry["commercial_params_ok"]
    assert {k: ok for k, (_, ok) in registry.FACTS.items()} == EXPECTED


def test_results_keys_resolve_to_their_registry_entry(reg):
    listed = [k for entry in reg.values() for k in entry["results_keys"]]
    assert sorted(listed) == sorted(RESULT_KEYS)
    for key, entry in reg.items():
        for results_key in entry["results_keys"]:
            assert registry.dataset_key(results_key) == key
            assert registry.commercial_ok(results_key) is entry["commercial_params_ok"]
            assert registry.get(results_key, YAML) == entry


def test_non_commercial_datasets_are_ammore_and_junyi():
    assert registry.non_commercial() == ("ammore_2024", "junyi_2015")
    for name in ("ammore_2024", "ammore", "ammore_2024:ghana", "ammore_2024_all", "junyi_2015", "junyi"):
        assert registry.commercial_ok(name) is False
    for name in (
        "assistments_2009",
        "assistments-2012",
        "eedi_2020",
        "eedi_2020_task12",
        "eedi_2020:task34",
        "xes3g5m",
    ):
        assert registry.commercial_ok(name) is True
    assert "NC" in registry.licence("ammore_2024") and "non-commercial" in registry.licence("junyi_2015")


def test_unknown_dataset_is_refused_not_guessed():
    for name in ("nope", "", "assistments", "eedi", "ammore_2025"):
        with pytest.raises(KeyError, match="unknown dataset"):
            registry.commercial_ok(name)
        with pytest.raises(KeyError, match="unknown dataset"):
            registry.licence(name)


def test_commercial_ok_needs_no_optional_package():
    """It must answer in a core install: PyYAML, pandas and numpy are made impossible to import."""
    code = (
        "import sys\n"
        "for name in ('yaml', 'pandas', 'numpy', 'pyarrow', 'sklearn'):\n"
        "    sys.modules[name] = None\n"
        "from kwan.data import registry\n"
        "import kwan.data.loaders, kwan.data.schema\n"
        "print(registry.commercial_ok('ammore_2024'), registry.commercial_ok('eedi_2020_task12'))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["False", "True"]


def test_load_registry_default_path_and_bad_file(tmp_path, monkeypatch):
    assert registry.registry_path() == YAML
    monkeypatch.setenv("KWAN_DATA_DIR", str(tmp_path))
    (tmp_path / "datasets.yaml").write_text("toy: {licence: MIT, commercial_params_ok: true}\n", encoding="utf-8")
    assert registry.load_registry() == {"toy": {"licence": "MIT", "commercial_params_ok": True}}
    (tmp_path / "datasets.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        registry.load_registry()


def test_raw_path_follows_the_registry(reg):
    assert registry.raw_path("junyi_2015", "raw", YAML) == Path("raw/junyi")
    assert registry.raw_path("eedi_2020_task12", "/data", YAML) == Path("/data/eedi/data")


def test_loaders_know_where_every_dataset_lives(reg):
    from kwan.data import loaders

    for entry in reg.values():
        for results_key in entry["results_keys"]:
            assert loaders.raw_path(results_key) == Path("raw") / entry["raw"], results_key


def test_download_script_fetches_every_dataset_in_the_registry(reg):
    script = (ROOT / "scripts" / "download_data.sh").read_text(encoding="utf-8")
    for key, entry in reg.items():
        assert entry["url"] in script, f"scripts/download_data.sh does not fetch {key}"
        fetch = entry["fetch"]
        if "archive" in fetch:
            assert fetch["archive"] in script, key
        if "drive_id" in fetch:
            assert fetch["drive_id"] in script
        for name in entry["files"]:
            assert Path(name).name in script, f"{key}: {name} is not checked by the download script"


def test_registry_text_has_no_em_dash():
    assert EM_DASH not in YAML.read_text(encoding="utf-8")
    assert EM_DASH not in Path(registry.__file__).read_text(encoding="utf-8")
