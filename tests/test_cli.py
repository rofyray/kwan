"""The `kwan` command (registry of subcommands) and the data-directory lookup."""

import subprocess
import sys
import types
from pathlib import Path

import pytest

import kwan
from kwan import _paths, cli

ROOT = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ----------------------------------------------------------------------------- version and help


def test_version_comes_from_the_single_source(capsys):
    from kwan._version import __version__

    assert kwan.__version__ == __version__
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"kwan {__version__}"


def test_no_command_prints_the_help(capsys):
    code, out, _ = run(capsys)
    assert code == 0
    for command in ("eval", "sim", "bench"):
        assert command in out
    assert EM_DASH not in out


def test_console_script_is_installed():
    done = subprocess.run([sys.executable, "-m", "kwan.cli", "--version"], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0 and done.stdout.strip() == f"kwan {kwan.__version__}"


@pytest.mark.parametrize(
    "argv",
    [
        ["eval", "--help"],
        ["eval", "assistments-2009", "--help"],
        ["eval", "eedi-34", "--help"],
        ["eval", "ammore", "--help"],
        ["sim", "--help"],
        ["bench", "--help"],
    ],
)
def test_every_command_has_help_without_em_dashes(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: kwan ") and EM_DASH not in out


def test_eval_needs_a_dataset_and_rejects_an_unknown_one(capsys):
    for argv in (["eval"], ["eval", "nope"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
    capsys.readouterr()


# ----------------------------------------------------------------------------- wiring to the existing mains


def test_sim_and_bench_are_wired_to_the_existing_mains(capsys, monkeypatch, tmp_path):
    from kwan.eval import latency, simulate

    calls = []
    monkeypatch.setattr(simulate, "main", lambda argv=None: calls.append(("sim", argv)))
    monkeypatch.setattr(latency, "main", lambda argv=None: calls.append(("bench", argv)))
    out = str(tmp_path / "r.json")
    assert cli.main(["sim"]) == 0
    assert cli.main(["sim", "g.json", "--out", out]) == 0
    assert cli.main(["bench", "--calls", "5", "--hardware", "test rig", "--out", out]) == 0
    assert calls == [
        ("sim", []),
        ("sim", ["g.json", "--out", out]),
        ("bench", ["--out", out, "--calls", "5", "--hardware", "test rig"]),
    ]


def test_eval_commands_are_wired_to_the_existing_mains(monkeypatch, tmp_path):
    from kwan.eval import ammore, eedi, kt

    calls = []
    for name, module in (("assistments", kt), ("eedi", eedi), ("ammore", ammore)):
        monkeypatch.setattr(module, "main", lambda argv=None, name=name: calls.append((name, argv)))
    out = str(tmp_path / "r.json")
    assert cli.main(["eval", "assistments-2009", "a.csv", "--bkt", "--out", out]) == 0
    assert cli.main(["eval", "eedi-34", "raw/eedi/data"]) == 0
    assert cli.main(["eval", "ammore", "raw/ammore", "--out", out]) == 0
    assert calls == [
        ("assistments", ["a.csv", "--bkt", "--out", out]),
        ("eedi", ["raw/eedi/data"]),
        ("ammore", ["raw/ammore", "--out", out]),
    ]


def test_bench_runs_for_real_through_the_command(capsys):
    assert cli.main(["bench", "--calls", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4 and lines[-1].startswith("all")


# ----------------------------------------------------------------------------- the registry


def test_registry_lists_the_agreed_modules():
    assert cli.REGISTRY == (
        "kwan.curriculum._cli",
        "kwan.graph._cli",
        "kwan.eval._cli",
        "kwan.eval.fit",
        "kwan.context._cli",
        "kwan._bundle_cli",
    )


def test_a_registry_module_that_does_not_exist_is_skipped_quietly(capsys, monkeypatch):
    monkeypatch.setattr(cli, "REGISTRY", ("kwan._no_such_cli", "kwan.eval._cli"))
    parser, skipped = cli.build_parser()
    assert list(skipped) == ["kwan._no_such_cli"]
    assert cli.main([]) == 0  # the help still lists what is available
    out, err = capsys.readouterr()
    assert "sim" in out and err == ""
    parser.parse_args(["bench", "--calls", "1"])  # and the available commands parse
    assert capsys.readouterr() == ("", "")


def test_a_new_registry_module_adds_its_command(capsys, monkeypatch):
    seen = []

    def register(subparsers):
        p = subparsers.add_parser("hello", help="say hello")
        p.add_argument("name")
        p.set_defaults(func=lambda args: seen.append(args.name))

    module = types.ModuleType("kwan._hello_cli")
    module.register = register
    monkeypatch.setitem(sys.modules, "kwan._hello_cli", module)
    monkeypatch.setattr(cli, "REGISTRY", (*cli.REGISTRY, "kwan._hello_cli"))
    assert cli.main(["hello", "Ama"]) == 0 and seen == ["Ama"]


def test_asking_for_a_command_whose_module_is_missing_prints_a_hint(capsys, monkeypatch):
    monkeypatch.setattr(cli, "REGISTRY", ("kwan._no_such_cli",))
    monkeypatch.setitem(cli.COMMANDS, "bundle", ("kwan._no_such_cli", None))
    code, out, err = run(capsys, "bundle")
    assert code == 2 and out == ""
    assert err.startswith("kwan bundle:") and "not part of this install" in err


def test_a_module_with_a_missing_optional_package_is_skipped_and_hinted(capsys, monkeypatch):
    class Broken(types.ModuleType):
        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named 'pdfplumber'", name="pdfplumber")

    monkeypatch.setitem(sys.modules, "kwan.curriculum._cli", Broken("kwan.curriculum._cli"))
    code, out, err = run(capsys)  # every other command still works
    assert code == 0 and "sim" in out and err == ""
    code, out, err = run(capsys, "curriculum", "parse")
    assert (
        code == 2 and "pdfplumber" in err and "uv add 'kwan[curriculum]'" in err and "uv sync --extra curriculum" in err
    )


def test_a_command_that_needs_a_missing_package_at_run_time_prints_the_extra(capsys, monkeypatch):
    def register(subparsers):
        p = subparsers.add_parser("heavy")

        def handler(args):
            raise ModuleNotFoundError("No module named 'pandas'", name="pandas")

        p.set_defaults(func=handler, extra="eval")

    module = types.ModuleType("kwan._heavy_cli")
    module.register = register
    monkeypatch.setitem(sys.modules, "kwan._heavy_cli", module)
    monkeypatch.setattr(cli, "REGISTRY", ("kwan._heavy_cli",))
    code, out, err = run(capsys, "heavy")
    assert code == 1 and "'pandas' is not installed" in err and "uv add 'kwan[eval]'" in err


# ----------------------------------------------------------------------------- data directory


def test_data_dir_is_the_repo_data_folder_by_default(monkeypatch):
    monkeypatch.delenv(_paths.ENV_VAR, raising=False)
    assert _paths.repo_root() == ROOT
    assert _paths.data_dir() == ROOT / "data"
    assert _paths.data_path("eval_results.json") == ROOT / "data" / "eval_results.json"
    assert _paths.data_path("eval_results.json").is_file() and _paths.data_dir().is_absolute()


def test_data_dir_follows_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(_paths.ENV_VAR, str(tmp_path))
    assert _paths.data_dir() == tmp_path.resolve()
    assert _paths.data_path("eval_results.json") == tmp_path.resolve() / "eval_results.json"


def test_importing_kwan_does_not_import_the_optional_packages():
    code = (
        "import sys, kwan, kwan.cli, kwan.learner.elo, kwan.recommend.recommender, kwan.graph.build, "
        "kwan.context.bank, kwan.eval.simulate, kwan.eval.latency, kwan.eval.results, kwan.eval._cli\n"
        "heavy = [m for m in ('pandas', 'numpy', 'sklearn', 'fastapi', 'streamlit', 'matplotlib') if m in sys.modules]\n"
        "print(heavy)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT.parent)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


OLD_ROOT_MODULES = (
    "api",
    "app",
    "bench_latency",
    "context",
    "engine",
    "eval_eedi",
    "eval_kt",
    "graph",
    "parse_curriculum",
    "results",
    "simulate",
)


def test_the_v1_compatibility_shims_and_the_v1_data_files_are_gone():
    """PRD 11.7 and decision D5: the root shims, the root test file and the v1 data files are removed."""
    for name in OLD_ROOT_MODULES:
        assert not (ROOT / f"{name}.py").exists(), name
    assert not (ROOT / "test_kwan.py").exists()
    assert not (ROOT / "data" / "graph.json").exists() and not (ROOT / "data" / "indicators.json").exists()
