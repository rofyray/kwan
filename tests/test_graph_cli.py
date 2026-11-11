"""`kwan graph`: the three graph commands are registered, and the decision-model commands join when present."""

import argparse
import sys
import types

import pytest

from kwan import cli
from kwan.graph import _cli


def graph_actions():
    parser = argparse.ArgumentParser(prog="kwan")
    _cli.register(parser.add_subparsers(dest="command"))
    graph = parser._subparsers._group_actions[0].choices["graph"]
    return parser, set(graph._subparsers._group_actions[0].choices)


def test_build_validate_and_export_are_registered():
    _, actions = graph_actions()
    assert {"build", "validate", "export"} <= actions


def test_register_into_of_the_augment_module_is_called_when_it_exists(monkeypatch):
    calls = []

    def register_into(graph_subparsers):
        calls.append(graph_subparsers)
        graph_subparsers.add_parser("augment-test").set_defaults(func=lambda args: 7)

    fake = types.ModuleType("kwan.graph._cli_augment")
    fake.register_into = register_into
    monkeypatch.setitem(sys.modules, "kwan.graph._cli_augment", fake)
    parser, actions = graph_actions()
    assert len(calls) == 1 and "augment-test" in actions
    args = parser.parse_args(["graph", "augment-test"])
    assert args.func(args) == 7


def test_graph_commands_work_without_the_augment_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "kwan.graph._cli_augment", None)  # importing it now raises ImportError
    _, actions = graph_actions()
    assert {"build", "validate", "export"} <= actions


def test_kwan_graph_needs_an_action(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["graph"])
    assert stop.value.code == 2 and "ACTION" in capsys.readouterr().err


def test_help_text_has_no_em_dash(capsys):
    for argv in (
        ["graph", "--help"],
        ["graph", "build", "--help"],
        ["graph", "validate", "--help"],
        ["graph", "export", "--help"],
    ):
        with pytest.raises(SystemExit):
            cli.main(argv)
    text = capsys.readouterr().out
    assert "build" in text and "--structure" in text and "—" not in text and "–" not in text


def test_export_declares_the_demo_extra():
    parser, _ = graph_actions()
    assert parser.parse_args(["graph", "export", "--out", "x.html"]).extra == "demo"
    assert not hasattr(parser.parse_args(["graph", "build"]), "extra")
