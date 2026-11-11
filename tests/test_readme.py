"""The README for 1.0 (PRD Phase 11.1 and the Phase 11 gate).

- every number in a README RESULTS block comes from data/eval_results.json (scripts/fill_docs.py --check);
- no em dash, no Round 1 placeholders, the NaCCA disclaimer is present, no email address;
- every relative link resolves;
- every fenced bash block starts with a real command, and every `kwan ...` command it names exists.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "data" / "eval_results.json"
EM_DASH = chr(0x2014)

PLACEHOLDERS = ("[DEMO URL]", "[VIDEO URL]", "[Names and roles]", "[Name]", "[email]")
DISCLAIMER = "Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education."
COMMANDS = {"pip", "make", "kwan", "python", "curl", "docker", "uvicorn", "streamlit", "git"}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[^\n]*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
EXTERNAL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
BLOCK = re.compile(r"<!-- RESULTS:START id=([\w-]+) source=([\w./-]+) -->(.*?)<!-- RESULTS:END -->", re.S)


def _load_fill_docs():
    spec = importlib.util.spec_from_file_location("fill_docs", ROOT / "scripts" / "fill_docs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fill_docs():
    return _load_fill_docs()


@pytest.fixture(scope="module")
def text() -> str:
    return README.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# numbers come from the results file


def test_every_results_block_matches_the_results_file(fill_docs, capsys) -> None:
    """scripts/fill_docs.py --check, in process: exit 0 means every block equals what the file renders."""
    code = fill_docs.main(["--check"])
    out = capsys.readouterr().out
    assert code == 0, f"RESULTS blocks are out of date; run `python scripts/fill_docs.py`:\n{out}"


def test_readme_blocks_render_from_the_file_one_by_one(fill_docs, text) -> None:
    res = fill_docs.load_results(RESULTS)
    blocks = BLOCK.findall(text)
    assert blocks, "the README keeps its numbers in RESULTS blocks"
    for block_id, source, body in blocks:
        assert source == "data/eval_results.json"
        assert block_id in fill_docs.RENDERERS, f"no renderer for {block_id}"
        assert body == fill_docs.render(block_id, res), f"{block_id} differs from the results file"


def test_readme_has_the_blocks_the_brief_asks_for(text) -> None:
    ids = {m.group(1) for m in BLOCK.finditer(text)}
    for needed in (
        "readme-kt-table",
        "readme-ghana",
        "readme-learner-fit",
        "readme-deep-kt",
        "readme-simulation-v2",
        "readme-simulation-v1",
        "readme-simulation-assumptions",
        "readme-goal-path",
        "readme-latency",
        "readme-curriculum",
        "readme-graph",
        "readme-graph-augment",
        "readme-contexts",
    ):
        assert needed in ids, f"README lacks the {needed} block"
    assert len(ids) == text.count("RESULTS:START") == text.count("<!-- RESULTS:END -->")


def test_readme_blocks_are_filled(text, fill_docs) -> None:
    res = fill_docs.load_results(RESULTS)
    for block_id, _, body in BLOCK.findall(text):
        assert "Results are not filled in yet" not in body, block_id
        if fill_docs.NOT_RUN in body:
            key = re.search(r"`([\w.]+)` is not in", body)
            assert key, f"{block_id}: a block that is not run must name the missing key:\n{body}"
            assert key.group(1).split(".")[0] not in res, f"{block_id} says not run but {key.group(1)} is in the file"


# ---------------------------------------------------------------------------------------------------
# copy rules


def test_no_em_dash(text) -> None:
    bad = [f"{n}: {line.strip()[:80]}" for n, line in enumerate(text.splitlines(), 1) if EM_DASH in line]
    assert not bad, "em dash found in README.md:\n" + "\n".join(bad)


def test_no_round_1_placeholders(text) -> None:
    found = [p for p in PLACEHOLDERS if p in text]
    assert not found, f"README still holds placeholders: {found}"
    assert "## Team" not in text


def test_nacca_disclaimer_is_present(text) -> None:
    assert DISCLAIMER in " ".join(text.split())


def test_no_email_address(text) -> None:
    assert not EMAIL.findall(text)


def test_demo_url_and_maintainer(text) -> None:
    assert "https://huggingface.co/spaces/rofyray/kwan" in text
    assert "Maintainer: Raymond Okyere-Forson (@rofyray)" in text
    assert "## Data and credits" in text and "## Licences" in text
    assert "CC BY 4.0" in text and "`LICENSE-CONTENT`" in text


def test_readme_does_not_mention_the_planning_kit(text) -> None:
    for banned in ("KWAN_PRD", "kwan_v2_kit", "Devpost"):
        assert banned not in text, f"README mentions {banned}, which is not part of the repository"


# ---------------------------------------------------------------------------------------------------
# links


def _link_targets(text: str) -> list[str]:
    stripped = HTML_COMMENT.sub("", FENCE.sub("", text))
    stripped = INLINE_CODE.sub("", stripped)
    return INLINE_LINK.findall(stripped)


def _slug(heading: str) -> str:
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    heading = re.sub(r"[^\w\- ]", "", heading)
    return re.sub(r"\s+", "-", heading).strip("-")


def _anchors(path: Path) -> set[str]:
    body = FENCE.sub("", path.read_text(encoding="utf-8"))
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", body, re.M)}


def test_relative_links_resolve(text) -> None:
    broken = []
    for target in _link_targets(text):
        if EXTERNAL.match(target) or target.startswith("//"):
            continue
        file_part, _, anchor = target.partition("#")
        dest = README if not file_part else (ROOT / file_part).resolve()
        if file_part.startswith("/"):
            broken.append(f"{target} (absolute path)")
        elif not dest.exists():
            broken.append(f"{target} (no such file)")
        elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            broken.append(f"{target} (no heading gives #{anchor})")
    assert not broken, "broken links in README.md:\n" + "\n".join(broken)


# ---------------------------------------------------------------------------------------------------
# commands


def _bash_blocks(text: str) -> list[str]:
    return [body for lang, body in FENCE.findall(text) if lang == "bash"]


def test_every_bash_block_starts_with_a_real_command(text) -> None:
    blocks = _bash_blocks(text)
    assert blocks, "the README should show bash examples"
    for body in blocks:
        first = body.strip().split()[0]
        assert first in COMMANDS, f"bash block starts with {first!r}, not a known command:\n{body[:120]}"


@cache
def _help(*args: str) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "kwan.cli", *args, "--help"], capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def _commands_in(help_text: str) -> set[str]:
    """The names listed under the positional-arguments heading of an argparse help text."""
    names: set[str] = set()
    for line in help_text.splitlines():
        m = re.match(r"^\s{2,6}([a-z][\w-]*)\s{2,}", line)
        if m:
            names.add(m.group(1))
    return names


def _kwan_invocations(text: str) -> set[tuple[str, ...]]:
    found: set[tuple[str, ...]] = set()
    for body in _bash_blocks(text):
        for line in body.splitlines():
            words = line.split("#", 1)[0].split()
            if words and words[0] == "kwan":
                found.add(tuple(w for w in words[1:3] if re.fullmatch(r"[a-z][\w-]*", w)))
    for code in INLINE_CODE.findall(HTML_COMMENT.sub("", text)):
        words = code.strip("`").split()
        if len(words) >= 2 and words[0] == "kwan" and re.fullmatch(r"[a-z][\w-]*", words[1]):
            sub = words[2] if len(words) > 2 and re.fullmatch(r"[a-z][\w-]*", words[2]) else None
            found.add((words[1], sub) if sub else (words[1],))
    return {f for f in found if f}


def test_cli_commands_named_in_the_readme_exist(text) -> None:
    top = _commands_in(_help())
    assert {"curriculum", "graph", "eval", "sim", "bench", "bundle"} <= top
    invocations = _kwan_invocations(text)
    assert invocations, "the README should name kwan commands"
    missing = []
    for inv in sorted(invocations):
        if inv[0] not in top:
            missing.append(" ".join(inv))
        elif len(inv) > 1 and inv[1] not in _commands_in(_help(inv[0])):
            missing.append(" ".join(inv))
    assert not missing, f"README names kwan commands that do not exist: {missing}"


# ---------------------------------------------------------------------------------------------------
# the script itself


def test_fill_docs_is_standard_library_only() -> None:
    source = (ROOT / "scripts" / "fill_docs.py").read_text(encoding="utf-8")
    imports = re.findall(r"^(?:from|import)\s+([\w.]+)", source, re.M)
    allowed = {"argparse", "json", "re", "sys", "collections.abc", "decimal", "pathlib", "typing", "__future__"}
    assert set(imports) <= allowed, set(imports) - allowed
    assert EM_DASH not in source


def test_fill_docs_rounds_half_up(fill_docs) -> None:
    assert fill_docs.num(0.7745, 3) == "0.775"
    assert fill_docs.num(0.185, 2) == "0.19"
    assert fill_docs.num(14.05, 1) == "14.1"
    assert fill_docs.count(1382727) == "1,382,727"
    assert fill_docs.pct(0.773) == "77.3%"


def test_fill_docs_says_not_run_for_a_missing_block(fill_docs) -> None:
    body = fill_docs.render("readme-latency", {})
    assert fill_docs.NOT_RUN in body and "`latency`" in body
