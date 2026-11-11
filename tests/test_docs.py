"""Checks on the documentation, licensing and community files.

- no em dash in any Markdown, YAML or CFF file of the docs and community set;
- every relative link in docs/ and in the community files points to a file that exists;
- CITATION.cff parses and has the keys the Citation File Format 1.2.0 requires;
- every file named in the mkdocs nav exists, the QA reports included;
- the RESULTS blocks are well formed and filled from data/eval_results.json by scripts/fill_docs.py;
- `mkdocs build --strict` passes (skipped where mkdocs-material is not installed; KWAN_MKDOCS can name a binary).

PyYAML comes with the `dev` extra. Where it is not installed, the two YAML checks fall back to a
small reader that understands the plain shapes these files use, so the suite still runs.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:  # PyYAML is in the dev extra; the fallbacks below keep the tests running without it
    import yaml
except ImportError:  # pragma: no cover - depends on the environment
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EM_DASH = chr(0x2014)  # built from its code point so that this file holds no em dash itself

# docs/qa/ is written by the curriculum and graph QA steps and docs/history/ holds planning papers;
# neither belongs to this set.
NOT_OURS = ("qa", "history")

COMMUNITY_FILES = (
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTORS.md",
    "LICENSE-CONTENT",
    "CITATION.cff",
    "mkdocs.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/workflows/docs.yml",
)

DOC_PAGES = (
    "index.md",
    "quickstart.md",
    "method.md",
    "evidence.md",
    "data-card.md",
    "model-card.md",
    "context-style-guide.md",
    "teacher-review-guide.md",
    "governance.md",
    "roadmap.md",
)

ISSUE_FORMS = ("bug.yml", "feature.yml", "teacher-review.yml", "curriculum-error.yml", "config.yml")


def _docs_markdown() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if p.relative_to(DOCS).parts[0] not in NOT_OURS)


def _owned_files() -> list[Path]:
    files = [ROOT / name for name in COMMUNITY_FILES]
    files += [ROOT / ".github" / "ISSUE_TEMPLATE" / name for name in ISSUE_FORMS]
    files += _docs_markdown()
    return files


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


OWNED = _owned_files()
MARKDOWN_WITH_LINKS = [p for p in OWNED if p.suffix == ".md"]


# ---------------------------------------------------------------------------------------------------
# the files are there


@pytest.mark.parametrize("path", OWNED, ids=_rel)
def test_file_exists_and_is_not_empty(path: Path) -> None:
    assert path.is_file(), f"{_rel(path)} is missing"
    assert path.read_text(encoding="utf-8").strip(), f"{_rel(path)} is empty"


@pytest.mark.parametrize("name", DOC_PAGES)
def test_every_planned_docs_page_exists(name: str) -> None:
    assert (DOCS / name).is_file(), f"docs/{name} is missing"


# ---------------------------------------------------------------------------------------------------
# copy rules


@pytest.mark.parametrize("path", OWNED, ids=_rel)
def test_no_em_dash(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = [f"{_rel(path)}:{n}: {line.strip()[:80]}" for n, line in enumerate(lines, 1) if EM_DASH in line]
    assert not bad, "em dash found (the copy rules ban it):\n" + "\n".join(bad)


EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# NaCCA's public contact address, needed in the permission request. No personal address is allowed.
ALLOWED_EMAILS = {"info@nacca.gov.gh"}


@pytest.mark.parametrize("path", OWNED, ids=_rel)
def test_no_personal_email_address(path: Path) -> None:
    found = set(EMAIL.findall(path.read_text(encoding="utf-8"))) - ALLOWED_EMAILS
    assert not found, f"{_rel(path)} holds an email address: {sorted(found)}"


def test_code_of_conduct_has_no_email_and_a_contact_method() -> None:
    text = (ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    assert not EMAIL.findall(text)
    assert "INSERT CONTACT METHOD" not in text
    assert "github.com/rofyray/kwan" in text
    assert "version 2.1" in text


DISCLAIMER = "Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education."


@pytest.mark.parametrize(
    "name",
    [
        "docs/index.md",
        "docs/governance.md",
        "docs/data-card.md",
        "docs/model-card.md",
        "docs/method.md",
        "docs/context-style-guide.md",
        "docs/teacher-review-guide.md",
        "CONTRIBUTING.md",
        "mkdocs.yml",
    ],
)
def test_nacca_disclaimer_is_present(name: str) -> None:
    text = " ".join((ROOT / name).read_text(encoding="utf-8").split())
    assert DISCLAIMER in text, f"{name} does not carry the NaCCA disclaimer"


# ---------------------------------------------------------------------------------------------------
# links

FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?\s*$", re.M)
EXTERNAL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")  # http:, https:, mailto: and friends


def _link_targets(text: str) -> list[str]:
    text = HTML_COMMENT.sub("", FENCE.sub("", text))
    text = INLINE_CODE.sub("", text)
    return INLINE_LINK.findall(text) + REFERENCE_LINK.findall(text)


def _slug(heading: str) -> str:
    """The anchor GitHub and mkdocs give a heading: lower case, punctuation dropped, spaces to hyphens."""
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    heading = re.sub(r"[^\w\- ]", "", heading)
    return re.sub(r"\s+", "-", heading).strip("-")


def _anchors(path: Path) -> set[str]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M)}


@pytest.mark.parametrize("path", MARKDOWN_WITH_LINKS, ids=_rel)
def test_relative_links_resolve(path: Path) -> None:
    broken = []
    for target in _link_targets(path.read_text(encoding="utf-8")):
        if EXTERNAL.match(target) or target.startswith("//"):
            continue
        file_part, _, anchor = target.partition("#")
        file_part = file_part.split("?", 1)[0]
        dest = path if not file_part else (path.parent / file_part).resolve()
        if file_part.startswith("/"):
            broken.append(f"{target} (absolute path; use a relative one)")
        elif not dest.exists():
            broken.append(f"{target} (no such file)")
        elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            broken.append(f"{target} (no heading gives the anchor #{anchor})")
    assert not broken, f"broken links in {_rel(path)}:\n" + "\n".join(broken)


def test_docs_pages_link_only_inside_docs() -> None:
    """mkdocs can only resolve links to files under docs/; anything else must be a full web address."""
    outside = []
    for path in _docs_markdown():
        for target in _link_targets(path.read_text(encoding="utf-8")):
            if EXTERNAL.match(target) or target.startswith("#"):
                continue
            dest = (path.parent / target.partition("#")[0]).resolve()
            if DOCS.resolve() not in dest.parents:
                outside.append(f"{_rel(path)} -> {target}")
    assert not outside, "docs pages link outside docs/:\n" + "\n".join(outside)


# ---------------------------------------------------------------------------------------------------
# CITATION.cff

CFF_REQUIRED = ("cff-version", "message", "title", "authors")


def _read_cff_without_yaml(text: str) -> dict:
    """Top-level keys and the list of authors, for the plain shape CITATION.cff uses."""
    data: dict = {}
    key = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        top = re.match(r"^([A-Za-z][\w-]*):\s*(.*)$", line)
        if top:
            key, value = top.group(1), top.group(2).strip()
            data[key] = [] if key in ("authors", "keywords") else value.strip("\"'")
            continue
        item = re.match(r"^\s+-\s+(.*)$", line)
        field = re.match(r"^\s+([A-Za-z][\w-]*):\s*(.*)$", line)
        if key == "authors" and item:
            name, _, value = item.group(1).partition(":")
            data[key].append({name.strip(): value.strip().strip("\"'")})
        elif key == "authors" and field and data[key]:
            data[key][-1][field.group(1)] = field.group(2).strip().strip("\"'")
        elif key == "keywords" and item:
            data[key].append(item.group(1).strip().strip("\"'"))
        elif isinstance(data.get(key), str):  # a folded block such as the abstract
            data[key] = (data[key].lstrip(">|-+ ") + " " + line.strip()).strip()
    return data


def _load_cff() -> dict:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    return yaml.safe_load(text) if yaml is not None else _read_cff_without_yaml(text)


def test_citation_cff_has_the_required_keys() -> None:
    cff = _load_cff()
    assert isinstance(cff, dict)
    missing = [k for k in CFF_REQUIRED if not cff.get(k)]
    assert not missing, f"CITATION.cff lacks {missing}"
    assert str(cff["cff-version"]) == "1.2.0"
    assert isinstance(cff["authors"], list) and cff["authors"]
    for author in cff["authors"]:
        assert author.get("family-names") and author.get("given-names"), f"author needs both names: {author}"
        assert "email" not in author and "orcid" not in author


def test_citation_cff_describes_kwan() -> None:
    cff = _load_cff()
    assert cff["title"].startswith("Kwan")
    assert cff.get("abstract")
    assert cff.get("license") == "MIT"
    assert cff.get("repository-code") == "https://github.com/rofyray/kwan"
    assert cff.get("type", "software") == "software"
    assert str(cff.get("version")) == "1.0.0rc1"
    author = cff["authors"][0]
    assert (author["given-names"], author["family-names"]) == ("Raymond", "Okyere-Forson")


# ---------------------------------------------------------------------------------------------------
# mkdocs

NAV_FILE = re.compile(r"(?:^|[\s:])([\w./-]+\.(?:md|html))\s*$")


def _nav_entries_without_yaml(text: str) -> list[str]:
    entries, in_nav = [], False
    for line in text.splitlines():
        if re.match(r"^nav:\s*$", line):
            in_nav = True
        elif in_nav and re.match(r"^\S", line) and not line.startswith("#"):
            break
        elif in_nav and line.strip() and not line.lstrip().startswith("#"):
            if line.rstrip().endswith(":"):  # a section title; its pages follow on the next lines
                continue
            value = line.split(": ", 1)[1].strip() if ": " in line else line.strip().lstrip("- ").strip()
            if value:
                entries.append(value)
    return entries


def _walk_nav(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [e for item in node for e in _walk_nav(item)]
    if isinstance(node, dict):
        return [e for value in node.values() for e in _walk_nav(value)]
    return []


def _nav_entries() -> list[str]:
    text = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    if yaml is not None:
        return _walk_nav(yaml.safe_load(text)["nav"])
    return _nav_entries_without_yaml(text)


def test_mkdocs_nav_files_exist() -> None:
    entries = _nav_entries()
    files = [e for e in entries if not EXTERNAL.match(e)]
    assert files, "the mkdocs nav lists no pages"
    missing = [e for e in files if not (DOCS / e).is_file()]
    assert not missing, f"mkdocs nav names files that do not exist under docs/: {missing}"


def test_mkdocs_nav_covers_every_docs_page() -> None:
    in_nav = {e for e in _nav_entries() if not EXTERNAL.match(e)}
    on_disk = {p.relative_to(DOCS).as_posix() for p in _docs_markdown()}
    assert on_disk <= in_nav, f"docs pages missing from the mkdocs nav: {sorted(on_disk - in_nav)}"
    assert set(DOC_PAGES) <= in_nav


def test_mkdocs_nav_lists_the_qa_reports_as_real_pages() -> None:
    """The QA reports are written by `kwan curriculum qa` and `kwan graph validate` and committed with the site."""
    qa = [e for e in _nav_entries() if not EXTERNAL.match(e) and e.startswith("qa/")]
    assert sorted(qa) == ["qa/curriculum-math.html", "qa/graph-math.md"], qa
    for name in qa:
        assert (DOCS / name).is_file(), f"docs/{name} is missing: rebuild it with the kwan CLI"
    assert "kwan graph validate" in (DOCS / "qa" / "graph-math.md").read_text(encoding="utf-8")


def _mkdocs_command() -> list[str] | None:
    """How to run mkdocs here: the venv's module, or the binary named by KWAN_MKDOCS, else nothing."""
    if importlib.util.find_spec("mkdocs") is not None and importlib.util.find_spec("material") is not None:
        return [sys.executable, "-m", "mkdocs"]
    binary = os.environ.get("KWAN_MKDOCS")
    if binary and Path(binary).is_file():
        return [binary]
    return None


@pytest.mark.skipif(_mkdocs_command() is None, reason="mkdocs-material is not installed (set KWAN_MKDOCS to a binary)")
def test_mkdocs_build_strict_passes(tmp_path: Path) -> None:
    """The Phase 11 gate: `mkdocs build --strict` with no warning."""
    cmd = _mkdocs_command()
    assert cmd is not None
    r = subprocess.run(
        [*cmd, "build", "--strict", "--site-dir", str(tmp_path / "site")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
        env={**os.environ, "NO_MKDOCS_2_WARNING": "1"},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING" not in r.stdout + r.stderr, r.stdout + r.stderr
    for page in ("evidence", "model-card", "method", "qa/graph-math"):
        assert (tmp_path / "site" / page / "index.html").is_file(), page
    assert (tmp_path / "site" / "qa" / "curriculum-math.html").is_file()
    built = (tmp_path / "site" / "evidence" / "index.html").read_text(encoding="utf-8")
    assert built.count("<table") >= 5, "the evidence page should render its RESULTS tables"


@pytest.mark.skipif(yaml is None, reason="PyYAML is not installed (it comes with the dev extra)")
@pytest.mark.parametrize(
    "name", [".github/ISSUE_TEMPLATE/" + n for n in ISSUE_FORMS] + [".github/workflows/docs.yml", "mkdocs.yml"]
)
def test_yaml_files_parse(name: str) -> None:
    assert isinstance(yaml.safe_load((ROOT / name).read_text(encoding="utf-8")), dict)


@pytest.mark.skipif(yaml is None, reason="PyYAML is not installed (it comes with the dev extra)")
@pytest.mark.parametrize("name", [n for n in ISSUE_FORMS if n != "config.yml"])
def test_issue_forms_have_what_github_needs(name: str) -> None:
    form = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / name).read_text(encoding="utf-8"))
    assert form.get("name") and form.get("description")
    body = form.get("body")
    assert isinstance(body, list) and any(block.get("type") != "markdown" for block in body)
    ids = [block["id"] for block in body if "id" in block]
    assert len(ids) == len(set(ids)), f"{name}: block ids must be unique"


def test_curriculum_error_form_asks_for_the_code_and_the_page() -> None:
    text = (ROOT / ".github" / "ISSUE_TEMPLATE" / "curriculum-error.yml").read_text(encoding="utf-8")
    assert "Indicator code" in text and "PDF page number" in text


def test_docs_workflow_builds_strictly_and_deploys_only_by_hand() -> None:
    text = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "mkdocs build --strict" in text
    assert re.search(r"^\s+push:", text, re.M) and re.search(r"^\s+pull_request:", text, re.M)
    deploy = text[text.index("\n  deploy:") :]
    assert "if: github.event_name == 'workflow_dispatch'" in deploy
    assert "deploy-pages" in deploy and "deploy-pages" not in text[: text.index("\n  deploy:")]


# ---------------------------------------------------------------------------------------------------
# placeholders that a later step fills in

RESULTS_START = re.compile(r"<!-- RESULTS:START id=([\w-]+) source=([\w./-]+) -->")
RESULTS_END = "<!-- RESULTS:END -->"
VERIFY_START = re.compile(r"<!-- VERIFY:START id=([\w-]+) lang=(\w+) -->")
VERIFY_END = "<!-- VERIFY:END -->"


RESULTS_PAGES = ("evidence.md", "model-card.md", "method.md")


@pytest.mark.parametrize("name", RESULTS_PAGES)
def test_results_blocks_are_well_formed(name: str) -> None:
    text = (DOCS / name).read_text(encoding="utf-8")
    ids = [m.group(1) for m in RESULTS_START.finditer(text)]
    assert ids, f"docs/{name} should keep its metrics in RESULTS blocks"
    assert len(ids) == len(set(ids)), "RESULTS block ids must be unique"
    assert text.count("RESULTS:START") == len(ids) == text.count(RESULTS_END)
    for block in re.findall(r"<!-- RESULTS:START[^>]*-->(.*?)<!-- RESULTS:END -->", text, re.S):
        assert "RESULTS:START" not in block, "RESULTS blocks must not nest"


def test_results_block_ids_are_unique_across_the_site() -> None:
    pages = [*_docs_markdown(), ROOT / "README.md"]
    ids = [m.group(1) for p in pages for m in RESULTS_START.finditer(p.read_text(encoding="utf-8"))]
    assert len(ids) == len(set(ids))


def _fill_docs():
    spec = importlib.util.spec_from_file_location("fill_docs", ROOT / "scripts" / "fill_docs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fill_docs_check_passes() -> None:
    """Every RESULTS block in the README and the docs equals what scripts/fill_docs.py renders from the results file."""
    assert _fill_docs().main(["--check"]) == 0, "run `python scripts/fill_docs.py`"


@pytest.mark.parametrize("name", RESULTS_PAGES)
def test_results_blocks_are_filled_where_the_results_exist(name: str) -> None:
    """A block says 'not run yet' only when its top-level key is really missing from the results file."""
    fill_docs = _fill_docs()
    results = fill_docs.load_results()
    text = (DOCS / name).read_text(encoding="utf-8")
    blocks = re.findall(r"<!-- RESULTS:START id=([\w-]+)[^>]*-->(.*?)<!-- RESULTS:END -->", text, re.S)
    assert blocks
    for block_id, body in blocks:
        assert "Results are not filled in yet" not in body, f"{name}: {block_id} still holds the placeholder text"
        assert body.strip(), f"{name}: {block_id} is empty; run scripts/fill_docs.py"
        if fill_docs.NOT_RUN in body:
            key = re.search(r"`([\w.]+)` is not in", body)
            assert key, f"{name}: {block_id} is not run and names no key:\n{body}"
            top = key.group(1).split(".")[0]
            assert top not in results, f"{name}: {block_id} says not run, but `{top}` is in the results file"


def test_every_results_block_has_a_renderer() -> None:
    fill_docs = _fill_docs()
    for name in fill_docs.FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        unknown = [b for b in fill_docs.block_ids(text) if b not in fill_docs.RENDERERS]
        assert not unknown, f"{name}: no renderer for {unknown}"


def test_quickstart_code_blocks_are_marked_for_verification() -> None:
    text = (DOCS / "quickstart.md").read_text(encoding="utf-8")
    ids = [m.group(1) for m in VERIFY_START.finditer(text)]
    assert len(ids) == len(set(ids)) and len(ids) == text.count(VERIFY_END)
    marked = re.findall(r"<!-- VERIFY:START[^>]*-->\n```(\w+)\n.*?\n```\n<!-- VERIFY:END -->", text, re.S)
    assert len(marked) == len(ids), "each VERIFY block wraps exactly one fenced code block"
    assert len(FENCE.findall(text)) == len(ids), "every code block in the quickstart sits inside a VERIFY block"
    for call in ("Kwan.from_bundle(", "kwan.update(", "kwan.recommend(", "kwan.mastery_map("):
        assert call in text, f"the quickstart should show {call}"
    for route in ("/v1/attempts", "/v1/learners/{id}/recommendations", "/v1/learners/{id}/mastery", "/v1/health"):
        assert route in text, f"the quickstart should list {route}"


# ---------------------------------------------------------------------------------------------------
# the context style guide keeps its own rules

GUIDE = DOCS / "context-style-guide.md"
REGIONS = (
    "Ahafo",
    "Ashanti",
    "Bono",
    "Bono East",
    "Central",
    "Eastern",
    "Greater Accra",
    "North East",
    "Northern",
    "Oti",
    "Savannah",
    "Upper East",
    "Upper West",
    "Volta",
    "Western",
    "Western North",
)
COINS = ("1 pesewa", "5 pesewas", "10 pesewas", "20 pesewas", "50 pesewas", "GH₵ 1", "GH₵ 2")
NOTES = ("GH₵ 1", "GH₵ 2", "GH₵ 5", "GH₵ 10", "GH₵ 20", "GH₵ 50", "GH₵ 100", "GH₵ 200")
MAX_SETTING_CHARS = 220


def _examples(kind: str) -> list[str]:
    return re.findall(rf"^> \*\*{kind} \d[^*]*\*\* (.+)$", GUIDE.read_text(encoding="utf-8"), re.M)


def _table_row(label: str) -> list[str]:
    row = re.search(rf"^\| {label} \| (.+?) \|$", GUIDE.read_text(encoding="utf-8"), re.M)
    assert row, f"the style guide needs a '{label}' row"
    return [cell.strip() for cell in row.group(1).split(",")]


def test_style_guide_lists_all_16_regions() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([A-Z][A-Za-z ]+?) \| [^|]+ \| [^|]+ \|$", text, re.M)
    listed = [r for r in rows if r in REGIONS]
    assert sorted(listed) == sorted(REGIONS), f"regions missing or repeated: {sorted(set(REGIONS) ^ set(listed))}"


def test_style_guide_lists_the_coins_and_notes_exactly() -> None:
    assert tuple(_table_row("Coins")) == COINS
    assert tuple(_table_row("Notes")) == NOTES


def test_style_guide_has_six_good_and_six_bad_examples() -> None:
    good, bad = _examples("Good"), _examples("Bad")
    assert len(good) == 6 and len(bad) == 6
    too_long = [g for g in good if len(g) > MAX_SETTING_CHARS]
    assert not too_long, f"good examples must be {MAX_SETTING_CHARS} characters or fewer: {too_long}"
    for example in good:
        assert example.rstrip().endswith("?"), f"a good example ends with its question: {example}"
        for amount in re.findall(r"GH\S*\s?[\d.,]+", example):
            assert re.fullmatch(r"GH₵ \d{1,3}(,\d{3})*(\.\d{2})?[.,]?", amount), (
                f"money is written like GH₵ 7.50: {amount}"
            )


def test_style_guide_names_are_balanced_between_girls_and_boys() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    table = text[text.index("| Tradition | Girls | Boys |") :].split("\n\n", 1)[0]
    for line in table.splitlines()[2:]:
        tradition, girls, boys = (cell.strip() for cell in line.strip("|").split("|"))
        if girls.startswith("*"):  # a row still waiting for a reviewer from that group
            continue
        n_girls, n_boys = len(girls.split(",")), len(boys.split(","))
        assert n_girls == n_boys, f"{tradition}: {n_girls} girls' names and {n_boys} boys' names"
        assert not set(girls.split(", ")) & set(boys.split(", ")), f"{tradition}: a name sits in both columns"
