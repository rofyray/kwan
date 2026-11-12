"""The Kwan logo: where each version lives, that the copies agree, and that the old compass emoji is gone."""

import subprocess
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_ASSETS = ROOT / "docs" / "assets"
PACKAGE_ASSETS = ROOT / "src" / "kwan" / "assets"
SHARED = ("kwan-logo.png", "kwan-logo-light.png", "kwan-logo-dark.png", "kwan-mark.png")
PNG = b"\x89PNG\r\n\x1a\n"
COMPASS = chr(0x1F9ED)


def test_every_logo_file_is_a_small_png():
    names = {*SHARED, "kwan-mark-light.png", "kwan-mark-dark.png", "favicon.png"}
    for name in names:
        data = (DOCS_ASSETS / name).read_bytes()
        assert data.startswith(PNG) and len(data) < 60_000, name


def test_the_package_ships_the_same_logos_as_the_docs():
    for name in SHARED:
        assert (PACKAGE_ASSETS / name).read_bytes() == (DOCS_ASSETS / name).read_bytes(), name
        assert (files("kwan") / "assets" / name).is_file()


def test_readme_docs_and_demos_use_the_logo():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "kwan-logo-dark.png" in readme and "kwan-logo-light.png" in readme
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "logo: assets/kwan-mark.png" in mkdocs and "favicon: assets/favicon.png" in mkdocs
    index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    assert "kwan-logo-light.png#only-light" in index and "kwan-logo-dark.png#only-dark" in index
    assert "page_icon=MARK" in (ROOT / "demo" / "app.py").read_text(encoding="utf-8")
    assert "favicon_path=MARK" in (ROOT / "demo" / "gradio_app.py").read_text(encoding="utf-8")


def test_no_committed_file_uses_the_compass_emoji():
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    offenders = []
    for name in listed:
        path = ROOT / name
        if path.suffix in {".png", ".jpg", ".pdf"} or not path.is_file() or name == "tests/test_brand.py":
            continue
        if COMPASS in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(name)
    assert not offenders, offenders
