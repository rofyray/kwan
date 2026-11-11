"""Document-family configuration (`families.yaml`): everything that differs between curriculum documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kwan._paths import repo_root

DEFAULT_FILE = Path(__file__).with_name("families.yaml")
DEFAULT_FAMILY = "nacca-math-primary"


@dataclass
class Document:
    """One PDF of a family."""

    id: str
    url: str
    file: str
    grades: list[int] = field(default_factory=list)
    local_copies: list[str] = field(default_factory=list)
    text_copies: list[str] = field(default_factory=list)


@dataclass
class Family:
    """One document family, with its patterns compiled."""

    name: str
    subject: str
    title: str
    grades: list[int]
    strands: dict[int, str]
    code_prefix: str
    code_re: re.Pattern[str]
    short_code_re: re.Pattern[str]
    grade_heading_re: re.Pattern[str]
    end_heading_re: re.Pattern[str]
    strand_heading_re: re.Pattern[str]
    sub_strand_heading_re: re.Pattern[str]
    header_anchor_re: re.Pattern[str]
    header_words: frozenset[str]
    footer_re: re.Pattern[str]
    exemplar_re: re.Pattern[str]
    competency_words: list[str]
    columns: dict[str, tuple[float, float]]
    snap_to_rulings: bool
    documents: list[Document]
    paths: dict[str, str]

    def code(self, m: re.Match[str]) -> str:
        """The normalised five-part code of a `code_re` match, for example `B4.1.1.1.5`."""
        g = m.groupdict()
        return f"{self.code_prefix}{g['grade']}.{g['strand']}.{g['sub_strand']}.{g['cs']}.{int(g['no'])}"

    def short_code(self, m: re.Match[str]) -> str:
        """The normalised four-part code of a `short_code_re` match, for example `B4.1.1.1`."""
        g = m.groupdict()
        return f"{self.code_prefix}{g['grade']}.{g['strand']}.{g['sub_strand']}.{g['cs']}"

    def split_code(self, code: str) -> tuple[int, int, int, int, int]:
        """`B4.1.2.3.10` -> (4, 1, 2, 3, 10)."""
        parts = code[len(self.code_prefix) :].split(".")
        if len(parts) != 5 or not all(p.isdigit() for p in parts):
            raise ValueError(f"not a five-part indicator code: {code!r}")
        g, s, ss, cs, no = (int(p) for p in parts)
        return g, s, ss, cs, no

    def path(self, key: str, root: Path | None = None) -> Path:
        """Absolute path of one of the family's files (`structure_file`, `full_file`, `corrections`, ...)."""
        base = root or repo_root() or Path.cwd()
        return base / self.paths[key]


DEFAULT_EXEMPLAR_MARKER = (
    r"^\s*(?:E\s?\.?\s?g\b\s?[.,:]?|Example(?=\s*\d))\s*\.?\s*(?:(?P<label>\d{1,2})(?!\d|,\d))?\s*[.:),]?\s*"
)

PATH_KEYS = (
    "corrections",
    "expected_counts",
    "label_drafts",
    "label_reviews",
    "structure_file",
    "full_file",
    "committed_full_file",
    "qa_report",
)


def family_from_dict(name: str, cfg: dict[str, Any]) -> Family:
    """Build a `Family` from one mapping of `families.yaml` (also used by tests with made-up families)."""
    flags = re.IGNORECASE
    subject = cfg["subject"]
    defaults = {
        "corrections": f"data/curriculum/{subject}.corrections.json",
        "expected_counts": f"data/curriculum/{subject}.expected_counts.json",
        "label_drafts": f"data/contexts/drafts/{subject}.*.json",
        "label_reviews": f"data/contexts/{subject}.label_reviews.json",
        "structure_file": f"data/curriculum/{subject}.structure.json",
        "full_file": f"build/curriculum/{subject}.full.json",
        "committed_full_file": f"data/curriculum/{subject}.full.json",
        "qa_report": f"docs/qa/curriculum-{subject}.html",
    }
    return Family(
        name=name,
        subject=subject,
        title=cfg.get("title", name),
        grades=[int(g) for g in cfg["grades"]],
        strands={int(k): str(v) for k, v in cfg["strands"].items()},
        code_prefix=cfg.get("code_prefix", "B"),
        code_re=re.compile(cfg["code_pattern"]),
        short_code_re=re.compile(cfg.get("short_code_pattern", r"(?!x)x")),
        grade_heading_re=re.compile(cfg["grade_heading"], flags),
        end_heading_re=re.compile(cfg.get("end_heading", r"(?!x)x")),
        strand_heading_re=re.compile(
            cfg.get("strand_heading", r"^\s*Strand\s*(?P<no>\d)\s*[:.\-]\s*(?P<name>.+)$"), flags
        ),
        sub_strand_heading_re=re.compile(
            cfg.get("sub_strand_heading", r"^\s*Sub-?\s?strand\s*(?P<no>\d)\s*:\s*(?P<name>.+)$"), flags
        ),
        header_anchor_re=re.compile(cfg.get("header_anchor", r"INDICATORS\s+AND\s+EXEMPL"), flags),
        header_words=frozenset(w.upper() for w in cfg.get("header_words", [])),
        footer_re=re.compile(cfg.get("footer_pattern", r"(?!x)x")),
        exemplar_re=re.compile(cfg.get("exemplar_marker", DEFAULT_EXEMPLAR_MARKER)),
        competency_words=list(cfg.get("competency_words", [])),
        columns={k: (float(v[0]), float(v[1])) for k, v in cfg.get("columns", {}).items()},
        snap_to_rulings=bool(cfg.get("snap_to_rulings", True)),
        documents=[
            Document(
                id=d["id"],
                url=d.get("url", ""),
                file=d.get("file", f"{d['id']}.pdf"),
                grades=[int(g) for g in d.get("grades", [])],
                local_copies=list(d.get("local_copies", [])),
                text_copies=list(d.get("text_copies", [])),
            )
            for d in cfg.get("documents", [])
        ],
        paths={k: cfg.get(k, defaults[k]) for k in PATH_KEYS},
    )


def load_families(path: Path | str | None = None) -> dict[str, Family]:
    """All families in a `families.yaml` file (needs PyYAML)."""
    import yaml

    with open(path or DEFAULT_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {name: family_from_dict(name, cfg) for name, cfg in data["families"].items()}


def load_family(name: str | None = None, path: Path | str | None = None) -> Family:
    """One family by name (the mathematics family by default)."""
    families = load_families(path)
    key = name or DEFAULT_FAMILY
    if key not in families:
        raise KeyError(f"unknown document family {key!r}; known: {', '.join(sorted(families))}")
    return families[key]
