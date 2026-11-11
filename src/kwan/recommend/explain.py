"""Reasons as data: every recommendation carries `rationale_parts`, and the text is rendered from them.

A part is `{"key": "builds_on", "params": {"prerequisite": "B1.1.1.1.2", "pct": 95}}`. The templates
are a flat mapping from key to text with `{named}` fields, one file per language in
`data/i18n/<lang>.json`. A part whose wording depends on a number comes as a pair of templates,
`<key>_one` and `<key>_other`, chosen by the part's `count` (`_one` when the count is exactly 1). A
language with no such difference can give the plain `<key>` instead, which wins when present.

`EN` is the built-in copy of the English templates, so the package works when `data/` is not on disk
(a wheel install). A test keeps it equal to `data/i18n/en.json`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

EN: dict[str, str] = {
    "builds_on": "Builds on {prerequisite}, where you are at {pct}%.",
    "starting_skill": "A starting skill with nothing before it.",
    "mastery_new": "Still new to you ({pct}%), so expect a few tries.",
    "mastery_close": "You are at {pct}%, close enough to master with practice.",
    "mastery_nearly": "Nearly there at {pct}%. A few correct answers will lock it in.",
    "goal_steps_one": "On the way to your goal, {count} step away.",
    "goal_steps_other": "On the way to your goal, {count} steps away.",
    "unlocks_one": "Opens the way to {count} later skill.",
    "unlocks_other": "Opens the way to {count} later skills.",
    "review": "You learned this before. A quick check keeps it fresh.",
}
DEFAULT_LANG = "en"
PLURAL_KEYS = ("goal_steps", "unlocks")  # keys that come as `_one` / `_other` pairs

_FIELD = re.compile(r"\{(\w+)\}")


def part(key: str, **params: Any) -> dict[str, Any]:
    """One rationale part. `params` hold only text and whole numbers, so a part survives JSON."""
    return {"key": key, "params": params}


def template_for(key: str, params: Mapping[str, Any], templates: Mapping[str, str]) -> str:
    """The template for one part: the plain key, else the `_one` / `_other` form chosen by `count`."""
    if key in templates:
        return templates[key]
    if "count" in params:
        form = f"{key}_one" if params["count"] == 1 else f"{key}_other"
        if form in templates:
            return templates[form]
    raise KeyError(f"no template for rationale key {key!r}")


def render_part(p: Mapping[str, Any], templates: Mapping[str, str]) -> str:
    """The sentence for one part. Every `{name}` in the template is replaced by `params[name]`."""
    params: Mapping[str, Any] = p.get("params") or {}
    key = str(p["key"])

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            raise KeyError(f"rationale key {key!r} needs the field {name!r}")
        return str(params[name])

    return _FIELD.sub(fill, template_for(key, params, templates))


def render(parts: Iterable[Mapping[str, Any]], templates: Mapping[str, str] | None = None) -> str:
    """The reason as one line of text: the rendered parts joined by single spaces.

    `templates` defaults to the built-in English ones."""
    t = EN if templates is None else templates
    return " ".join(render_part(p, t) for p in parts)


def resolve_templates(i18n: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Templates per language from what the caller passed to the recommender.

    Accepts None (built-in English), a mapping of language to templates (a bundle's `i18n` block) or
    one flat set of templates (taken as English). Every language is filled up with the built-in
    English templates, so a missing key falls back to English instead of failing.
    """
    out: dict[str, dict[str, str]] = {DEFAULT_LANG: dict(EN)}
    if not i18n:
        return out
    by_lang: Mapping[str, Any] = {DEFAULT_LANG: i18n} if all(isinstance(v, str) for v in i18n.values()) else i18n
    english = {**EN, **{str(k): str(v) for k, v in (by_lang.get(DEFAULT_LANG) or {}).items()}}
    out[DEFAULT_LANG] = english
    for lang, table in by_lang.items():
        if lang != DEFAULT_LANG:
            out[str(lang)] = {**english, **{str(k): str(v) for k, v in (table or {}).items()}}
    return out


def load_templates(path: str | Path) -> dict[str, str]:
    """Read one templates file such as `data/i18n/en.json`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ValueError(f"{path}: expected a flat JSON object of key to template text")
    return {str(k): v for k, v in data.items()}


__all__ = [
    "DEFAULT_LANG",
    "EN",
    "PLURAL_KEYS",
    "load_templates",
    "part",
    "render",
    "render_part",
    "resolve_templates",
    "template_for",
]
