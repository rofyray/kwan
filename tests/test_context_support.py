"""Synthetic practice settings shared by the context tests, and a check that they are sound.

Nothing here is curriculum text or a committed draft: the settings below were written for the tests.
The other `test_context_*.py` files import `make_world` to get a small, self-contained data folder.
"""

import json
from pathlib import Path

from kwan.context import checks
from kwan.context.drafts import DraftSet
from kwan.context.settings import canonical_id, save_contexts, text_hash

MODEL = "draft:test-model"

# bare code -> (short label, [settings]); every setting passes the automatic checks for its class
SYNTHETIC = {
    "B1.1.1.1.1": (
        "Count objects up to 20",
        [
            "Ama has 5 oranges at Makola Market. She gets 2 more. How many oranges does she have now?",
            "Kofi sees 3 goats and 4 hens in Tamale. How many animals does he see?",
            "Count the bottle tops Esi has in Ho. She has 6 red and 3 blue. How many in all?",
        ],
    ),
    "B2.1.2.1.1": (
        "Add two numbers within 100",
        [
            "Yaw sells 12 balls of kenkey in Tema. Then he sells 20 more. How many does he sell?",
            "A trotro in Kumasi has 14 people. 5 more get in. How many people are in it now?",
        ],
    ),
    "B3.1.4.1.1": (
        "Work out change from cedi notes",
        [
            "Fati buys waakye for GH₵ 15 in Wa. She pays with a GH₵ 20 note. How much change does she get?",
            "Kwame has two GH₵ 5 notes and one GH₵ 2 coin in Keta. How much money does he have?",
            "Abla pays GH₵ 8 for koko in Hohoe. She gives a GH₵ 10 note. What is her change?",
        ],
    ),
    "B4.1.2.1.1": (
        "Recall times tables up to 12",
        [
            "A crate holds 12 bottles of sobolo. Dede in Nungua has 6 full crates. How many bottles does she have?",
            "Musah plants 9 rows of yam near Yendi with 8 in each row. How many yams does he plant?",
            "Each oware board has 12 houses. How many houses are on 4 boards at a club in Sunyani?",
        ],
    ),
    "B5.1.3.1.1": (
        "Add fractions with the same denominator",
        [
            "Efua eats 2/8 of a pineapple in Cape Coast and Kojo eats 3/8. What fraction do they eat in all?",
            "A tank in Bole is 1/5 full. Rain adds 2/5 more. What fraction of the tank is full now?",
        ],
    ),
    "B6.1.4.1.1": (
        "Write a ratio in its simplest form",
        [
            "A class in Axim has 18 girls and 12 boys. Write the ratio of girls to boys in its simplest form.",
            "Senyo mixes 2 cups of gari with 6 cups of water in Aflao. Write the ratio of gari to water.",
            "A farm near Goaso has 10 cocoa trees for every 4 orange trees. Write this ratio in its simplest form.",
        ],
    ),
}


def draft_set(data=None) -> DraftSet:
    """A `DraftSet` built from the synthetic data (or from `data` in the same shape)."""
    data = SYNTHETIC if data is None else data
    ds = DraftSet(subject="math", mapping="corrections")
    for code, (label, settings) in data.items():
        ds.items[canonical_id(code)] = {
            "short_label": label,
            "settings": list(settings),
            "source": MODEL,
            "key": code,
            "file": "math.test.json",
        }
    return ds


def contexts_doc(data=None, reviewed=()) -> dict:
    """A settings document with every synthetic setting as an unreviewed draft.

    `reviewed` lists texts to mark as already reviewed (by reviewer "T1" on 2026-01-05).
    """
    data = SYNTHETIC if data is None else data
    doc = {"schema_version": 1, "subject": "math", "settings": {}, "rejected": {}, "rejected_labels": {}}
    for code, (_, settings) in data.items():
        doc["settings"][canonical_id(code)] = [
            {
                "text": text,
                "lang": "en",
                "reviewed": text in reviewed,
                "source": MODEL,
                "reviewer": "T1" if text in reviewed else None,
                "reviewed_at": "2026-01-05" if text in reviewed else None,
            }
            for text in settings
        ]
    return doc


class World:
    """Paths of a small data folder under a temporary directory."""

    def __init__(self, root: Path, data=None, reviewed=()):
        self.root = root
        self.data = root / "data"
        self.contexts = self.data / "contexts" / "math.json"
        self.labels = self.data / "contexts" / "label_reviews.json"
        self.translations = self.data / "contexts" / "translations"
        self.out = root / "build" / "review"
        self.drafts = draft_set(data)
        save_contexts(contexts_doc(data, reviewed), self.contexts)

    def doc(self) -> dict:
        return json.loads(self.contexts.read_text(encoding="utf-8"))

    def label_reviews(self) -> dict:
        return json.loads(self.labels.read_text(encoding="utf-8")) if self.labels.exists() else {}

    def tdoc(self, lang: str) -> dict:
        return json.loads((self.translations / f"{lang}.json").read_text(encoding="utf-8"))

    def entry(self, code: str, text: str) -> dict | None:
        return next((e for e in self.doc()["settings"].get(canonical_id(code), []) if e["text"] == text), None)

    def kwargs(self) -> dict:
        """Keyword arguments that point `sheets.export_review` / `import_review` at this folder."""
        return {
            "contexts": self.contexts,
            "drafts": self.drafts,
            "label_reviews": self.labels,
            "translations": self.translations,
        }


def make_world(tmp_path: Path, data=None, reviewed=()) -> World:
    """Create the folder and return its paths."""
    return World(tmp_path, data, reviewed)


# ----------------------------------------------------------------------------- the fixtures are sound


def test_every_synthetic_setting_passes_the_automatic_checks():
    for code, (label, settings) in SYNTHETIC.items():
        assert not checks.check_label(label), (code, label)
        for i, text in enumerate(settings):
            failures = checks.check_setting(text, checks.grade_of(code), siblings=settings[:i])
            assert not failures, (code, text, [str(f) for f in failures])


def test_synthetic_settings_cover_all_six_classes():
    assert sorted({checks.grade_of(code) for code in SYNTHETIC}) == [1, 2, 3, 4, 5, 6]
    assert sum(len(s) for _, s in SYNTHETIC.values()) == 16


def test_make_world_writes_unreviewed_drafts(tmp_path):
    world = make_world(tmp_path)
    doc = world.doc()
    assert len(doc["settings"]) == len(SYNTHETIC)
    assert all(not e["reviewed"] and e["reviewer"] is None for entries in doc["settings"].values() for e in entries)
    first = SYNTHETIC["B1.1.1.1.1"][1][0]
    assert world.entry("B1.1.1.1.1", first)["source"] == MODEL
    assert len(text_hash(first)) == 16
