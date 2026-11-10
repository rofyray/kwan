"""Single writer for data/eval_results.json.

Every number quoted in the README, the demo or a deck must come from that file, and every block in it
must be written by a script in this repo. Each script owns one top-level key and calls `update()`,
which merges that key into the file and leaves the other blocks untouched.
"""
import json, os, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PATH = ROOT / "data" / "eval_results.json"

def load(path=DEFAULT_PATH) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def update(key: str, block: dict, path=DEFAULT_PATH) -> dict:
    """Replace one top-level block and write the file atomically. Returns the full results dict."""
    path = Path(path)
    res = load(path)
    res[key] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return res
