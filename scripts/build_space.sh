#!/usr/bin/env bash
# Assemble the Hugging Face Space bundle for the Kwan demo (PRD Phase 8.2) into build/space/ and check that it
# runs on its own. Nothing is uploaded: this script only prepares the folder and tells you how to push it.
# Safe to re-run: build/space/ is rebuilt from scratch every time.
#   PY=/path/to/python bash scripts/build_space.sh     # choose the interpreter (default: .venv/bin/python, else python3)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/build/space"
SPACE_README="$ROOT/deploy/space/README.md"

# The file set of PRD Phase 8.2, plus every other module app.py imports (results.py reads data/eval_results.json).
FILES=(
  app.py
  graph.py
  engine.py
  context.py
  simulate.py
  results.py
  requirements.txt
  data/graph.json
  data/indicators.json
  data/eval_results.json
)

say()  { echo "[build_space] $*"; }
fail() { echo "[build_space] ERROR: $*" >&2; exit 1; }

if [ -z "${PY:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="python3"; fi
fi
command -v "$PY" >/dev/null 2>&1 || fail "Python interpreter '$PY' not found. Set PY=/path/to/python."
"$PY" -c "import streamlit" 2>/dev/null || fail "'$PY' has no streamlit. Run: pip install -r requirements.txt (or set PY)."

[ -f "$SPACE_README" ] || fail "missing $SPACE_README"
for f in "${FILES[@]}"; do
  [ -f "$ROOT/$f" ] || fail "missing $f (data files are built by: make curriculum graph eval sim)"
done
head -n 1 "$SPACE_README" | grep -qx -- "---" || fail "deploy/space/README.md must start with the Space front matter (---)"

say "assembling $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  mkdir -p "$OUT/$(dirname "$f")"
  cp "$ROOT/$f" "$OUT/$f"
done
cp "$SPACE_README" "$OUT/README.md"

say "checking that the bundle runs on its own"
# Run the app headless from inside the bundle. PYTHONPATH is cleared so nothing can be imported from the repo by
# accident, and the check fails if any module or data file was loaded from outside the bundle.
(
  cd "$OUT"
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" - "$ROOT" <<'PYCHECK'
import sys
from pathlib import Path

repo, here = Path(sys.argv[1]).resolve(), Path.cwd().resolve()

import logging
import streamlit
from streamlit.testing.v1 import AppTest

# AppTest has no browser session, and Streamlit says so once in a log line that only reads like a problem
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

at = AppTest.from_file(str(here / "app.py"), default_timeout=120).run()
problems = [str(e.value) for e in at.exception]
tabs = [t.label for t in at.tabs]
if tabs != ["Pupil view", "Curriculum map", "Evidence"]:
    problems.append(f"unexpected tabs: {tabs}")
cards = [m for m in at.markdown if m.value.startswith("**Why this skill:**")]
if len(cards) != 3:
    problems.append(f"expected 3 recommendation cards, found {len(cards)}")
if len(at.dataframe) < 3:
    problems.append(f"expected the mastery, curriculum and evidence tables, found {len(at.dataframe)} tables")

# every repo module the app pulled in must be the bundled copy
for name, mod in sorted(sys.modules.items()):
    f = getattr(mod, "__file__", None)
    if not f:
        continue
    f = Path(f).resolve()
    if repo in f.parents and here not in f.parents and "site-packages" not in f.parts and ".venv" not in f.parts:
        problems.append(f"module '{name}' was loaded from outside the bundle: {f}")
for name in ("graph", "engine", "context", "simulate", "results"):
    mod = sys.modules.get(name)
    if mod is None or Path(mod.__file__).resolve().parent != here:
        problems.append(f"module '{name}' was not loaded from the bundle")
for name in ("graph", "results"):
    path = getattr(sys.modules.get(name), "DEFAULT_PATH", None)
    if path is not None and Path(path).resolve().parent != here / "data":
        problems.append(f"{name}.py reads its data file from outside the bundle: {path}")

if problems:
    print("\n".join("  - " + p for p in problems), file=sys.stderr)
    sys.exit(1)

sdk = next((line.split(":", 1)[1].strip() for line in (here / "README.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("sdk_version:")), "not set")
print(f"[build_space] app ran from the bundle: 3 tabs, 3 cards, {len(at.dataframe)} tables, no exception "
      f"(Streamlit {streamlit.__version__} on Python {sys.version_info.major}.{sys.version_info.minor})")
if sdk != streamlit.__version__:
    print(f"[build_space] note: the Space README pins sdk_version {sdk}, this check ran on Streamlit {streamlit.__version__}")
PYCHECK
) || fail "the bundle does not run on its own (see above). Nothing was published."

find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +

say "bundle ready: $OUT"
(cd "$OUT" && find . -type f | sed 's|^\./|    |' | LC_ALL=C sort)
cat <<EOF

Nothing has been uploaded. To publish the bundle to a Hugging Face Space yourself:

  1. Create an empty Space named kwan at https://huggingface.co/new-space (hardware: CPU basic, public).
     The README.md in the bundle is what sets sdk: streamlit and app_file: app.py.
  2. hf auth login
  3. hf upload <your-username>/kwan "$OUT" . --repo-type=space

(hf comes with: pip install -U huggingface_hub. On older installs the same two commands are
 huggingface-cli login and huggingface-cli upload.)
Then open https://huggingface.co/spaces/<your-username>/kwan and check that the three tabs load.
EOF
