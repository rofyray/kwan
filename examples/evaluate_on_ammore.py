"""Evaluate Kwan's learner model on AMMORE, the Rori WhatsApp maths tutor dataset (Ghana among others).

Run:       python examples/evaluate_on_ammore.py [--raw raw/ammore] [--out results.json]

AMMORE is CC BY-NC 4.0: evaluation only. Parameters fitted on it never ship in a Kwan bundle. The
dataset is not part of the package; this script runs when the files are on disk and otherwise says
how to get them. It needs the `eval` extra (`pip install 'kwan[eval]'`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NEEDED = ("AMMORE_student_answers.csv",)
URL = "https://github.com/owenhenkel/ammore_dataset"


def how_to_get(raw: Path) -> str:
    return "\n".join(
        [
            f"AMMORE is not at {raw} (looking for {', '.join(NEEDED)}).",
            "",
            "To get it (about 13 MB, CC BY-NC 4.0, evaluation only):",
            f"    git clone {URL} {raw}",
            "or, from a source checkout of Kwan:",
            "    bash scripts/download_data.sh ammore",
            "",
            "Then run this script again, or `kwan eval ammore`.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate Kwan's learner model on AMMORE (all countries and Ghana only).")
    ap.add_argument("--raw", default="raw/ammore", help="directory holding the AMMORE CSV files (default: raw/ammore)")
    ap.add_argument("--out", metavar="PATH", help="also write the `ammore_2024` block to this results file")
    args = ap.parse_args(argv)
    raw = Path(args.raw)
    if not all((raw / name).is_file() for name in NEEDED):
        print(how_to_get(raw))
        return 0
    try:
        from kwan.eval import ammore
    except ModuleNotFoundError as exc:
        print(f"the package {exc.name!r} is not installed; run: pip install 'kwan[eval]'", file=sys.stderr)
        return 1
    ammore.main([str(raw), *(["--out", args.out] if args.out else [])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
