"""Command-line entries for evaluation: `kwan eval <dataset>`, `kwan eval all`, `kwan sim` and `kwan bench`.

Nothing heavy is imported here. Each handler imports its module when the command runs, so
`kwan sim` and `kwan bench` work without pandas or scikit-learn.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

OUT_HELP = "merge this command's block into this results file, e.g. data/eval_results.json"
FOLDS_HELP = "also run K-fold cross-validation by student (5 matches the pyKT set-up) and add it as `cv`"


def _out(args: argparse.Namespace) -> list[str]:
    return ["--out", args.out] if args.out else []


RAW = "raw"  # where `make data` puts the datasets, relative to the working directory

# name after `kwan eval` -> (help line, what the optional path is)
NEW_DATASETS = {
    "eedi-12": (
        "Eedi NeurIPS 2020 Education Challenge, task 1/2 (15.9M answers)",
        "directory holding train_data/ and metadata/ (default: raw/eedi/data)",
    ),
    "assistments-2012": (
        "ASSISTments 2012-2013",
        "the CSV or the directory holding it (default: raw/assist2012)",
    ),
    "assistments-2015": (
        "ASSISTments 2015 (100 skill builders)",
        "the CSV or the directory holding it (default: raw/assist2015)",
    ),
    "assistments-2017": (
        "ASSISTments 2017 (Longitudinal Data Mining Competition)",
        "the CSV or the directory holding it (default: raw/assist2017)",
    ),
    "xes3g5m": (
        "XES3G5M (Grade 3 maths, China)",
        "directory holding the unpacked XES3G5M folder (default: raw/xes3g5m)",
    ),
    "junyi": (
        "Junyi Academy 2015 problem log (non-commercial data: evaluation only)",
        "directory holding junyi_ProblemLog_original.csv (default: raw/junyi)",
    ),
}


def _assistments(args: argparse.Namespace) -> int:
    from kwan.eval import kt

    kt.main([args.csv or f"{RAW}/skill_builder_data_corrected.csv", *(["--bkt"] if args.bkt else []), *_out(args)])
    return 0


def _eedi(args: argparse.Namespace) -> int:
    from kwan.eval import eedi

    eedi.main([args.root or f"{RAW}/eedi/data", *_out(args)])
    return 0


def _ammore(args: argparse.Namespace) -> int:
    from kwan.eval import ammore

    ammore.main([args.root or f"{RAW}/ammore", *_out(args)])
    return 0


def _dataset(args: argparse.Namespace) -> int:
    from kwan.eval import harness

    try:
        harness.run_dataset(args.dataset, args.path, args.out, folds=args.folds)
    except FileNotFoundError as exc:
        print(f"kwan eval {exc}", file=sys.stderr)
        return 1
    return 0


def _all(args: argparse.Namespace) -> int:
    from kwan.eval import harness

    only = [name.strip() for name in args.only.split(",") if name.strip()] if args.only else None
    try:
        blocks = harness.run_all(args.raw, args.out, only=only, bkt=False if args.no_bkt else None, folds=args.folds)
    except KeyError as exc:
        print(f"kwan eval all: {exc.args[0]}", file=sys.stderr)
        return 2
    done = [key for key, block in blocks.items() if "skipped" not in block]
    skipped = [key for key in blocks if key not in done]
    print(f"evaluated {len(done)} dataset(s): {', '.join(done) or 'none'}", file=sys.stderr)
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped)}", file=sys.stderr)
    return 0


def _sim(args: argparse.Namespace) -> int:
    from kwan.eval import simulate

    argv = [*([args.graph] if args.graph else []), *_out(args)]
    if args.v2:
        argv.append("--v2")
    if args.learners is not None:
        argv += ["--learners", str(args.learners)]
    simulate.main(argv)
    return 0


def _bench(args: argparse.Namespace) -> int:
    from kwan.eval import latency

    argv = [*([args.graph] if args.graph else []), *_out(args)]
    if args.calls is not None:
        argv += ["--calls", str(args.calls)]
    if args.hardware is not None:
        argv += ["--hardware", args.hardware]
    latency.main(argv)
    return 0


def register(subparsers: Any) -> None:
    """Add `eval`, `sim` and `bench` to the `kwan` command."""
    ev = subparsers.add_parser("eval", help="validate the learner model on a public dataset")
    datasets = ev.add_subparsers(dest="dataset", metavar="DATASET", required=True)

    from kwan.eval import deep

    deep.register(datasets)

    p = datasets.add_parser("assistments-2009", help="ASSISTments 2009 skill-builder (corrected)")
    p.add_argument(
        "csv",
        nargs="?",
        help="path to skill_builder_data_corrected.csv (default: raw/skill_builder_data_corrected.csv)",
    )
    p.add_argument("--bkt", action="store_true", help="also fit the BKT baseline (grid search, a few minutes)")
    p.add_argument("--out", metavar="PATH", help=OUT_HELP)
    p.set_defaults(func=_assistments, extra="eval")

    p = datasets.add_parser("eedi-34", help="Eedi NeurIPS 2020 Education Challenge, task 3/4 subset")
    p.add_argument("root", nargs="?", help="directory holding train_data/ and metadata/ (default: raw/eedi/data)")
    p.add_argument("--out", metavar="PATH", help=OUT_HELP)
    p.set_defaults(func=_eedi, extra="eval")

    p = datasets.add_parser("ammore", help="AMMORE (Rori, Rising Academies): all countries and Ghana only")
    p.add_argument("root", nargs="?", help="directory holding AMMORE_student_answers.csv (default: raw/ammore)")
    p.add_argument("--out", metavar="PATH", help=OUT_HELP)
    p.set_defaults(func=_ammore, extra="eval")

    for name, (title, path_help) in NEW_DATASETS.items():
        p = datasets.add_parser(name, help=title)
        p.add_argument("path", nargs="?", help=path_help)
        p.add_argument("--folds", type=int, default=0, metavar="K", help=FOLDS_HELP)
        p.add_argument("--out", metavar="PATH", help=OUT_HELP)
        p.set_defaults(func=_dataset, extra="eval")

    p = datasets.add_parser("all", help="every dataset whose raw files are present (missing ones are skipped)")
    p.add_argument("--raw", default=RAW, metavar="DIR", help="directory the datasets were downloaded to (default: raw)")
    p.add_argument("--only", metavar="NAMES", help="comma-separated dataset names, e.g. ammore,eedi-12")
    p.add_argument("--no-bkt", action="store_true", help="leave out the BKT grid search on ASSISTments 2009")
    p.add_argument("--folds", type=int, default=0, metavar="K", help=FOLDS_HELP)
    p.add_argument("--out", metavar="PATH", help=OUT_HELP)
    p.set_defaults(func=_all, extra="eval")

    p = subparsers.add_parser("sim", help="compare the Kwan policy with syllabus order and random practice")
    p.add_argument("graph", nargs="?", help="curriculum graph JSON, schema v2 or v1 (default: the committed graph)")
    p.add_argument(
        "--v2", action="store_true", help="also print the v2 report (forgetting, four profiles, five policies)"
    )
    p.add_argument("--learners", type=int, metavar="N", help="simulated learners per profile and policy (default: 30)")
    p.add_argument("--out", metavar="PATH", help=OUT_HELP)
    p.set_defaults(func=_sim, extra=None)

    p = subparsers.add_parser("bench", help="time the recommender on three seeded learner states")
    p.add_argument("graph", nargs="?", help="curriculum graph JSON, schema v2 or v1 (default: the committed graph)")
    p.add_argument("--calls", type=int, metavar="N", help="timed calls per learner state")
    p.add_argument("--hardware", metavar="TEXT", help="what to record as the hardware")
    p.add_argument("--out", metavar="PATH", help=OUT_HELP)
    p.set_defaults(func=_bench, extra=None)
