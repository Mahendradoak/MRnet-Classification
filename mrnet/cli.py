"""Command line.

    python -m mrnet.cli train    --data MRNet-v1.0 --plane axial --out runs/axial
    python -m mrnet.cli train    --data MRNet-v1.0 --plane all   --out runs/joint
    python -m mrnet.cli fuse     --runs runs/sagittal runs/coronal runs/axial --out runs/late_fusion
    python -m mrnet.cli evaluate --run runs/late_fusion --compare runs/axial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import PLANES, TASKS, make_splits
from .engine import late_fusion, train
from .metrics import paired_auc_difference, summarise
from .model import MultiPlaneNet, PlaneNet


def cmd_train(a) -> None:
    tasks = tuple(a.tasks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = make_splits(a.data, val_frac=a.val_frac, seed=a.seed, tasks=tasks)
    (out / "splits.json").write_text(json.dumps({k: v.case.tolist() for k, v in splits.items()}))
    (out / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    plane = None if a.plane == "all" else a.plane
    model = (MultiPlaneNet(len(tasks), PLANES, a.pool, not a.no_pretrained) if plane is None
             else PlaneNet(len(tasks), a.pool, not a.no_pretrained))
    res = train(model, a.data, splits, tasks, PLANES, plane, out, epochs=a.epochs, lr=a.lr, accum=a.accum,
                patience=a.patience, workers=a.workers, seed=a.seed)
    (out / "train_summary.json").write_text(json.dumps(res, indent=1))
    print(res)


def cmd_fuse(a) -> None:
    runs = {Path(r).name: Path(r) for r in a.runs}
    cfg = json.loads((Path(a.runs[0]) / "config.json").read_text())
    late_fusion(runs, tuple(cfg["tasks"]), Path(a.out))
    (Path(a.out) / "config.json").write_text(json.dumps({"tasks": cfg["tasks"], "fused_runs": a.runs}, indent=1))


def cmd_evaluate(a) -> None:
    run = Path(a.run)
    tasks = json.loads((run / "config.json").read_text())["tasks"]
    val, test = pd.read_csv(run / "val_pred.csv"), pd.read_csv(run / "test_pred.csv")
    rows = []
    for t in tasks:
        rows.append({"task": t, **summarise(test[f"y_{t}"], test[f"p_{t}"], val[f"y_{t}"], val[f"p_{t}"])})
    res = pd.DataFrame(rows)
    res.to_csv(run / "test_metrics.csv", index=False)
    print(res.round(3).to_string(index=False))
    if a.compare:
        other = pd.read_csv(Path(a.compare) / "test_pred.csv")
        m = test.merge(other, on="case", suffixes=("", "_b"), validate="one_to_one")
        comp = {t: paired_auc_difference(m[f"y_{t}"], m[f"p_{t}"], m[f"p_{t}_b"]) for t in tasks}
        (run / f"compare_vs_{Path(a.compare).name}.json").write_text(json.dumps(comp, indent=1))
        for t, c in comp.items():
            print(f"{t:9s} dAUC={c['auc_diff']:+.3f} 95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] "
                  f"p={c['p_bootstrap']:.3f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mrnet")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="path to MRNet-v1.0")
    t.add_argument("--out", required=True)
    t.add_argument("--plane", choices=[*PLANES, "all"], required=True)
    t.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    t.add_argument("--pool", choices=["attention", "max"], default="attention")
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--accum", type=int, default=8, help="exams per optimiser step")
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--val-frac", type=float, default=0.15)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)
    f = sub.add_parser("fuse")
    f.add_argument("--runs", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fuse)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--compare", help="another run to compare against (paired bootstrap on AUC)")
    e.set_defaults(func=cmd_evaluate)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
