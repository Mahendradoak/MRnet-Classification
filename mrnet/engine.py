"""Training and prediction for single-plane and joint multi-plane models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import MRNetExams, collate_one


def _forward(model: nn.Module, exam: dict, plane: str | None, device: torch.device) -> torch.Tensor:
    if plane is not None:                                   # single-plane model
        return model(exam[plane].to(device))["logits"]
    moved = {k: (v.to(device) if torch.is_tensor(v) and k != "labels" else v) for k, v in exam.items()}
    return model(moved)["logits"]


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, plane: str | None, tasks: tuple[str, ...],
            device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for exam in loader:
        prob = torch.sigmoid(_forward(model, exam, plane, device)).cpu().numpy()
        rows.append({"case": exam["case"], **{f"y_{t}": float(exam["labels"][i]) for i, t in enumerate(tasks)},
                     **{f"p_{t}": float(prob[i]) for i, t in enumerate(tasks)}})
    return pd.DataFrame(rows)


def mean_auc(pred: pd.DataFrame, tasks: tuple[str, ...]) -> float:
    aucs = [roc_auc_score(pred[f"y_{t}"], pred[f"p_{t}"]) for t in tasks if pred[f"y_{t}"].nunique() == 2]
    return float(np.mean(aucs)) if aucs else float("nan")


def train(model: nn.Module, root: str, splits: dict[str, pd.DataFrame], tasks: tuple[str, ...], planes: tuple[str, ...],
          plane: str | None, out_dir: Path, epochs: int = 30, lr: float = 1e-5, weight_decay: float = 1e-2,
          accum: int = 8, patience: int = 6, workers: int = 4, seed: int = 42) -> dict:
    """``plane=None`` trains a joint multi-plane model; otherwise a single-plane model.

    Class imbalance is handled with per-task ``pos_weight`` in the BCE loss
    (computed on the training split only) instead of a weighted sampler.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_planes = planes if plane is None else (plane,)

    def loader(part: str, train_mode: bool) -> DataLoader:
        ds = MRNetExams(root, splits[part], tasks, use_planes, train=train_mode)
        return DataLoader(ds, batch_size=1, shuffle=train_mode, num_workers=workers, collate_fn=collate_one,
                          generator=torch.Generator().manual_seed(seed) if train_mode else None)

    tr, va = loader("train", True), loader("val", False)
    y = splits["train"][list(tasks)].to_numpy()
    pos_weight = torch.tensor((len(y) - y.sum(0)) / np.maximum(y.sum(0), 1), dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, bad, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for step, exam in enumerate(tr, 1):
            loss = loss_fn(_forward(model, exam, plane, dev), exam["labels"].to(dev)) / accum
            loss.backward()
            total += loss.item() * accum
            if step % accum == 0 or step == len(tr):
                opt.step()
                opt.zero_grad(set_to_none=True)
        sched.step()
        val_pred = predict(model, va, plane, tasks, dev)
        auc = mean_auc(val_pred, tasks)
        history.append({"epoch": epoch, "train_loss": total / len(tr), "val_mean_auc": auc,
                        **{f"val_auc_{t}": float(roc_auc_score(val_pred[f"y_{t}"], val_pred[f"p_{t}"]))
                           for t in tasks if val_pred[f"y_{t}"].nunique() == 2},
                        "seconds": time.time() - t0})
        print(json.dumps(history[-1]))
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev))
    predict(model, va, plane, tasks, dev).to_csv(out_dir / "val_pred.csv", index=False)
    predict(model, loader("test", False), plane, tasks, dev).to_csv(out_dir / "test_pred.csv", index=False)
    return {"best_val_mean_auc": best, "epochs_run": len(history)}


def late_fusion(plane_runs: dict[str, Path], tasks: tuple[str, ...], out_dir: Path) -> None:
    """MRNet-style fusion: per task, a logistic regression on the three plane
    probabilities, fitted on the INTERNAL validation predictions, applied to test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = {}
    for part in ("val", "test"):
        merged = None
        for p, run in plane_runs.items():
            df = pd.read_csv(run / f"{part}_pred.csv")
            df = df.rename(columns={f"p_{t}": f"p_{t}_{p}" for t in tasks})
            keep = ["case"] + [f"p_{t}_{p}" for t in tasks] + ([f"y_{t}" for t in tasks] if merged is None else [])
            merged = df[keep] if merged is None else merged.merge(df[keep], on="case", validate="one_to_one")
        parts[part] = merged
    coefs = {}
    for t in tasks:
        cols = [f"p_{t}_{p}" for p in plane_runs]
        lr = LogisticRegression(C=1.0).fit(parts["val"][cols], parts["val"][f"y_{t}"])
        coefs[t] = dict(zip(cols, lr.coef_[0].tolist(), strict=True)) | {"intercept": float(lr.intercept_[0])}
        for part in ("val", "test"):
            parts[part][f"p_{t}"] = lr.predict_proba(parts[part][cols])[:, 1]
    for part in ("val", "test"):
        parts[part].to_csv(out_dir / f"{part}_pred.csv", index=False)
    (out_dir / "fusion_coefficients.json").write_text(json.dumps(coefs, indent=1))
