"""MRNet-v1.0 loading with patient-aligned planes.

Layout of the Stanford release::

    MRNet-v1.0/
        train/{sagittal,coronal,axial}/0000.npy ... 1129.npy   (S x 256 x 256, uint8)
        valid/{sagittal,coronal,axial}/1130.npy ... 1249.npy
        train-{abnormal,acl,meniscus}.csv   valid-{...}.csv    (case,label; no header)

The official test set is hidden. This code therefore:
  * carves a stratified internal validation set out of ``train`` for model
    selection, early stopping, threshold choice and fusion weights;
  * uses the official ``valid`` set (120 exams) only once, as the test set.

The original script tuned and reported on ``valid``, and its multi-plane loader
zipped three independently shuffled loaders, so the sagittal, coronal and axial
inputs of a training sample came from different patients. Here one sample is
one exam, and all three planes are read for that exam's ID.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PLANES = ("sagittal", "coronal", "axial")
TASKS = ("abnormal", "acl", "meniscus")


def read_labels(root: str | Path, split: str, tasks: tuple[str, ...] = TASKS) -> pd.DataFrame:
    """One row per exam with an integer column per task. ``split`` is 'train' or 'valid'."""
    root = Path(root)
    df = None
    for t in tasks:
        cur = pd.read_csv(root / f"{split}-{t}.csv", header=None, names=["case", t], dtype={"case": int, t: int})
        df = cur if df is None else df.merge(cur, on="case", how="inner", validate="one_to_one")
    assert df is not None
    df["source"] = split
    return df.sort_values("case").reset_index(drop=True)


def make_splits(root: str | Path, val_frac: float = 0.15, seed: int = 42,
                tasks: tuple[str, ...] = TASKS) -> dict[str, pd.DataFrame]:
    """train / internal-val from the official train set; test = official valid set."""
    full = read_labels(root, "train", tasks)
    strat = full[list(tasks)].astype(str).agg("".join, axis=1)
    if strat.value_counts().min() < 2:      # rare label combinations: stratify on first task only
        strat = full[tasks[0]]
    tr, va = train_test_split(full, test_size=val_frac, random_state=seed, stratify=strat)
    test = read_labels(root, "valid", tasks)
    splits = {"train": tr.sort_values("case").reset_index(drop=True),
              "val": va.sort_values("case").reset_index(drop=True), "test": test}
    ids = [set(s.case) for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "case leakage between splits"
    return splits


def volume_path(root: str | Path, source: str, plane: str, case: int) -> Path:
    return Path(root) / source / plane / f"{case:04d}.npy"


def preprocess(vol: np.ndarray, crop: int = 224) -> np.ndarray:
    """Centre-crop to ``crop`` and z-score the whole exam. Keeps EVERY slice."""
    vol = vol.astype(np.float32)
    s, h, w = vol.shape
    t, l = max((h - crop) // 2, 0), max((w - crop) // 2, 0)
    vol = vol[:, t:t + crop, l:l + crop]
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def augment(vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exam-level augmentation applied identically to every slice of one plane:
    small translation, intensity scale/shift. No flips (knee laterality is meaningful
    for some planes and MRNet volumes are not laterality-normalised)."""
    dy, dx = rng.integers(-12, 13, size=2)
    vol = np.roll(vol, (int(dy), int(dx)), axis=(1, 2))
    return (vol * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)).astype(np.float32)


class MRNetExams(Dataset):
    """Each item is ONE exam: all three planes of the same case plus its labels."""

    def __init__(self, root: str | Path, frame: pd.DataFrame, tasks: tuple[str, ...] = TASKS,
                 planes: tuple[str, ...] = PLANES, train: bool = False, crop: int = 224):
        self.root, self.frame, self.tasks, self.planes = Path(root), frame.reset_index(drop=True), tasks, planes
        self.train, self.crop = train, crop

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> dict:
        row = self.frame.iloc[i]
        case, source = int(row.case), row.source
        rng = np.random.default_rng([torch.initial_seed() % 2**32, i]) if self.train else None
        out = {"case": case, "labels": torch.tensor([float(row[t]) for t in self.tasks])}
        for p in self.planes:
            vol = preprocess(np.load(volume_path(self.root, source, p, case)), self.crop)
            if rng is not None:
                vol = augment(vol, rng)
            out[p] = torch.from_numpy(vol[:, None].copy())      # (S, 1, H, W); S varies by exam
        return out


def collate_one(batch: list[dict]) -> dict:
    """Exams have different slice counts, so batches hold exactly one exam."""
    assert len(batch) == 1
    return batch[0]
