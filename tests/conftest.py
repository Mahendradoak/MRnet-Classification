"""Synthetic MRNet-v1.0-shaped dataset (small images, variable slice counts)."""

import numpy as np
import pandas as pd
import pytest

from mrnet.data import PLANES, TASKS


@pytest.fixture(scope="session")
def mrnet_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("MRNet-v1.0")
    rng = np.random.default_rng(0)
    splits = {"train": range(0, 40), "valid": range(40, 56)}
    for split, cases in splits.items():
        labels = {t: [] for t in TASKS}
        for c in cases:
            acl = int(c % 3 == 0)
            men = int(c % 4 == 0)
            abn = int(acl or men or c % 5 == 0)
            for t, v in zip(TASKS, (abn, acl, men), strict=True):
                labels[t].append(v)
            for p in PLANES:
                s = int(rng.integers(8, 20))                   # MRNet exams vary in slice count
                vol = rng.integers(0, 60, size=(s, 48, 48)).astype(np.uint8)
                if acl:
                    vol[s // 2 - 1: s // 2 + 2, 18:30, 18:30] = 220   # a "lesion" on central slices
                d = root / split / p
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{c:04d}.npy", vol)
        for t in TASKS:
            pd.DataFrame({"case": list(cases), "label": labels[t]}).to_csv(
                root / f"{split}-{t}.csv", header=False, index=False)
    return root
