import json

import numpy as np
import pytest
import torch

import mrnet.data as data_mod
from mrnet.cli import main
from mrnet.data import PLANES, MRNetExams, make_splits, read_labels
from mrnet.metrics import auc_ci, paired_auc_difference, threshold_metrics, youden_threshold
from mrnet.model import GatedAttentionPool, PlaneNet, slice_encoder


def test_labels_and_splits(mrnet_root):
    lab = read_labels(mrnet_root, "train")
    assert list(lab.columns[:4]) == ["case", "abnormal", "acl", "meniscus"] and len(lab) == 40
    s = make_splits(mrnet_root, val_frac=0.25)
    assert set(s["test"].case) == set(range(40, 56))           # official valid = test only
    assert not set(s["train"].case) & set(s["val"].case)
    assert len(s["train"]) + len(s["val"]) == 40


def test_all_planes_come_from_the_same_exam(mrnet_root, monkeypatch):
    loaded = []
    real_load = np.load

    def spy(path, *a, **k):
        loaded.append(str(path))
        return real_load(path, *a, **k)

    monkeypatch.setattr(data_mod.np, "load", spy)
    s = make_splits(mrnet_root, val_frac=0.25)
    ds = MRNetExams(mrnet_root, s["train"], train=True)
    for i in range(5):
        loaded.clear()
        exam = ds[i]
        ids = {p.rsplit("/", 1)[-1] for p in loaded}
        assert ids == {f"{exam['case']:04d}.npy"}, loaded      # one case ID across all three planes
        assert len(loaded) == 3


def test_every_slice_is_kept(mrnet_root):
    s = make_splits(mrnet_root, val_frac=0.25)
    ds = MRNetExams(mrnet_root, s["test"])
    exam = ds[0]
    raw = np.load(mrnet_root / "valid" / "axial" / f"{exam['case']:04d}.npy")
    assert exam["axial"].shape[0] == raw.shape[0]                # no 16-slice centre crop


def test_attention_pool_is_a_distribution_and_permutation_invariant():
    torch.manual_seed(0)
    pool = GatedAttentionPool(dim=8, hidden=4)
    h = torch.randn(11, 8)
    z, a = pool(h)
    assert torch.isclose(a.sum(), torch.tensor(1.0))
    perm = torch.randperm(11)
    z2, _ = pool(h[perm])
    assert torch.allclose(z, z2, atol=1e-6)


def test_encoder_and_plane_net_shapes():
    enc = slice_encoder(pretrained=False)
    assert enc.conv1.in_channels == 1
    net = PlaneNet(n_tasks=3, pretrained=False)
    out = net(torch.randn(13, 1, 48, 48))
    assert out["logits"].shape == (3,) and out["attention"].shape == (13,)


def test_metrics():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    p = np.array([.1, .3, .8, .7, .2, .9, .6, .4, .85, .15])
    auc, lo, hi = auc_ci(y, p, n_boot=300)
    assert lo <= auc <= hi
    same = paired_auc_difference(y, p, p, n_boot=200)
    assert same["auc_diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    thr = youden_threshold(y, p)
    m = threshold_metrics(y, p, thr)
    assert m["tp"] + m["fn"] == y.sum()


@pytest.mark.parametrize("pool", ["attention", "max"])
def test_end_to_end(mrnet_root, tmp_path, pool):
    common = ["--data", str(mrnet_root), "--epochs", "2", "--no-pretrained", "--workers", "0",
              "--accum", "4", "--val-frac", "0.25", "--pool", pool]
    for p in PLANES:
        main(["train", "--plane", p, "--out", str(tmp_path / p), *common])
    main(["train", "--plane", "all", "--out", str(tmp_path / "joint"), *common])
    main(["fuse", "--runs", *[str(tmp_path / p) for p in PLANES], "--out", str(tmp_path / "fusion")])
    main(["evaluate", "--run", str(tmp_path / "fusion"), "--compare", str(tmp_path / "axial")])
    main(["evaluate", "--run", str(tmp_path / "joint")])
    assert (tmp_path / "fusion" / "test_metrics.csv").exists()
    coef = json.loads((tmp_path / "fusion" / "fusion_coefficients.json").read_text())
    assert set(coef) == {"abnormal", "acl", "meniscus"}
    splits = json.loads((tmp_path / "axial" / "splits.json").read_text())
    assert set(splits["test"]) == set(range(40, 56))
