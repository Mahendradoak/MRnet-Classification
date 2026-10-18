# MRNet knee MRI classification

Detects **abnormality, anterior cruciate ligament (ACL) tear and meniscal tear** from knee MRI in the Stanford [MRNet-v1.0](https://stanfordmlgroup.github.io/competitions/mrnet/) dataset. Each exam has sagittal, coronal and axial series.

## What it does

1. **Treats one exam as one sample.** The sagittal, coronal and axial series loaded together always belong to the same patient, and all three labels (abnormal, ACL, meniscus) are predicted jointly.
2. **Keeps every slice.** MRNet exams have roughly 17–60 slices. All of them are used, with a 224×224 centre crop and an exam-level z-score. There is no fixed 16-slice crop that would discard off-centre pathology.
3. **Encodes slices and pools them as a bag (multiple-instance learning).** Each slice goes through an ImageNet-pretrained ResNet18 adapted to 1 channel; the pretrained first-layer kernel is summed over RGB rather than discarded. The bag of slice features is pooled by either
   - **max pooling**, the MRNet baseline, or
   - **gated attention** (Ilse et al., 2018), whose per-slice weights show which slices drove the prediction.
4. **Supports two fusion strategies:**
   - **Late fusion.** One model per plane, then a per-task logistic regression on the three plane probabilities, as in the original MRNet paper.
   - **Joint fusion.** Three plane encoders whose exam embeddings are concatenated and trained end-to-end.
5. **Separates tuning data from test data.**
   - A stratified **internal validation set** (15%) is carved out of the official training data. It is used for early stopping, model selection, decision thresholds and fusion weights.
   - The **official validation set (120 exams) is used once, as the test set**, because the real MRNet test set is hidden.
6. **Reports results with uncertainty.** Per task:
   - test AUC with a 95% bootstrap CI;
   - Brier score;
   - sensitivity and specificity at a threshold chosen on the internal validation set;
   - a paired bootstrap test of the AUC difference between any two models.

## Quick start

```bash
pip install -r requirements.txt
# MRNet-v1.0 must be requested from Stanford; unpack it to ./MRNet-v1.0

for p in sagittal coronal axial; do
  python -m mrnet.cli train --data MRNet-v1.0 --plane $p --out runs/$p
done
python -m mrnet.cli fuse     --runs runs/sagittal runs/coronal runs/axial --out runs/late_fusion
python -m mrnet.cli train    --data MRNet-v1.0 --plane all --out runs/joint
python -m mrnet.cli evaluate --run runs/late_fusion --compare runs/axial   # does fusion beat the best plane?
python -m mrnet.cli evaluate --run runs/joint --compare runs/late_fusion
```

Each run directory contains:
- `config.json` and `splits.json` (exact case IDs);
- `best.pt` and `history.csv`;
- `val_pred.csv` and `test_pred.csv` (per-exam probabilities);
- `test_metrics.csv`;
- `compare_vs_<run>.json`.

## Tests

```bash
pytest -q      # 8 tests on a synthetic MRNet-shaped dataset (~3 min on CPU)
ruff check .
```

The tests check:
- that the official validation set is only ever test data;
- that the three planes of a sample are read from one case ID;
- that no slices are dropped;
- that attention weights sum to 1 and are permutation-invariant;
- the bootstrap metrics;
- the full train → fuse → evaluate path with both pooling types.

## What was wrong before (`archive/`)

The original script (`archive/original_code.py`) and report (`archive/MIA_report_original.pdf`) had the following problems:

| Problem | Consequence | Fix |
|---|---|---|
| Multi-plane training `zip`ped three DataLoaders, each with its own random `WeightedRandomSampler`; the label came from the sagittal loader only | Training samples combined three **different patients**. The multi-plane results in report Tables 3–4 (MP2 0.838, MPLR 0.820, MPFuseNet 0.686) and the "+0.011 over axial" claim are not valid | One dataset item = one exam with all planes |
| Official validation set used for early stopping **and** for the reported AUC | Optimistic estimates; no untouched test data | Internal validation split for tuning; official valid used only for final evaluation |
| Single 0.011 AUC difference reported on 120 exams with no uncertainty | Difference is well within sampling noise | Bootstrap CIs and paired comparison |
| `max_slices=16` centre crop | Off-centre lesions removed from many exams | All slices |
| First conv layer replaced with random weights | Lost pretrained low-level filters | Pretrained kernel summed to 1 channel |
| Z-score undone by per-slice min-max to uint8 before the transforms | Inconsistent intensity scaling across slices | Exam-level z-score only |
| ACL task only; 3–8 epochs | Incomplete, under-trained | All three tasks jointly; early stopping on internal validation AUC |
| Plotting test function generated curves from hard-coded dummy numbers ("best AUC 0.95") | Risk of fabricated figures entering reports | Removed |

The single-plane results in the report (axial val AUC 0.827) were selected on the same data they were reported on, so they should also be re-run. For context, the original MRNet paper reports roughly 0.94 / 0.97 / 0.85 test AUC for abnormal / ACL / meniscus.

## Status

Tested end-to-end on synthetic data. **Real results still need to be produced** on a GPU; budget roughly 1–3 h per plane model on a T4.

Licence: none yet. Note that MRNet's data-use agreement forbids redistributing the images.
