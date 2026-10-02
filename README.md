# SBP — code for the CIFAR-100 and miniImageNet experiments

## Contents

| File | Purpose |
| --- | --- |
| `sbp.py` | The SBP budgeting algorithm: mask allocation, gradient hooks, free-channel reinitialization. Backbone-agnostic. |
| `run_cifar100_partial_dist.py` | Entry point — CIFAR-100. |
| `run_mini_imagenet_partial_dist.py` | Entry point — miniImageNet. |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU.

## Data

Both datasets are read from `./data`, relative to the working directory.

**CIFAR-100** downloads automatically on first run.

**miniImageNet** must be placed at `./data/miniimagenet/split`, with the
standard split CSVs:

```
data/miniimagenet/split/
    images/            # all 60,000 JPEGs, flat
    train.csv          # columns: filename,label
    val.csv
    test.csv
```

To use an existing copy elsewhere, symlink `./data` to it, or pass
`data_dir=...` to `run()`.

## Running

Single seed:

```bash
python run_cifar100_partial_dist.py --seed 0
python run_mini_imagenet_partial_dist.py --seed 0
```

## Options

Defaults reproduce the configuration described in the paper: 60 base classes,
8 incremental sessions of 5-way/5-shot, 130 base epochs, 25 incremental
iterations per session, base budget 0.85, iCaRL class order.

| Flag | Default | Effect |
| --- | --- | --- |
| `--seed` | 1993 | Random seed; also controls the class order and few-shot sampling. |
| `--base-epochs` | 130 | Base-session training epochs. |
| `--incremental-iterations` | 25 | Gradient steps per incremental session. |
| `--initial-budget` | 0.85 | Fraction of channel capacity allocated to the base session; the rest is split evenly across incremental sessions. |
| `--no-partial-distance` | off | Ablation: compute distances over all feature dimensions. |
| `--freeze-mask` | off | Ablation: pin the active subspace to the base-session set instead of growing it. |
| `--masked-imprint` | off | Ablation: zero inactive dimensions in the stored prototype instead of storing the full centroid. |
| `--no-reset-free-weights` | off | Ablation (CIFAR-100 only): disable per-epoch reinitialization of free channels. |

Runs are seeded (`torch`, `numpy`, `random`, `cudnn.deterministic=True`).
Exact bit-reproducibility across different GPU models or CUDA versions is not
guaranteed.
