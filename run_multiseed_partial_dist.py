"""
Multi-seed driver for the SBP partial-distance + imprinting ablation.

Runs the experiment across seeds, saves per-seed JSON, and reports the
mean +/- std across seeds (session-wise and for the key metrics).

Usage:
    python run_multiseed_partial_dist.py --dataset cifar100
    python run_multiseed_partial_dist.py --dataset miniimagenet
    python run_multiseed_partial_dist.py --dataset cifar100 --aggregate-only

Per-seed results are cached to results_partial_dist/<dataset>/seed<N>.json,
and existing seeds are skipped, so an interrupted sweep resumes cheaply.
"""
import argparse
import json
import os
from typing import Dict, List

import numpy as np

RESULT_ROOT = os.environ.get("SBP_RESULT_ROOT", "./results_partial_dist")
DEFAULT_SEEDS = [1993, 1994, 1995, 1996, 1997]


def seed_path(dataset: str, seed: int) -> str:
    return os.path.join(RESULT_ROOT, dataset, f"seed{seed}.json")


def run_one(dataset: str, seed: int) -> Dict:
    """Run a single seed and persist its per-session results."""
    if dataset == "cifar100":
        import run_cifar100_partial_dist as mod
        kwargs = dict(base_epochs=130, incremental_iterations=25)
    elif dataset == "miniimagenet":
        import run_mini_imagenet_partial_dist as mod
        # Matched to the CIFAR-100 settings (130 base epochs / 25 incremental
        # iterations) rather than run_mini_imagenet.py's own 200/120. The
        # 200/120 sweep was unstable (std of 16-23 points across seeds in the
        # middle sessions); 120 incremental iterations is ~5x CIFAR's gradient
        # budget per session, which drives old features off their prototypes.
        # Archived at results_partial_dist/miniimagenet_200ep_120iter/.
        kwargs = dict(base_epochs=130, incremental_iterations=25)
    else:
        raise ValueError(dataset)

    out = mod.run(seed=seed, initial_budget=0.85, num_incremental_sessions=8, **kwargs)

    # Keep only what aggregation needs (per_class_acc is large and per-seed).
    slim = {
        "seed": seed,
        "dataset": dataset,
        "session_acc": [r["overall_acc"] for r in out["session_results"]],
        "base_acc": [r["base_classes_acc"] for r in out["session_results"]],
        "new_acc": [r["new_classes_acc"] for r in out["session_results"]],
        "final_acc": out["final_acc"],
        "base_session_acc": out["base_session_acc"],
        "avg_acc": out["avg_acc"],
        "forgetting": out["forgetting"],
        "performance_drop": out["base_session_acc"] - out["final_acc"],
    }
    path = seed_path(dataset, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(slim, f, indent=2)
    print(f"\n[Saved] {path}")
    return slim


def aggregate(dataset: str, seeds: List[int]) -> None:
    runs = []
    missing = []
    for s in seeds:
        p = seed_path(dataset, s)
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        else:
            missing.append(s)

    if not runs:
        print(f"[Aggregate] no results found for {dataset}")
        return
    if missing:
        print(f"[Aggregate] WARNING: missing seeds {missing} -- averaging over {len(runs)} seed(s)")

    acc = np.array([r["session_acc"] for r in runs])   # [seeds, sessions]
    base = np.array([r["base_acc"] for r in runs])
    new = np.array([r["new_acc"] for r in runs])
    n_sessions = acc.shape[1]

    print(f"\n{'='*74}")
    print(f"SBP PARTIAL-DISTANCE + IMPRINTING -- {dataset.upper()} "
          f"({len(runs)} seeds: {[r['seed'] for r in runs]})")
    print(f"{'='*74}")
    print(f"{'Session':<9}{'Overall (mean±std)':<24}{'Base':<20}{'New':<20}")
    print("-" * 74)
    for i in range(n_sessions):
        print(f"{i:<9}"
              f"{acc[:, i].mean():>6.2f} ± {acc[:, i].std():<14.2f}"
              f"{base[:, i].mean():>6.2f} ± {base[:, i].std():<11.2f}"
              f"{new[:, i].mean():>6.2f} ± {new[:, i].std():<11.2f}")

    def ms(key):
        v = np.array([r[key] for r in runs])
        return v.mean(), v.std()

    print(f"\n--- Key Metrics (mean ± std over {len(runs)} seeds) ---")
    for label, key in [("Final Accuracy", "final_acc"),
                       ("Base Session Accuracy", "base_session_acc"),
                       ("Performance Drop", "performance_drop"),
                       ("Average Session Accuracy", "avg_acc"),
                       ("Forgetting (Base Classes)", "forgetting")]:
        m, s = ms(key)
        print(f"{label:<28}: {m:6.2f} ± {s:.2f}")

    # compact row, matching the format used for the other methods
    print(f"\nSession  " + "".join(f"{i:<7}" for i in range(n_sessions)))
    print(f"SBP-PD   " + "".join(f"{acc[:, i].mean():<7.1f}" for i in range(n_sessions)))
    print(f"(±std)   " + "".join(f"{acc[:, i].std():<7.1f}" for i in range(n_sessions)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar100", "miniimagenet"])
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()

    if not a.aggregate_only:
        for s in a.seeds:
            if os.path.exists(seed_path(a.dataset, s)):
                print(f"[Skip] seed {s} already done -> {seed_path(a.dataset, s)}")
                continue
            print(f"\n{'#'*74}\n# {a.dataset} | seed {s}\n{'#'*74}")
            run_one(a.dataset, s)

    aggregate(a.dataset, a.seeds)
