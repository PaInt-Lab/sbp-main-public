import copy
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sbp import SBP
from utils import *
from resnet import *
from data.mini_imagenet import *

# ResNet-18's last conv before avgpool: its 512 output channels are exactly
# the 512 feature dimensions the prototype classifier operates on.
FEATURE_LAYER = "layer4.1.conv2.weight"

class PartialDistanceClassifier(MahalanobisClassifier):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        # Default all-ones (== conventional full-space distance) so behaviour
        # before the first assignment is well defined.
        self.register_buffer("active_mask", torch.ones(in_features))

    def forward(self, x):
        diff = (x.unsqueeze(1) - self.weight.unsqueeze(0)) * self.active_mask
        transformed = torch.matmul(diff, self.precision_matrix)
        dist_sq = torch.sum(diff * transformed, dim=2)
        return -self.scale * dist_sq

@torch.no_grad()
def update_active_mask(model: nn.Module, budgeter, verbose: bool = True) -> torch.Tensor:
    """active = frozen | assigned for the feature-producing layer."""
    device = model.fc_out.active_mask.device
    if FEATURE_LAYER not in budgeter.frozen_mask:
        if verbose:
            print(f"  [PartialDist] WARNING: {FEATURE_LAYER} not tracked by budgeter; "
                  f"leaving mask unchanged")
        return model.fc_out.active_mask

    frozen = budgeter.frozen_mask[FEATURE_LAYER].to(device)
    assigned = budgeter.assigned_mask[FEATURE_LAYER].to(device)
    active = (frozen | assigned).float()
    model.fc_out.active_mask.copy_(active)
    if verbose:
        n, d = int(active.sum().item()), active.numel()
        print(f"  [PartialDist] active dims: {n}/{d} ({100.0*n/d:.1f}%) "
              f"[frozen={int(frozen.sum())}, assigned={int(assigned.sum())}]")
    return active


@torch.no_grad()
def imprint_partial(model, loader, device, classes, mask_prototypes: bool = False, **kw):
    imprint_classifier_weights(model, loader, device, classes, **kw)
    if mask_prototypes:
        model.fc_out.weight.data[classes] *= model.fc_out.active_mask


def train_incremental_ce(model, dataloader, optimizer, scheduler, device,
                         budgeter, num_iterations, max_grad_norm=1.0):
    """cross-entropy incremental training under SBP gradient masking.
    """
    model.eval()  # BN/LN kept frozen; SBP masks gradients
    iteration, total_loss = 0, 0.0
    while iteration < num_iterations:
        for inputs, targets in dataloader:
            if iteration >= num_iterations:
                break
            if iteration % max(1, len(dataloader)) == 0:
                budgeter.new_epoch_update()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            iteration += 1
    return total_loss / max(1, num_iterations)


def create_fscil_task_schedule(num_classes: int = 100, base_classes: int = 60,
                                n_way: int = 5, num_sessions: int = 8, seed: int = 42,
                                use_icarl_order: bool = True):

    random.seed(seed)
    class_order = list(range(num_classes))
    random.shuffle(class_order)

    # Session 0: Base Classes
    tasks = [class_order[:base_classes]]

    # Sessions 1-N: Incremental batches
    remaining = class_order[base_classes:]
    for i in range(0, len(remaining), n_way):
        if len(tasks) - 1 < num_sessions:
            tasks.append(remaining[i:i + n_way])

    return tasks

@torch.no_grad()
def evaluate_partial(model, dataloader, device, seen_classes):
    model.eval()
    correct = total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        mask = torch.zeros(logits.size(1), device=device)
        mask[seen_classes] = 1
        logits = logits * mask - 1e9 * (1 - mask)
        correct += logits.max(1)[1].eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate_per_class_partial(model, dataloader, device, seen_classes):
    model.eval()
    correct = {c: 0 for c in seen_classes}
    total = {c: 0 for c in seen_classes}
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        mask = torch.zeros(logits.size(1), device=device)
        mask[seen_classes] = 1
        logits = logits * mask - 1e9 * (1 - mask)
        preds = logits.max(1)[1]
        for i in range(targets.size(0)):
            c = targets[i].item()
            if c in total:
                total[c] += 1
                correct[c] += int(preds[i].item() == c)
    return {c: 100.0 * correct[c] / total[c] if total[c] else 0.0 for c in seen_classes}

def train_base_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    budgeter: SBP,
    max_grad_norm: float = 1.0,
    time_tracker: Optional[TrackTime] = None,
    use_cutmix_mixup: bool = True,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    budgeter.new_epoch_update()

    for inputs, targets in dataloader:
        t0 = time.time()
        inputs = inputs.to(device)
        targets = targets.to(device)
        if time_tracker:
            time_tracker.add('data_loading', time.time() - t0)

        # Apply Mixup / CutMix
        r = np.random.rand()
        if use_cutmix_mixup and r < 0.5:
            if r < 0.25:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, alpha=1.0, device=device)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        logits = model(inputs)

        if lam < 1.0:
            loss = mixup_criterion(F.cross_entropy, logits, targets_a, targets_b, lam)
        else:
            loss = F.cross_entropy(logits, targets)

        if time_tracker:
            time_tracker.add('forward_pass', time.time() - t0)

        t0 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if time_tracker:
            time_tracker.add('backward_pass', time.time() - t0)

        _, preds = logits.max(1)
        total += targets.size(0)
        correct += (lam * preds.eq(targets_a).sum().float() + (1 - lam) * preds.eq(targets_b).sum().float()).item()
        total_loss += loss.item()

    return total_loss / len(dataloader), 100.0 * correct / total

    
def run(
    base_classes: int = 60,
    n_way: int = 5,
    k_shot: int = 5,
    num_incremental_sessions: int = 8,
    base_epochs: int = 130,
    incremental_iterations: int = 25,
    initial_budget: float = 0.85,
    seed: int = 1993,
    use_icarl_order: bool = True,
    reset_free_weights: bool = True,
    max_grad_norm: float = 1.0,
    use_partial_distance: bool = True,
    freeze_mask_after_base: bool = False,
    mask_prototypes: bool = False,
    data_dir: str = "./data/miniimagenet/split",
):
    tracker = TrackTime()
    t_start = time.time()

    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    import random as _r; _r.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print("SBP + PARTIAL DISTANCE + IMPRINTING (miniImageNet, ResNet-18)")
    print(f"Partial distance: {'ON' if use_partial_distance else 'OFF (full-space control)'}")
    print(f"Device: {device} | seed={seed} | initial_budget={initial_budget}")
    print(f"{'='*70}\n")

    num_classes = 100
    trainset, testset = get_miniimagenet_datasets(data_dir=data_dir, seed=seed)
    tasks = create_fscil_task_schedule(num_classes, base_classes, n_way,
                                       num_incremental_sessions, seed,
                                       use_icarl_order=use_icarl_order)
    budgets = compute_budget_per_task(len(tasks), initial_budget)
    print(f"Task schedule: {[len(t) for t in tasks]}")
    print(f"Budgets: {[f'{b*100:.2f}%' for b in budgets]}\n")

    model = ResNet18(num_classes=num_classes).to(device)
    # Swap in the partial-distance head, preserving the initialized weights.
    head = PartialDistanceClassifier(model.feature_dim, num_classes).to(device)
    head.weight.data.copy_(model.fc_out.weight.data)
    head.precision_matrix.copy_(model.fc_out.precision_matrix)
    head.scale = model.fc_out.scale
    model.fc_out = head

    budgeter = SBP(
        model,
        budget_slice=budgets[0],
        replay_ratio=0.0,
        replay_weight=0.0,
        reset_assigned_weights=False,
        reset_free_weights=reset_free_weights,
        device=device,
        classifier_params={"fc_out.weight"},
        task_to_class={i + 1: t for i, t in enumerate(tasks)},
    )

    seen_classes: List[int] = []
    session_results: List[Dict] = []
    hooks = []

    for task_idx, task_classes in enumerate(tasks):
        is_base = task_idx == 0
        print(f"\n{'='*70}\nSESSION {task_idx}: {'BASE' if is_base else 'INCREMENTAL'} "
              f"| classes {task_classes[:6]}{'...' if len(task_classes) > 6 else ''}\n{'='*70}")

        for h in hooks:
            h.remove()
        hooks = []
        seen_classes.extend(task_classes)

        budgeter.budget_slice = budgets[task_idx]
        budgeter.new_task_update(task_id=task_idx + 1, task_classes=task_classes)

        if use_partial_distance and not (freeze_mask_after_base and task_idx > 0):
            update_active_mask(model, budgeter)

        if task_idx > 0:
            old = [c for c in seen_classes if c not in task_classes]
            if old and model.fc_out.weight.requires_grad:
                hooks.append(model.fc_out.weight.register_hook(make_classifier_hook(old)))

        if is_base:
            tracker.start("base_training")
            train_ds = TaskSubset(trainset, task_classes)
            loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                                num_workers=8, drop_last=True)
            opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=base_epochs)
            print(f"Training {base_epochs} epochs on {len(train_ds)} images")
            for ep in range(base_epochs):
                loss, acc = train_base_epoch(model, loader, opt, device, budgeter,
                                             max_grad_norm=max_grad_norm, time_tracker=tracker)
                sched.step()
                if (ep + 1) % 20 == 0 or ep == base_epochs - 1:
                    print(f"  epoch {ep+1:3d}/{base_epochs}: loss={loss:.4f} acc={acc:.2f}%")

            update_global_covariance(model, loader, device)
            print("[Imprint] base prototypes (active subspace)")
            imprint_partial(model, loader, device, task_classes,
                            mask_prototypes=mask_prototypes)
            set_bn_hard_freeze(model)
            tracker.stop("base_training")
        else:
            tracker.start("incremental_training")
            fs_ds = FewShotSubset(trainset, task_classes, k_shot=k_shot, seed=seed + task_idx)
            loader = DataLoader(fs_ds, batch_size=min(64, len(fs_ds)), shuffle=True,
                                num_workers=2, drop_last=False)
            print(f"Few-shot samples: {len(fs_ds)}")

            opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=incremental_iterations)
            train_incremental_ce(model, loader, opt, sched, device, budgeter,
                                 incremental_iterations, max_grad_norm)

            # Imprint AFTER training, on the (now updated) active subspace.
            print("[Imprint] new-class prototypes (active subspace)")
            imprint_partial(model, loader, device, task_classes,
                            mask_prototypes=mask_prototypes, augment_rounds=10)
            tracker.stop("incremental_training")
            # No drift compensation: assigned params for old tasks are frozen,
            # so old classes' active dims are unchanged by this session.

        tracker.start("evaluation")
        eval_loader = DataLoader(TaskSubset(testset, seen_classes), batch_size=128,
                                 shuffle=False, num_workers=8)
        overall = evaluate_partial(model, eval_loader, device, seen_classes)
        per_class = evaluate_per_class_partial(model, eval_loader, device, seen_classes)
        tracker.stop("evaluation")

        base_acc = float(np.mean([per_class.get(c, 0) for c in tasks[0]]))
        new_acc = float(np.mean([per_class.get(c, 0) for c in task_classes]))
        session_results.append({
            "session": task_idx, "is_base": is_base, "overall_acc": overall,
            "base_classes_acc": base_acc, "new_classes_acc": new_acc,
            "per_class_acc": per_class, "num_classes": len(seen_classes),
        })
        print(f"\n--- Session {task_idx}: Overall {overall:.2f}% | "
              f"Base {base_acc:.2f}% | New {new_acc:.2f}% ---")

    tracker.times["total"] = time.time() - t_start

    print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
    print(f"{'Session':<10}{'Type':<12}{'Overall':<10}{'Base':<10}{'New':<10}{'NumCls':<8}")
    print("-" * 62)
    for r in session_results:
        stype = "BASE" if r["is_base"] else f"INC-{r['session']}"
        print(f"{r['session']:<10}{stype:<12}"
              f"{r['overall_acc']:<10.2f}{r['base_classes_acc']:<10.2f}"
              f"{r['new_classes_acc']:<10.2f}{r['num_classes']:<8}")

    final = session_results[-1]["overall_acc"]
    base0 = session_results[0]["overall_acc"]
    avg = float(np.mean([r["overall_acc"] for r in session_results]))
    forget = session_results[0]["base_classes_acc"] - session_results[-1]["base_classes_acc"]
    print(f"\n--- Key Metrics ---")
    print(f"Final Accuracy (all {len(seen_classes)} classes): {final:.2f}%")
    print(f"Base Session Accuracy: {base0:.2f}%")
    print(f"Performance Drop: {base0 - final:.2f}%")
    print(f"Average Session Accuracy: {avg:.2f}%")
    print(f"Forgetting (Base Classes): {forget:.2f}%")

    sessions = [r["session"] for r in session_results]
    print(f"\nSession  " + "".join(f"{s:<6}" for s in sessions))
    print(f"SBP-PD   " + "".join(f"{r['overall_acc']:<6.1f}" for r in session_results))


    print(f"\nTotal wall-clock: {tracker.times['total']/3600:.2f} h")
    return {"session_results": session_results, "final_acc": final,
            "base_session_acc": base0, "avg_acc": avg, "forgetting": forget}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1993)
    p.add_argument("--base-epochs", type=int, default=130)
    p.add_argument("--incremental-iterations", type=int, default=25)
    p.add_argument("--initial-budget", type=float, default=0.85)
    p.add_argument("--freeze-mask", dest="freeze_mask_after_base", action="store_true",
                   help="ablation: pin the active subspace to the base-session set "
                        "instead of growing it each session")
    p.add_argument("--masked-imprint", dest="mask_prototypes", action="store_true",
                   help="ablation: zero inactive dims in the stored prototype instead "
                        "of storing the full centroid")
    p.add_argument("--no-partial-distance", action="store_true",
                   help="full-space control: same pipeline, distances over all dims")
    a = p.parse_args()
    run(seed=a.seed, base_epochs=a.base_epochs,
        incremental_iterations=a.incremental_iterations,
        initial_budget=a.initial_budget,
        use_partial_distance=not a.no_partial_distance,
        freeze_mask_after_base=a.freeze_mask_after_base,
        mask_prototypes=a.mask_prototypes)
