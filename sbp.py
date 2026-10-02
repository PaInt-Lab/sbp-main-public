import math
from typing import List, Dict, Optional, Set, Tuple, Union
import torch
import torch.nn as nn

__all__ = ["SBP"]

def _get_module_from_name(root: nn.Module, fullname: str) -> Tuple[nn.Module, str]:
    parts = fullname.split(".")
    mod = root
    for p in parts[:-1]:
        mod = getattr(mod, p)
    return mod, parts[-1]  

def _is_bn_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

def _is_bn_param(network: nn.Module, fullname: str) -> bool:
    mod, _ = _get_module_from_name(network, fullname)
    return _is_bn_module(mod)

def _is_norm_module(m: nn.Module) -> bool:
    return isinstance(m, (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.GroupNorm, nn.LayerNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
    ))

def _is_norm_param(network: nn.Module, fullname: str) -> bool:
    mod, _ = _get_module_from_name(network, fullname)
    return _is_norm_module(mod)


def _kaiming_init_subset(t: torch.Tensor, idx: torch.Tensor, generator: Optional[torch.Generator] = None):
    """
    Kaiming uniform initialization for a subset of tensor elements.
    """
    try:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(t)
    except ValueError:
        fan_in = t.numel()
    
    bound = (1.0 / math.sqrt(fan_in)) if fan_in > 0 else 1.0
    std = bound * math.sqrt(3.0)
    
    # Create random values on CPU (with generator), then move to target device
    random_values = torch.randn(idx.numel(), dtype=t.dtype, generator=generator) * std
    random_values = random_values.to(t.device)
    
    t.view(-1)[idx] = random_values

def _expand(mask_1d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """View (C,) mask as (C,1,1,…) matching `ref` for broadcasting."""
    return mask_1d.view(-1, *([1] * (ref.ndim - 1)))

class SBP:
    def __init__(
        self, 
        network: nn.Module, 
        budget_slice: float=0.05, 
        replay_ratio: float=0.0, 
        reset_assigned_weights: bool=False, 
        reset_free_weights: bool=True,
        replay_weight: float=0.0,
        device: torch.device = torch.device("cuda"), 
        classifier_params: Optional[Set[str]]=None, 
        task_to_class: Optional[Dict[int, List[int]]]=None
        ) -> None:
        
        self.network = network
        self.budget_slice = float(budget_slice)
        self.replay_ratio = float(replay_ratio)
        self.replay_weight = float(replay_weight)
        self.reset_assigned_weights = bool(reset_assigned_weights) # whether to reset assigned weights when starting a new task
        self.reset_free_weights = bool(reset_free_weights) # whether to reset free weights when starting a new task
        self.device = device
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(42)
        self._fwd_hooks: List = []  # handles for registered forward hooks
        
        self.classifier_params = classifier_params or set() # parameter names of classifier layer
        self.task_to_class = task_to_class or {} # map each task to its classes (5 per task for CIFAR-100)
        
        self.named_params: Dict[str, torch.Tensor] = {} # All named parameters in network
        self.frozen_mask: Dict[str, torch.Tensor] = {} # Mask for frozen weights
        self.assigned_mask: Dict[str, torch.Tensor] = {} # Mask for assigned weights
        self.free_mask: Dict[str, torch.Tensor] = {} # Mask for free weights
        self.forward_keep: Dict[str, torch.Tensor] = {} # float vector (1.0 or 0.0) for forward gating (so free neurons don't contribute to layer's output)
        self.current_task_id = 0
        self.epoch_in_task = 0
        
        self.register_parameters() # register parameters and initialize masks
        
    def set_replay(self, replay_ratio: float) -> None:
        self.replay_ratio = float(replay_ratio)
        self.replay_weight = float(replay_ratio)
        
    def _count(self, mask: torch.Tensor) -> int:
       return int(mask.sum().item())
        
    def is_classifier_output(self, param_name: str) -> bool:
        """Check if this parameter is a classifier output (neuron-level budgeting)."""
        return any(cp in param_name for cp in self.classifier_params)
        
    def register_parameters(self) -> None:
        print("Registering parameters for SBP budgeting algorithm...")
        new_map: Dict[str, torch.Tensor] = {} # temporary map to hold updated parameters
        
        # Iterate through all named parameters
        for name, param in self.network.named_parameters():
            # print("NAME:", name)
            if param.requires_grad: # only consider parameters that require gradients
                new_map[name] = param # store parameter in new map
                
        # Purge stale
        for dead in set(self.named_params) - set(new_map):
            for d in [
                self.named_params,
                self.frozen_mask,
                self.assigned_mask,
                self.free_mask,
                self.forward_keep,
            ]:
                d.pop(dead, None)
            
        for name, param in new_map.items():
            if name in self.named_params:
                continue
            
            # if _is_norm_param(self.network, name): 
            #     continue
            
            # if _is_bn_param(self.network, name):
            #     continue
        
            self.named_params[name] = param # add new parameter to named parameters list

            # Skip scalar parameters (like fc_out.scale)
            if param.ndim == 0:
                continue

            out_dim = param.size(0)  # output dimension of parameter

            self.frozen_mask[name] = torch.zeros(out_dim, dtype=torch.bool, device=self.device) # initialize frozen mask
            self.assigned_mask[name] = torch.zeros(out_dim, dtype=torch.bool, device=self.device) # initialize assigned mask
            self.free_mask[name] = torch.ones(out_dim, dtype=torch.bool, device=self.device) # initialize free mask
            self.forward_keep[name] = torch.ones(out_dim, dtype=torch.float32, device=self.device) # initialize forward keep mask

            param.register_hook(self._make_sbp_hook(name)) # register hook for SBP budgeting algorithm
        
        print("Budgeter parameters registered. Total backbone parameters:", len(self.named_params))
        # self.apply_forward_masks()
        
    def get_neuron_states(self, layer_or_param: str) -> torch.Tensor:
        """
        Return a 1D tensor of ints of length = out_features for this layer/param:
          0 = free, 1 = assigned, 2 = frozen

        `layer_or_param` can be either:
          - a full param name like "fc1.weight"
          - or a module name like "fc1" (we'll map it to "fc1.weight" or "fc1.bias").
        """
        name = layer_or_param

        # If we were given a layer name (e.g. "fc1"), map it to a real param key
        if name not in self.free_mask:
            # Prefer the weight tensor as the canonical neuron-level mask
            candidates = [layer_or_param + ".weight", layer_or_param + ".bias"]
            found = None
            for cand in candidates:
                if cand in self.free_mask:
                    found = cand
                    break
            if found is None:
                raise KeyError(
                    f"[SBP.get_neuron_states] "
                    f"No mask found for '{layer_or_param}'. Tried: {candidates}"
                )
            name = found  # use the param key we found

        free_mask     = self.free_mask[name]      # [out_features] bool
        assigned_mask = self.assigned_mask[name] # [out_features] bool
        frozen_mask   = self.frozen_mask[name]   # [out_features] bool

        # states: 0 = free, 1 = assigned, 2 = frozen
        states = torch.full_like(free_mask, -1, dtype=torch.long)
        states[free_mask] = 0
        states[assigned_mask] = 1
        states[frozen_mask] = 2

        return states

    
    def _make_sbp_hook(self, name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            assigned_mask_x = _expand(self.assigned_mask[name], grad)
            frozen_mask_x = _expand(self.frozen_mask[name], grad)
            free_mask_x = _expand(self.free_mask[name], grad)

            # 1. Assigned weights get full gradient
            assigned_grad = grad * assigned_mask_x

            # 2. Frozen weights: If Replay is OFF, this MUST be 0.0
            frozen_grad = grad * frozen_mask_x * self.replay_weight

            # 3. Free weights: MUST BE ZERO if you are re-initializing them
            #    Otherwise the optimizer wastes energy updating weights you will delete.
            free_grad = grad * free_mask_x * 0.0

            return assigned_grad + frozen_grad + free_grad
        return hook
            
        
    def _make_forward_hook(self, param_name: str):
        """
        Returns a forward hook that zeros free-channel activations in the module output.

        During normal (incremental) training the hook computes a keep-mask as
          keep = ~(frozen | assigned)   (i.e. NOT free)
        and multiplies the module's output by it, silencing every free channel.

        The mask is computed live from self.frozen_mask / self.assigned_mask so it
        always reflects the latest state without needing to re-register hooks.

          * self.current_task_id == 0           – base training; every neuron is live
        """
        def hook(_module, _input, output):
            if self.current_task_id == 0:
                return output
            if param_name not in self.frozen_mask:
                return output
            # Compute keep-mask on-the-fly so mask updates (new_task_update /
            # new_epoch_update) are reflected immediately without re-registering.
            free_mask = ~(self.frozen_mask[param_name] | self.assigned_mask[param_name])
            if not free_mask.any():
                return output  # all channels active – nothing to zero
            keep = (~free_mask).to(dtype=output.dtype, device=output.device)
            # Reshape to (1, C, 1, …) to broadcast over batch and spatial dims.
            keep = keep.view(1, -1, *([1] * (output.ndim - 2)))
            return output * keep
        return hook

    def apply_forward_masks(self) -> None:
        """
        Register (or re-register) forward hooks that zero free-channel activations.

        Only hooks Conv2d and Linear modules whose .weight param appears in
        self.free_mask.  Uses module-id deduplication so weight + bias of the
        same layer don't produce duplicate hooks.

        Safe to call multiple times – old hooks are removed before new ones
        are added.
        """
        for h in self._fwd_hooks:
            h.remove()
        self._fwd_hooks.clear()

        hooked: Set[int] = set()
        for name in self.free_mask:
            if not name.endswith(".weight"):
                continue
            try:
                mod, _ = _get_module_from_name(self.network, name)
            except AttributeError:
                continue
            if not isinstance(mod, (nn.Conv2d, nn.Linear)):
                continue
            if id(mod) in hooked:
                continue
            hooked.add(id(mod))
            h = mod.register_forward_hook(self._make_forward_hook(name))
            self._fwd_hooks.append(h)

        print(f"[ForwardMask] Registered forward hooks on {len(self._fwd_hooks)} modules.")

    def new_task_update(self, task_id: int, task_classes: List[int] = None,
                        importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> None:
        print(f"Updating masks for new task {task_id}...")
        
        self.current_task_id = task_id
        self.register_parameters()  # ensure parameters are registered
        
        # self.replay_weight = float(self.replay_ratio)  # reset the scalar used by the gradient hook to scale frozen gradients
        
        if self.current_task_id > 1:  # not 0-indexed, we start at task 1 (not task 0)
            print(f"SBP: Freezing Task {task_id-1} channels/neurons...")
            for n in self.assigned_mask:
                self.frozen_mask[n] = self.frozen_mask[n] | self.assigned_mask[n]  # freeze assigned weights from previous task
                self.assigned_mask[n].zero_()  # reset assigned mask for new task
        
        newly_assigned = 0
        chosen_by_layer = {}
        
        for name, param in self.network.named_parameters():
            # print(f"\nNAME: {name}")
            if name not in self.assigned_mask:
                continue
            
            layer = name.rsplit('.', 1)[0]
            # print(f"LAYER: {layer}")
            C = self.assigned_mask[name].numel()
            # print(f"Total neurons/channels: {C}")
            
            free_1d = ~(self.frozen_mask[name] | self.assigned_mask[name])
            free_1dx = free_1d.nonzero(as_tuple=True)[0]
            
            if layer in {'fc_out'}:  # Check if it's a classifier parameter 
                if task_classes is None:
                    print(f"[WARN] No task_classes provided for classifier {name}")
                    continue
                
                for class_id in task_classes:
                    self.assigned_mask[name][class_id] = True
                    stride = param.numel() // C
                    newly_assigned += stride
                
                if self.reset_assigned_weights:
                    for class_id in task_classes:
                        if class_id < C:
                            channel_slice = param.data[class_id].view(-1)
                            channel_idx = torch.arange(channel_slice.numel(), device=self.device)
                            _kaiming_init_subset(channel_slice, channel_idx, generator=self.rng)
            
            else:
                if layer not in chosen_by_layer:
                    k = round(self.budget_slice * C)
                    if task_classes:
                        k = max(len(task_classes), k)
                    k = min(max(1, k), free_1dx.numel())

                    if importance_scores is not None and name in importance_scores:
                        # Top-k free neurons by descending importance score
                        scores = importance_scores[name].to("cpu")
                        free_scores = scores[free_1dx]
                        top_k_idx = free_scores.argsort(descending=True)[:k]
                        chosen_by_layer[layer] = free_1dx[top_k_idx]
                    else:
                        chosen_by_layer[layer] = free_1dx[
                            torch.randperm(free_1dx.numel(), generator=self.rng, device="cpu")[:k]
                        ]
            
                # Optional: re-initialize assigned weights
                # Skip gain parameters - they should stay at 1.0
                is_gain_param = 'gain' in name and '.gain' in name
                is_skip_gain = 'skip_gain' in name

                if self.reset_assigned_weights and not is_gain_param and not is_skip_gain:
                    for c in chosen_by_layer[layer]:
                        # print(f"  Initializing channel/neuron: {c}")
                        channel_slice = param.data[c].view(-1)
                        channel_idx = torch.arange(channel_slice.numel(), device=self.device)
                        _kaiming_init_subset(channel_slice, channel_idx, generator=self.rng)
                
                chosen = chosen_by_layer[layer]
                self.assigned_mask[name][chosen] = True
                stride = param.numel() // C
                newly_assigned += chosen.numel() * stride

        print(f"\nSBP: Newly assigned {newly_assigned} parameters for Task {task_id}.")
        print(f"({self.budget_slice * 100:.1f}% of channels per layer).")
        self.epoch_in_task = 0

        
    def new_epoch_update(self) -> Dict[str, torch.Tensor]:
        print(f"SBP: Updating free masks for epoch {self.epoch_in_task} in Task {self.current_task_id}...")
        num_free = 0
        num_reset = 0
        
        for name, param in self.network.named_parameters():
            if name not in self.free_mask:
                continue
            
            if _is_norm_param(self.network, name):
                # do not mask, zero, or re-init GN/BN/LN/IN parameters
                continue
            
            if _is_norm_param(self.network, name):
                if self.current_task_id > 1:  # After base
                    param.requires_grad = False  # Hard freeze BN
                continue
            
            free_mask_1d = ~(self.frozen_mask[name] | self.assigned_mask[name]) # free weights only 
            self.free_mask[name] = free_mask_1d # update epoch free mask
            self.forward_keep[name] = (~free_mask_1d).float() # update forward keep mask
            num_free += int(free_mask_1d.sum().item())
            
            is_gain_param = 'gain' in name and '.gain' in name  # e.g., "conv.gain"
            is_skip_gain = 'skip_gain' in name

            # if self.reset_free_weights and free_mask_1d.any():
            if self.reset_free_weights and free_mask_1d.any() and not is_gain_param and not is_skip_gain:
                idx = free_mask_1d.nonzero(as_tuple=True)[0]
                # print("IDX: ", free_mask_1d)
                stride = param.numel() // free_mask_1d.numel()
                flat_idx = torch.cat([
                    torch.arange(c * stride, (c + 1) * stride, device=self.device)
                    for c in idx
                ])
                _kaiming_init_subset(param.data.view(-1), flat_idx, generator=self.rng)
                num_reset += int(flat_idx.numel())
            

        print(f"SBP: Epoch {self.epoch_in_task} in Task {self.current_task_id} - {num_free} free parameters.")
        
        self.epoch_in_task += 1
        return self.free_mask
    
    def state_dict(self):
        return {
            "frozen": {k: v.cpu() for k, v in self.frozen_mask.items()},
            "assigned": {k: v.cpu() for k, v in self.assigned_mask.items()},
            "current_task_id": self.current_task_id,
            "epoch_in_task": self.epoch_in_task,
            "cfg": {
                "budget_slice": self.budget_slice,
                "replay_ratio": self.replay_ratio,
                "reset_assigned_weights": self.reset_assigned_weights,
                "reset_free_weights": self.reset_free_weights,
            },
        }

    def load_state_dict(self, state):
        self.register_parameters()
        
        cfg = state.get("cfg", {})
        self.slice_pct = cfg.get("budget_slice", self.budget_slice)
        self.replay_ratio = cfg.get("replay_ratio", self.replay_ratio)
        self.replay_weight = float(self.replay_ratio)
        self.reset_assigned_weights = cfg.get("reset_assigned_weights", False)
        self.reset_free_weights = cfg.get("reset_free_weights", True)
        self.current_task_id = state.get("current_task_id", 0)
        self.epoch_in_task = state.get("epoch_in_task", 0)

        for bank, target in [
            (state["frozen"], self.frozen_mask),
            (state["assigned"], self.assigned_mask),
        ]:
            for n, m in bank.items():
                if n in target and m.shape == target[n].shape:
                    target[n].copy_(m.to(self.device))

        print(f"Budgeter: State loaded (task {self.current_task_id}, epoch {self.epoch_in_task}).")

    def print_mask_summary(self):
        """Print human-readable summary of mask allocation."""
        tot = sum(p.numel() for p in self.named_params.values())
        frozen = sum(
            self._count(m) * (self.named_params[n].numel() // m.numel()) 
            for n, m in self.frozen_mask.items()
        )
        assigned = sum(
            self._count(m) * (self.named_params[n].numel() // m.numel()) 
            for n, m in self.assigned_mask.items()
        )
        free = tot - frozen - assigned
        
        print(
            f"Mask Summary (Task {self.current_task_id}): "
            f"total={tot:,} | frozen={frozen:,} ({frozen/tot*100:.1f}%) | "
            f"assigned={assigned:,} ({assigned/tot*100:.1f}%) | "
            f"free={free:,} ({free/tot*100:.1f}%)"
        )

