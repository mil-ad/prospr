import copy
import types
from math import ceil
from typing import Callable, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector


def pruning_filter_factory(num_classes: int, structured: bool):
    """
    By default we're pruning all nn.Linear and nn.Conv2d layers. In case of structured
    pruning we have to exclude the very last nn.Linear layer as its shape needs to match
    number of classes.
    """

    def filter_fn(m):
        if isinstance(m, nn.Conv2d):
            return True
        elif isinstance(m, nn.Linear):
            if structured and m.out_features == num_classes:
                return False
            else:
                return True

    return filter_fn


def attach_masks_as_parameter(
    net: nn.Module,
    filter_fn: Callable,
    structured: bool,
    gradient_tie: bool,
    masks_init_values: Optional[List[torch.Tensor]] = None,
    override_forward: bool = True,
    make_weights_constants: bool = True,
) -> List[torch.Tensor]:
    def masked_conv2d_fwd(self, x):
        return F.conv2d(
            x,
            self.weight * self.weight_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def masked_linear_fwd(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

    last_layer = None

    all_weight_masks = []

    if masks_init_values:
        masks_init_values = iter(masks_init_values)

    for name, layer in net.named_modules():
        if not filter_fn(layer):
            continue

        if structured:
            if gradient_tie and (
                "downsample" in name or ("n_block" in name and "conv2" in name)
            ):
                # tie the weight mask of conv and downsample layer
                layer.weight_mask = last_layer.weight_mask
            else:
                # Same channels, 1 all other dimensions
                shape = [layer.weight.shape[0]] + [1] * (layer.weight.ndim - 1)
                layer.weight_mask = nn.Parameter(
                    torch.ones(shape, device=layer.weight.device)
                )

        else:
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))

        if masks_init_values:
            layer.weight_mask.data[:] = next(masks_init_values)[:]

        all_weight_masks.append(layer.weight_mask)

        if make_weights_constants:
            layer.weight.requires_grad = False

        if "ds_block" in name or last_layer is None:
            last_layer = layer

        if override_forward:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(masked_conv2d_fwd, layer)
            elif isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(masked_linear_fwd, layer)
            else:
                raise TypeError

    return all_weight_masks


def _pre_forward_hook(m, x):
    if m.keep_mask.ndim == 1:
        m.weight.data = m.weight.data * m.keep_mask.view(-1, 1, 1, 1)
        if m.bias is not None:
            m.bias.data = m.bias.data * m.keep_mask
    else:
        m.weight.data = m.weight.data * m.keep_mask


def apply_masks_with_hooks(net, keep_masks, structured, filter_fn, return_clone=True):

    if return_clone:
        net = copy.deepcopy(net)

    prunable_layers = filter(filter_fn, net.modules())

    if isinstance(keep_masks, torch.Tensor):
        # keep masks need to be reshaped first
        pointer = 0
        keep_masks_list = []
        for layer in prunable_layers:
            if structured:
                num_param = layer.weight.shape[0]
                shape = layer.weight.shape[0]
            else:
                num_param = layer.weight.numel()
                shape = layer.weight.shape

            keep_masks_list.append(
                keep_masks[pointer : pointer + num_param].view(shape)
            )

            pointer += num_param
        assert pointer == keep_masks.numel(), "Didn't use all keep mask parameters"

        keep_masks = keep_masks_list

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        if structured:
            assert layer.weight.shape[0] == keep_mask.shape[0]
        else:
            assert layer.weight.shape == keep_mask.shape

        layer.register_buffer("keep_mask", keep_mask)
        layer.register_forward_pre_hook(_pre_forward_hook)

    return net


def keep_mask_from_scores(
    scores_vec, keep_ratio, uncat_ref: Optional[List[torch.Tensor]] = None
):
    num_to_keep = ceil(len(scores_vec) * keep_ratio)

    keep_mask = torch.zeros_like(scores_vec, dtype=torch.float32)
    _, keep_idx = torch.topk(scores_vec, num_to_keep)
    keep_mask[keep_idx] = 1.0

    if uncat_ref:
        keep_mask = uncat_as(keep_mask, uncat_ref)

    return keep_mask


def keep_masks_stats(keep_masks: Iterable[torch.Tensor]):

    keep_masks_vec = parameters_to_vector(keep_masks)
    total_params = keep_masks_vec.numel()
    kept_params = keep_masks_vec.sum()

    total_params_prune_ratio = 1 - kept_params / total_params
    total_params_compression_ratio = total_params / kept_params

    for idx, mask in enumerate(keep_masks):
        layer_params = mask.numel()
        layer_kept = mask.sum().int()
        layer_prune_ratio = 1 - layer_kept / layer_params

        print(
            f"Layer {idx:2} - Params: {layer_params:>10,} -> {layer_kept:>10,} "
            f"(pruned {layer_prune_ratio:2.2%})"
        )

    print(
        f"Overall:\n"
        f"\tParams: {total_params:,} -> {kept_params:,} "
        f"(pruned {total_params_prune_ratio:.2%} "
        f"- compressed: {total_params_compression_ratio:.2f}x)"
    )


def keep_masks_health_check(keep_masks):
    for mask in keep_masks:
        assert mask.bool().any(), "❌ Attempting to prune the entire layer!"


def uncat_as(x: torch.Tensor, recipe: Union[List[torch.Tensor], List[torch.Size]]):
    result = []

    processed = 0
    for ref in recipe:

        if isinstance(ref, torch.Size):
            shape = ref
        else:
            shape = ref.shape

        result.append(x[processed : processed + ref.numel()].reshape(shape))
        processed += ref.numel()

    assert processed == len(x)

    return result


def get_module_from_param_name(m: nn.Module, param_name: str):

    # TODO: Can we just use nn.get_submodule()?

    names = param_name.split(".")[:-1]

    # recursively go through modules
    for name in names:
        m = getattr(m, name)
    return m
