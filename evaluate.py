from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import eval_mode

__all__ = ["evaluate"]


def _accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy specified values of k (copied from PyTorch source code)
    """
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            res.append(correct[:k].reshape(-1).float().sum(0, keepdim=True))

        return res


def evaluate(
    model: nn.Module, dataloader: torch.utils.data.DataLoader
) -> Tuple[float, float]:

    losses = []
    total_correct1 = 0
    total_correct5 = 0

    with eval_mode(model), torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)

            losses.append(F.cross_entropy(pred, y).item())
            correct1, correct5 = _accuracy(pred, y, topk=(1, 5))
            total_correct1 += correct1
            total_correct5 += correct5

    avg_loss = sum(losses) / len(losses)
    acc1 = total_correct1 / len(dataloader.dataset)
    acc5 = total_correct5 / len(dataloader.dataset)

    return avg_loss, acc1.item(), acc5.item()
