import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


def create_logdir(root: Union[str, Path] = None):

    if (root is None) or (root == ""):
        root = Path.cwd()
    else:
        root = Path(root)

    # When running multiple jobs in parallel (e.g. Slurm) we could get the same
    # timestamp so let's allow ourselves to try a few times
    for _ in range(10):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")

            log_dir = root / "runs" / timestamp

            log_dir.mkdir(parents=True)
        except FileExistsError:
            sleep(1)
            continue
        else:
            break
    else:
        raise SystemExit("Could not create logdir.")

    return log_dir


def save_repo_status(path: Union[str, Path]):
    path = Path(path)

    with (path / "git_commit.txt").open("w") as f:
        subprocess.run(["git", "rev-parse", "HEAD"], stdout=f)

    with (path / "workspace_changes.diff").open("w") as f:
        subprocess.run(["git", "diff"], stdout=f)


def save_command_line(path: Union[str, Path]):
    path = Path(path)

    with open(path / "command_line.txt", "w") as f:
        f.write("python " + " ".join(sys.argv))


def set_seed(seed: int, allow_nondeterminism: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if allow_nondeterminism is False:
        # This can make the training slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unconcatenate(x: torch.Tensor, orig_list: List[torch.Tensor]):
    result = []

    processed = 0
    for ref in orig_list:
        result.append(x[processed : processed + ref.numel()].reshape(ref.shape))
        processed += ref.numel()

    return result


def save_checkpoint(
    logdir,
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    max_checkpoints=None,
):

    state = {
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }

    p = logdir / f"chkpt_epoch_{epoch}.pt"
    torch.save(state, p)

    if max_checkpoints:
        chkpts = sorted(logdir.glob("chkpt_e[0-9]*.pt"), key=os.path.getmtime)
        num_unwanted_chckpts = len(chkpts) - max_checkpoints
        if num_unwanted_chckpts > 0:
            for c in chkpts[0:num_unwanted_chckpts]:
                c.unlink()


def load_checkpoint(
    path: Union[Path, str],
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
):

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError

    print(f"🛻 Loading from checkpoint file {path}.")

    chkpt = torch.load(path)

    model.load_state_dict(chkpt["model"])
    print("✅ Loaded the model.")

    optimiser.load_state_dict(chkpt["optimiser"])
    print("✅ Loaded the optimiser.")

    lr_scheduler.load_state_dict(chkpt["lr_scheduler"])
    print("✅ Loaded the LR scheduler.")


@contextmanager
def eval_mode(model: nn.Module):
    """
    Sets training mode to False and restores it when exiting.
    """
    is_training = model.training
    try:
        model.eval()
        yield model
    finally:
        if is_training:
            model.train()


class Hyperparameters:
    def __init__(self, **kwargs):
        self.from_dict(kwargs)

    def from_argparse(self, args):
        self.from_dict(args.__dict__)

    def from_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__dict__}

    def from_json(self, j):
        d = json.loads(j)
        return self.from_dict(d)

    def to_json(self, path: Path):
        j = json.dumps(self.as_dict(), indent=4, sort_keys=True)
        path.write_text(j)

    def __contains__(self, k):
        return k in self.__dict__

    def __str__(self):
        s = [f"{k}={v}" for k, v in self.as_dict().items()]
        return ",".join(s)
