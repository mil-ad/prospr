import gc
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import prospr
import utils
from datasets import dataloader_factory
from evaluate import evaluate
from models import model_factory
from prospr.utils import pruning_filter_factory


def get_pruned_model(model: nn.Module, hparams: utils.Hyperparameters):

    if torch.cuda.is_available() and not hparams.prune_on_cpu:
        model = model.cuda()
    else:
        model = model.cpu()

    train_dataloader, *_, num_classes = dataloader_factory(
        hparams.dataset, hparams.meta_batch_size
    )

    filter_fn = pruning_filter_factory(num_classes, hparams.structured_pruning)

    if not hparams.structured_pruning:

        return prospr.prune(
            model,
            hparams.prune_ratio,
            train_dataloader,
            filter_fn,
            hparams.inner_steps,
            hparams.inner_lr,
            hparams.inner_momentum,
            hparams.meta_grads_mode,
            hparams.structured_pruning,
            hparams.new_data_in_inner,
        )

    else:  # structured
        raise NotImplementedError("Coming soon!")


def get_optimizer(model: nn.Module, hparams: utils.Hyperparameters):

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hparams.lr,
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay,
    )

    if hparams.lr_milestones is not None:
        milestones = hparams.lr_milestones
    elif hparams.dataset == "imagenet":
        milestones = [
            int(hparams.epochs * 0.3),
            int(hparams.epochs * 0.6),
            int(hparams.epochs * 0.9),
        ]
    else:
        milestones = [int(hparams.epochs * 0.5), int(hparams.epochs * 0.75)]

    # lr_decay is typically a single float (a list of length 1) which means we apply the
    # same decay at every milestone using MultiStepLR. To be more flexible we allow it
    # to be a list of floats defining the decay at every milestone; in that case we have
    # to construct a function that computes the compound decay given the epoch.
    if len(hparams.lr_decay) > 1:
        assert len(hparams.lr_decay) == len(milestones)

        def compute_decay(epoch):
            num_decays = sum([epoch > milestone for milestone in milestones])
            return np.prod(hparams.lr_decay[:num_decays])

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, compute_decay)

    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=hparams.lr_decay[0]
        )

    return optimizer, lr_scheduler


def train_one_epoch(model, train_loader, optimizer):
    model.train()

    losses = []

    start_time = datetime.now()

    for x, y in tqdm(train_loader, leave=False):
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)

    end_time = datetime.now()
    return avg_loss, end_time - start_time


def train(hparams: utils.Hyperparameters):

    log_dir = utils.create_logdir(hparams.logroot)

    utils.save_repo_status(log_dir)
    utils.save_command_line(log_dir)

    utils.set_seed(hparams.seed, hparams.allow_nondeterminism)

    train_data, _, test_data, _ = dataloader_factory(
        hparams.dataset, hparams.batch_size
    )

    model = model_factory(hparams.model, hparams.dataset, hparams.no_model_patching)

    if hparams.prune_ratio > 0:
        model, masks = get_pruned_model(model, hparams)
        torch.save(masks, log_dir / "pruning_keep_mask.pt")

        gc.collect()
        torch.cuda.empty_cache()

    optimizer, lr_scheduler = get_optimizer(model, hparams)

    model = model.cuda()

    for epoch in range(1, hparams.epochs + 1):

        avg_train_loss, epoch_time = train_one_epoch(model, train_data, optimizer)

        test_loss, test_acc1, test_acc5 = evaluate(model, test_data)

        print(
            f"📸 Epoch {epoch} (finished in {epoch_time})\n",
            f"\tTrain loss:\t{avg_train_loss:.4f}\n",
            f"\tTest loss:\t{test_loss:.4f}\n",
            f"\tTest acc:\t{test_acc1:.4f}\n",
            f"\tTest top-5 acc:\t{test_acc5:.4f}",
        )

        lr_scheduler.step()

        if hparams.store_checkpoints and (epoch % hparams.checkpoint_interval == 0):
            utils.save_checkpoint(
                log_dir,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                hparams.max_checkpoints,
            )

    print(
        "✅ Training finished\n",
        f"\tFinal test acc: {test_acc1}\n",
        f"\tFinal test acc@5: {test_acc5}",
    )

    if hparams.store_checkpoints:
        p = log_dir / "trained_model.pt"
        torch.save(model.state_dict(), p)
