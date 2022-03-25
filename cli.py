import argparse
import random
from pathlib import Path

import utils
from train import train


def parse_args():

    parser = argparse.ArgumentParser(
        description="Prospect Pruning (ProsPr): Finding Trainable Weights at "
        "Initialization Using Meta-Gradients"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "resnet20", "resnet50", "vgg19", "vgg16"],
    )

    parser.add_argument(
        "--no-model-patching",
        action="store_true",
        help="Disables automatic patching of ResNet and VGG models to work with "
        "input sizes smaller than ImageNet",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "cifar100", "tiny_imagenet", "imagenet"],
    )

    # Pruning Args
    pruning_parser = parser.add_argument_group("Pruning Args")

    pruning_parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.8,
        help="Pruning ratio [0 to 1) (default: %(default)s)",
    )

    pruning_parser.add_argument(
        "--structured-pruning",
        action="store_true",
        help="Structured pruning instead of pruning invididual parameters "
        "(default: %(default)s)",
    )

    pruning_parser.add_argument(
        "--prune-on-cpu",
        action="store_true",
        help="Do the pruning steps on CPU",
    )

    pruning_parser.add_argument(
        "--inner-steps",
        help="Number of steps in the inner loop (default: 3)",
        type=int,
        default=3,
    )

    pruning_parser.add_argument(
        "--inner-lr",
        type=float,
        default=0.1,
        help="Learning for the inner loop (default: 0.1)",
    )

    pruning_parser.add_argument(
        "--inner-momentum",
        type=float,
        default=0,
        help="SGD momentum for the inner loop (default: 0)",
    )

    pruning_parser.add_argument(
        "--meta-grads-mode",
        choices=["full", "first_order"],
        required=False,
        default="full",
        help="Whether to use the first-order approximation of ProsPr or the full "
        "computation graph (default: full)",
    )

    pruning_parser.add_argument(
        "--new-data-in-inner",
        default=False,
        help="Get a new batch of data in every step of the inner loop. Otherwise "
        "the batch from outer loop is used (default: False)",
    )

    pruning_parser.add_argument(
        "--meta-batch-size",
        type=int,
        default=128,
        help="Batch size for ProsPr's training steps (default: %(default)s)",
    )

    training_group = parser.add_argument_group("Training Hyper-parameters")

    training_group.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train (default: %(default)s)",
    )

    training_group.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: %(default)s)",
    )

    training_group.add_argument("--lr", type=float, default=0.1, help="Learning rate")

    training_group.add_argument(
        "--lr-milestones",
        required=False,
        nargs="+",
        type=int,
        default=None,
        help="LR decay milestones (default: set based on dataset)",
    )

    training_group.add_argument(
        "--lr-decay",
        nargs="+",
        type=float,
        default=[0.1],
        help="Multiplicative factor of learning rate decay. It can be either a "
        "single float that will be used at all milestones or a list of floats "
        "specifying decay rate at each milestone. (default: %(default)s)",
    )

    training_group.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum for the main training loop (default: %(default)s)",
    )

    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay (default: 5e-4)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(1, 1e3),
        help="Random seed (default: random)",
    )

    parser.add_argument(
        "--allow-nondeterminism",
        action="store_true",
        help="disables CUDA/cuDNN determinism",
    )

    parser.add_argument(
        "--logroot",
        default=Path.cwd(),
        type=Path,
    )

    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument("--store-checkpoints", action="store_true")

    parser.add_argument("--checkpoint-interval", type=int, default=10)

    parser.add_argument("--max-checkpoints", type=int, default=1)

    return parser.parse_args()


def cli():
    args = parse_args()
    hparams = utils.Hyperparameters(**vars(args))
    train(hparams)


if __name__ == "__main__":
    cli()
