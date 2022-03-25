import os
import zipfile
from pathlib import Path
from typing import Optional

import requests
import torch
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm


def get_cifar10(root):
    num_classes = 10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root / "CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root / "CIFAR10", train=False, transform=test_transform, download=False
    )

    return num_classes, train_dataset, test_dataset


def get_cifar100(root):
    num_classes = 100

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train_dataset = datasets.CIFAR100(
        root / "CIFAR100", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_dataset = datasets.CIFAR100(
        root / "CIFAR100", train=False, transform=test_transform, download=False
    )

    return num_classes, train_dataset, test_dataset


def get_imagenet(root: Path, standardize=True):

    num_classes = 1000

    # NB for ImageNet we don't have access to the test dataset and instead use
    # validation set as test.

    train_transforms = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    val_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]

    if standardize:
        normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_transforms += [normalizer]
        val_transforms += [normalizer]

    if (root / "ImageNet_Torchvision").exists():
        train = datasets.ImageNet(
            root / "ImageNet_Torchvision",
            split="train",
            download=False,
            transform=transforms.Compose(train_transforms),
        )
        val = datasets.ImageNet(
            root / "ImageNet_Torchvision",
            split="val",
            download=False,
            transform=transforms.Compose(val_transforms),
        )
    elif (root / "imagenet").exists():
        train = datasets.ImageFolder(
            root / "imagenet" / "train",
            transform=transforms.Compose(train_transforms),
        )

        val = datasets.ImageFolder(
            root / "imagenet" / "val",
            transform=transforms.Compose(val_transforms),
        )
    else:
        raise FileNotFoundError

    return num_classes, train, val


def get_tiny_imagenet(root: Path):
    num_classes = 200

    if not (root / "tiny-imagenet-200").exists():
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        checksum = "90528d7ca1a48142e341f4ef8d21d0de"
        chunk_size = 1024

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            zip_file = root / "tiny-imagenet-200.zip"
            total_length = int(r.headers.get("content-length"))
            with tqdm(total=total_length, desc="tiny-imagenet-200.zip") as pbar:
                with zip_file.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)

        if not check_integrity(zip_file, checksum):
            raise OSError

        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(root)

        # Validation images are all in a single directory. We need to have a
        # sub-directory for each class so that we can use ImageFolder class. The train
        # directory already has this structure but validation does not because that'd be
        # too easy.
        val_dir = root / "tiny-imagenet-200/val"
        with (val_dir / "val_annotations.txt").open("r") as f:
            file_classdir = [(line.split("\t")[0:2]) for line in f.readlines()]

        for filename, class_dir in file_classdir:
            (val_dir / class_dir).mkdir(exist_ok=True)

            (val_dir / "images" / filename).rename(val_dir / class_dir / filename)

        (val_dir / "images").rmdir()
        zip_file.unlink()

    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2770, 0.2691, 0.2821],
    )

    train_dataset = datasets.ImageFolder(
        root / "tiny-imagenet-200" / "train",
        transforms.Compose(
            [
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_dataset = datasets.ImageFolder(
        root / "tiny-imagenet-200" / "val",
        transforms.Compose([transforms.ToTensor(), normalize]),
    )

    return num_classes, train_dataset, test_dataset


def dataloader_factory(
    dataset: str,
    batch_size=128,
    root: Path = Path.home() / "datasets",
    val_from_train: Optional[float] = None,
):

    dataset_getters = {
        "cifar10": get_cifar10,
        "cifar100": get_cifar100,
        "tiny_imagenet": get_tiny_imagenet,
        "imagenet": get_imagenet,
    }

    num_classes, train_dataset, test_dataset = dataset_getters[dataset](root)

    if dataset in ["imagenet", "imagenet_dogs", "imagenet_notdogs"]:
        kwargs = {
            "num_workers": int(os.getenv("SLURM_CPUS_ON_NODE", 6)),
            "pin_memory": True,
        }
    else:
        kwargs = {"num_workers": 4, "pin_memory": True}

    if val_from_train:
        val_len = int(len(train_dataset) * val_from_train)
        train_len = len(train_dataset) - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_len, val_len]
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
    else:
        val_loader = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, val_loader, test_loader, num_classes
