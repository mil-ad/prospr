# Prospect Pruning (ProsPr)

[![arXiv](https://img.shields.io/badge/arXiv-2202.08132-red)](https://arxiv.org/abs/2202.08132)
![PyTorch v1.9.1](https://img.shields.io/badge/PyTorch-v1.9.1-orange)
![license: MIT](https://img.shields.io/github/license/mil-ad/prospr)
[![Follow @notmilad](https://img.shields.io/twitter/follow/notmilad?style=social)](https://twitter.com/intent/follow?screen_name=notmilad)

The code for **"Prospect Pruning: Finding Trainable Weights at Initialization Using Meta-Gradients"**

## Installation

### 1️⃣ Reproducing results

You can replicate the development environment and use the same models and training script used in the paper with:

```
$ conda env create -f environment.yml
```

If you'd like to use the exact same package versions we used:

```
$ conda env create -f environment_pinned.yml
```


This will create the Conda environment `prospr`. The project's entry point is [`cli.py`](https://github.com/mil-ad/prospr/blob/main/cli.py)

To see the available options and switches:

```
$ python cli.py -h
```

### 2️⃣ As a package

You can also install and use ProsPr as a package inside your own projects:

```
$ pip install git+ssh://git@github.com/mil-ad/prospr.git
```

The `prospr` package can then be imported and used:

```py
import prospr

help(prospr)

pruned_model = prospr.prune(
    model,
    prune_ratio=0.98,
    dataloader=train_dataloader,
    filter_fn=prune_filter_fn,
    num_steps=3,
    inner_lr=0.5,
    inner_momentum=0.9,
)
```

## Citation

```tex
@article{alizadeh2022prospect,
  title = {Prospect Pruning: Finding Trainable Weights at Initialization using Meta-Gradients},
  author = {Alizadeh, Milad and Tailor, Shyam A. and Zintgraf, Luisa M and van Amersfoort, Joost and Farquhar, Sebastian and Lane, Nicholas Donald and Gal, Yarin},
  booktitle = {International Conference on Learning Representations},
  year = {2022}
}
```

[![back to top](https://img.shields.io/badge/back%20to%20top-%E2%86%A9-blue)](#prospect-pruning-prospr)
