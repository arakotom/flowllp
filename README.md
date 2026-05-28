# FlowLLP

Minimal research code for FlowLLP-style learning from label proportions, with an end-to-end image pipeline for MNIST/CIFAR10.

## What Is In This Repository

- Main runnable script: `flowLLP_end2end.py`
- Core utilities and models: `utils.py`, `models.py`
- Additional experimental scripts: `flowLLP.py`, `bagloss.py`

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `torch` and `torchvision` should be installed in compatible versions for your CUDA/CPU setup.
- This code auto-detects CUDA with PyTorch (`torch.cuda.is_available()`).

## Required Checkpoint

`flowLLP_end2end.py` loads the file `resnet18-f37072fd.pth` from the repository root.

- Keep this file in the same directory as `flowLLP_end2end.py` when running.
- If missing, execution will fail at model loading.

## Quick Start (Smoke Test)

Run a short MNIST experiment to verify the pipeline and outputs:

```bash
python flowLLP_end2end.py \
	--data mnist \
	--img_size 64 \
	--bag_size 100 \
	--num_epochs_bagloss 1 \
	--num_epochs_prop 10 \
	--num_epochs_all 1 \
	--s 0
```

This will download MNIST automatically (if not already present) into `./data`.

## Typical Runs

MNIST:

```bash
python flowLLP_end2end.py --data mnist --img_size 64 --s 0
```

CIFAR10:

```bash
python flowLLP_end2end.py --data cifar10 --img_size 64 --s 0
```

Tabular pipeline (`flowLLP.py`):

```bash
python flowLLP.py
```

## Main Arguments

Key options from `flowLLP_end2end.py`:

- `--data`: dataset (`mnist` or `cifar10`)
- `--bag_size`: number of samples per bag
- `--nb_class_in_bag`: classes allowed in each bag
- `--num_epochs_bagloss`: bag-loss pretraining epochs
- `--num_epochs_prop`: anchor/propagation epochs
- `--num_epochs_all`: final joint training epochs
- `--lr_bagloss`, `--lr_prop`, `--lr_all`: stage learning rates
- `--lmbd_bagloss`, `--lmbd_anchor`: final loss weights
- `--s`: random seed

Print all options:

```bash
python flowLLP_end2end.py --help
```

## Outputs

By default, outputs are written to:

- Results: `./results/end2end/{data}_{img_size}/single/`
- Models: `./models/{data}_{img_size}/`

The script prints the full output filename prefix at startup.

## Reproducibility Notes

- The script sets deterministic flags for PyTorch/CUDA based on `--s`.
- For paper-level reproduction, run multiple seeds and report mean/std.
- Runtime can be long with default epochs (`num_epochs_prop=3000`).

## Current Repository Caveats

- `flowLLP_end2end.py` and `flowLLP.py` are runnable in this repository.
- `bagloss.py` may depend on additional data helpers depending on your local setup.

## Citation

If you use this code, please cite the corresponding FlowLLP paper.
