# FlowLLP

Research code for FlowLLP-style learning from label proportions.

This repository includes:

- An end-to-end image pipeline for MNIST/CIFAR10.
- A tabular pipeline script used for FlowLLP experiments.
- Model and training utilities used by both pipelines.

## Repository Contents

- `flowLLP_end2end.py`: end-to-end image experiments (MNIST/CIFAR10) with bag-loss pretraining, anchor learning, and final fine-tuning.
- `flowLLP.py`: tabular FlowLLP experiment script.
- `bagloss.py`: bag-level baseline and bagging utilities.
- `models.py`, `utils.py`: neural network definitions and training/evaluation helpers.
- `requirements.txt`: Python dependencies.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- Use compatible `torch` and `torchvision` versions for your CUDA/CPU environment.
- GPU is used automatically when available.

## Required Files

`flowLLP_end2end.py` requires the ImageNet-pretrained ResNet18 checkpoint file `resnet18-f37072fd.pth` in the repository root.

If this file is missing, the script will fail when loading the backbone.

## Running Experiments

### 1) End-to-End Image Pipeline

Smoke test (fast sanity check):

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

Typical MNIST run:

```bash
python flowLLP_end2end.py --data mnist --img_size 64 --s 0
```

Typical CIFAR10 run:

```bash
python flowLLP_end2end.py --data cifar10 --img_size 64 --s 0
```

Datasets are loaded via torchvision and downloaded into `./data` when needed.

### 2) Tabular Pipeline

```bash
python flowLLP.py
```

This script is configured directly in code (no CLI flags) and currently uses `dry_beans` defaults in the `__main__` block.

## Main CLI Arguments (flowLLP_end2end.py)

Use:

```bash
python flowLLP_end2end.py --help
```

Important flags:

- `--data`: `mnist` or `cifar10`.
- `--img_size`: image resize resolution.
- `--bag_size`: number of samples per bag.
- `--nb_class_in_bag`: number of classes allowed per bag.
- `--num_epochs_bagloss`: bag-loss pretraining epochs.
- `--num_epochs_prop`: anchor-learning epochs.
- `--num_epochs_all`: final fine-tuning epochs.
- `--lr_bagloss`, `--lr_prop`, `--lr_all`: learning rates for each stage.
- `--lmbd_bagloss`, `--lmbd_anchor`: loss weights in final training.
- `--s`: random seed.

## Outputs

For `flowLLP_end2end.py`, outputs are created automatically:

- Results files (`.npz`): `./results/end2end/{data}_{img_size}/single/`
- Saved model weights (`.pth`): `./models/{data}_{img_size}/`

The script prints the exact result file path at startup and updates it during training.

`flowLLP.py` primarily reports metrics to stdout and does not write the same structured result artifacts by default.

## Reproducibility

- Both main scripts set deterministic seeds and CuDNN flags.
- For robust reporting, run multiple seeds and aggregate mean/std.
- Default settings can be long, especially anchor learning (`num_epochs_prop=3000` in `flowLLP_end2end.py`).

## Notes on Data Helpers

`flowLLP.py` and `bagloss.py` import functions from a module named `data`.

If your environment already provides that module (as in your current setup), these scripts run normally.
If not, add the corresponding project data helper module to your Python path.

## Citation

If this code is useful in your work, please cite the corresponding FlowLLP paper.
