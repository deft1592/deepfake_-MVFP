# AAAI27 FREPDD-CLIP reproduction snapshot

This directory contains the FREPDD-CLIP training code, the reconstructed stage-1
experiment configuration, dataset metadata, and the scripts needed to reproduce
training.

## Experiments

`experiments/stage1_all.yaml` trains the complete FREPDD + CLIP model on
FaceForensics++ and validates it on Celeb-DF-v2 and Celeb-DF-v1.

## Included artifacts

- `experiments/stage1_all.yaml`: stage-1 training configuration.
- `run_stage1.sh`: complete training entry point.
- `training/pretrained/clip-vit-base-patch16/`: location for local PyTorch CLIP
  weights (weight files are excluded from Git).
- `preprocessing/dataset_json_v3/`: metadata for the three datasets used.

The packaged `detectors`, `networks` and `dataset` registries intentionally load
only FREPDD-CLIP dependencies. This avoids importing unrelated detectors that
require DECA and other multi-gigabyte assets; the actual experiment model,
trainer and dataset implementation files are unchanged copies.

The 74 GB LMDB image database is intentionally not duplicated. In the current
workspace, `datasets/lmdb` points to the parent DeepfakeBench dataset. After
moving this folder, configure another dataset location with:

```bash
./configure_data.sh /absolute/path/to/datasets/lmdb
```

## Environment

Use the repository's environment definition:

```bash
conda env create -f environment.yml
conda activate deepfake
```

The original runs used two GPUs through `torchrun`. Override the process count
or master port with `NPROC_PER_NODE` and `MASTER_PORT` if needed.

## Training

After configuring the dataset and environment, start training directly with:

```bash
chmod +x run_stage1.sh configure_data.sh
./run_stage1.sh
```

The script checks the dataset link and CLIP weights, then launches distributed
training with `torchrun`. It uses two GPUs by default. For example, to use one
GPU or change the rendezvous port:

```bash
NPROC_PER_NODE=1 MASTER_PORT=29502 ./run_stage1.sh
```

New runs are written under `logs/training/`. The best checkpoint is also copied
to `artifacts/checkpoints/stage1_reproduced_best.pth`. Both directories are
excluded from Git. Due to GPU kernels, library versions and data-loader
scheduling, exact floating-point equality is not guaranteed even with the
recorded random seed.

For discriminator ablations, see `DISCRIMINATOR_ABLATION.md` and the helper
scripts `run_discriminator_ablation.sh` / `run_all_discriminator_ablations.sh`.
