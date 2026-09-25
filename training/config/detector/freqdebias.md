# FreqDebias reproduction

The implementation follows the CVPR 2025 paper's ResNet-34 backbone,
Fo-Mixup, confidence sampling, CAM consistency, and vMF hyperspherical
consistency. The paper trains a vanilla FF++ selector for 30 epochs before the
50-epoch FreqDebias run.

## Paper protocol

First train the selector. With two DDP processes, a per-process batch of 16
gives the paper's global batch size of 32:

```bash
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29500 \
  training/train.py \
  --detector_path ./training/config/detector/resnet34.yaml \
  --train_dataset "FaceForensics++" \
  --test_dataset "FaceForensics++" \
  --epochs 30 \
  --train_batch_size 16 \
  --ddp \
  > freqdebias_selector_output.log 2>&1
```

Locate that run's `ckpt_best.pth`, then start FreqDebias:

```bash
SELECTOR_CHECKPOINT=./logs/training/resnet34_<timestamp>/test/FaceForensics++/ckpt_best.pth \
  bash train.sh
```

Running `bash train.sh` without `SELECTOR_CHECKPOINT` is supported for an
end-to-end experiment; OHEM then uses detached current-model predictions.

## Public-paper ambiguities

No official implementation is public, and the arXiv source does not contain
the supplementary material referenced by the paper. The main paper omits the
frequency partition count and CAM K-means details, and writes amplitude noise
as `N(1, 0)`. These choices are documented and configurable in
`freqdebias.yaml`; the defaults use 8 radial by 8 angular segments, two CAM
clusters, and zero amplitude-noise variance.
