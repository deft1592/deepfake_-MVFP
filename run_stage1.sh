#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

torchrun_bin="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
if [[ -z "$torchrun_bin" ]]; then
  echo "torchrun not found; activate the DeepfakeBench environment first." >&2
  exit 1
fi

if [[ ! -f training/pretrained/clip-vit-base-patch16/pytorch_model.bin ]]; then
  echo "Missing CLIP weights under training/pretrained/clip-vit-base-patch16." >&2
  exit 1
fi
if [[ ! -e datasets/lmdb ]]; then
  echo "Missing LMDB dataset. Run: ./configure_data.sh /path/to/datasets/lmdb" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"$torchrun_bin" \
  --nproc_per_node="${NPROC_PER_NODE:-2}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29501}" \
  training/train_new.py \
  --detector_path ./experiments/stage1_all.yaml \
  --task_target frepdd_clip_vit_b16_all_repro \
  --no-save_feat \
  --ddp

latest_run="$(find logs/training -maxdepth 1 -type d -name 'frepddCg_frepdd_clip_vit_b16_all_repro_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -n "$latest_run" && -f "$latest_run/val/avg/ckpt_best.pth" ]]; then
  mkdir -p artifacts/checkpoints
  cp "$latest_run/val/avg/ckpt_best.pth" artifacts/checkpoints/stage1_reproduced_best.pth
  echo "Stage-1 best checkpoint: artifacts/checkpoints/stage1_reproduced_best.pth"
fi
