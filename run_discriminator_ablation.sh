#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

variant="${1:-}"
case "$variant" in
  none|mag|phase|radial|band|mag_phase|mag_radial|mag_band|phase_radial|phase_band|radial_band|wo_mag|wo_phase|wo_radial|wo_band|all) ;;
  *)
    echo "Usage: $0 {none|mag|phase|radial|band|mag_phase|mag_radial|mag_band|phase_radial|phase_band|radial_band|wo_mag|wo_phase|wo_radial|wo_band|all}" >&2
    exit 2
    ;;
esac

torchrun_bin="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
if [[ -z "$torchrun_bin" ]]; then
  echo "torchrun not found; activate the DeepfakeBench environment first." >&2
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

nproc="${NPROC_PER_NODE:-2}"
port="${MASTER_PORT:-29511}"
stage1_config="./experiments/ablation_stage1_${variant}.yaml"
stage1_target="frepdd_clip_vit_b16_ablation_${variant}"

"$torchrun_bin" \
  --nproc_per_node="$nproc" --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port="$port" \
  training/train_new.py \
  --detector_path "$stage1_config" \
  --task_target "$stage1_target" \
  --no-save_feat --ddp

stage1_run="$(find logs/training -maxdepth 1 -type d -name "frepddCg_${stage1_target}_*" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
stage1_checkpoint="$stage1_run/val/avg/ckpt_best.pth"
if [[ ! -f "$stage1_checkpoint" ]]; then
  echo "Stage-1 checkpoint not found under $stage1_run" >&2
  exit 1  
fi
mkdir -p artifacts/checkpoints
cp "$stage1_checkpoint" "artifacts/checkpoints/${stage1_target}_best.pth"
echo "Completed discriminator ablation: $variant"
echo "Best checkpoint: artifacts/checkpoints/${stage1_target}_best.pth"
