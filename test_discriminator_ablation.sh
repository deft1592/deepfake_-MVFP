#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

usage() {
  cat >&2 <<'EOF'
Usage: ./test_discriminator_ablation.sh ABLATION_TYPE [MODE] [LOG_FILE]

MODE:
  standard (default), gan, wdf_c40

ABLATION_TYPE:
  none, mag, phase, radial, band,
  mag_phase, mag_radial, mag_band,
  phase_radial, phase_band, radial_band,
  wo_mag, wo_phase, wo_radial, wo_band, all

The wdf_c40 mode evaluates WDF and FaceForensics++ c40. Optional variables:
  NPROC_PER_NODE  Number of GPUs (default: 2)
  MASTER_PORT     torchrun port (default: 29521)
  CHECKPOINT_PATH Override the default checkpoint path
  OUTPUT_ROOT     Result root (default: ./ablation_test_results)
  LOG_FILE        Log file path; overridden by the optional second argument
  DATASET_JSON_FOLDER  Directory containing the four dataset JSON files
  GAN_DATASET_ROOT     GAN dataset root (default: ../GAN_generated_dataset)
EOF
}

variant="${1:-}"
mode="standard"
case "${2:-}" in
  standard|gan|wdf_c40) mode="$2" ;;
  "") ;;
  *) mode="standard" ;;
esac
case "$variant" in
  none|mag|phase|radial|band|mag_phase|mag_radial|mag_band|phase_radial|phase_band|radial_band|wo_mag|wo_phase|wo_radial|wo_band|all) ;;
  *)
    usage
    exit 2
    ;;
esac

torchrun_bin="${TORCHRUN_BIN:-$(command -v torchrun || true)}"
if [[ -z "$torchrun_bin" ]]; then
  echo "torchrun not found; activate the DeepfakeBench environment first." >&2
  exit 1
fi
if [[ "$mode" == "standard" && ! -e datasets/lmdb ]]; then
  echo "Missing LMDB dataset. Run: ./configure_data.sh /path/to/datasets/lmdb" >&2
  exit 1
fi

config="./experiments/ablation_stage1_${variant}.yaml"
checkpoint="${CHECKPOINT_PATH:-./artifacts/checkpoints/frepdd_clip_vit_b16_ablation_${variant}_best.pth}"
if [[ ! -f "$config" ]]; then
  echo "Ablation config not found: $config" >&2
  exit 1
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  echo "Train it first with: ./run_discriminator_ablation.sh $variant" >&2
  exit 1
fi

dataset_json_folder="${DATASET_JSON_FOLDER:-$root_dir/preprocessing/dataset_json_v3}"
gan_dataset_root="${GAN_DATASET_ROOT:-$root_dir/../GAN_generated_dataset}"
required_datasets=(Celeb-DF-v1 Celeb-DF-v2 DFDC DFDCP)
if [[ "$mode" == "standard" ]]; then
  json_files_complete=true
  for dataset in "${required_datasets[@]}"; do
    if [[ ! -f "$dataset_json_folder/$dataset.json" ]]; then
      json_files_complete=false
      break
    fi
  done
  if [[ "$json_files_complete" == false && -z "${DATASET_JSON_FOLDER:-}" ]]; then
    dataset_json_folder="$root_dir/../preprocessing/dataset_json_v3"
  fi
  for dataset in "${required_datasets[@]}"; do
    if [[ ! -f "$dataset_json_folder/$dataset.json" ]]; then
      echo "Dataset index not found: $dataset_json_folder/$dataset.json" >&2
      echo "Set DATASET_JSON_FOLDER to the complete dataset_json_v3 directory." >&2
      exit 1
    fi
  done
elif [[ "$mode" == "gan" ]]; then
  for dataset in biggan cyclegan stylegan stylegan2; do
    if [[ ! -d "$gan_dataset_root/$dataset" ]]; then
      echo "GAN dataset directory not found: $gan_dataset_root/$dataset" >&2
      exit 1
    fi
    for label_dir in 0_real 1_fake; do
      if ! find "$gan_dataset_root/$dataset" -type d -name "$label_dir" -print -quit | grep -q .; then
        echo "Missing $label_dir directory under $gan_dataset_root/$dataset" >&2
        exit 1
      fi
    done
  done
else
  if [[ "$variant" != "mag" ]]; then
    echo "The wdf_c40 protocol requires ABLATION_TYPE=mag." >&2
    exit 2
  fi
  config="./experiments/ablation_test_mag_wdf_ffpp_c40.yaml"
  if [[ ! -f "$dataset_json_folder/FaceForensics++.json" && -z "${DATASET_JSON_FOLDER:-}" ]]; then
    dataset_json_folder="$root_dir/../preprocessing/dataset_json_v3"
  fi
  if [[ ! -f "$dataset_json_folder/FaceForensics++.json" ]]; then
    echo "Dataset index not found: $dataset_json_folder/FaceForensics++.json" >&2
    exit 1
  fi
  if [[ ! -d "$root_dir/../WDF/real_test/images" || ! -d "$root_dir/../WDF/fake_test/images" ]]; then
    echo "WDF image directories not found under $root_dir/../WDF" >&2
    exit 1
  fi
fi

export TOKENIZERS_PARALLELISM=false
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

nproc="${NPROC_PER_NODE:-2}"
port="${MASTER_PORT:-29521}"
output_dir="${OUTPUT_ROOT:-./ablation_test_results}/${variant}"
if [[ "$mode" == "gan" ]]; then
  output_dir+="/gan"
elif [[ "$mode" == "wdf_c40" ]]; then
  output_dir+="/wdf_c40"
fi
mkdir -p "$output_dir"
timestamp="$(date '+%Y%m%d_%H%M%S')"
if [[ "$mode" != "standard" ]]; then
  log_file="${3:-${LOG_FILE:-$output_dir/test_${variant}_${mode}_${timestamp}.log}}"
else
  log_file="${2:-${LOG_FILE:-$output_dir/test_${variant}_${timestamp}.log}}"
fi
mkdir -p "$(dirname "$log_file")"

{
  echo "Ablation type: $variant"
  echo "Checkpoint: $checkpoint"
  if [[ "$mode" == "gan" ]]; then
    echo "Datasets: BigGAN, CycleGAN, StyleGAN, StyleGAN2"
    echo "GAN dataset root: $gan_dataset_root"
  elif [[ "$mode" == "wdf_c40" ]]; then
    echo "Datasets: WDF, FaceForensics++ c40"
    echo "Dataset indexes: $dataset_json_folder"
  else
    echo "Datasets: CDF-V1, CDF-V2, DFDC, DFDCP"
    echo "Dataset indexes: $dataset_json_folder"
  fi
  echo "Output: $output_dir"
  echo "Log file: $log_file"

  test_args=(
    --detector_path "$config"
    --weights_path "$checkpoint"
    --test_data_split test
    --output_dir "$output_dir"
  )
  if [[ "$mode" == "gan" ]]; then
    test_args+=(
      --gan_dataset_root "$gan_dataset_root"
      --test_dataset GAN_biggan GAN_cyclegan GAN_stylegan GAN_stylegan2
    )
  elif [[ "$mode" == "wdf_c40" ]]; then
    test_args+=(
      --test_dataset WDF FaceForensics++_c40
      --dataset_json_folder "$dataset_json_folder"
      --lmdb
    )
  else
    test_args+=(
      --test_dataset Celeb-DF-v1 Celeb-DF-v2 DFDC DFDCP
      --dataset_json_folder "$dataset_json_folder"
      --lmdb
    )
  fi

  "$torchrun_bin" \
    --nproc_per_node="$nproc" --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="$port" \
    training/test.py \
    "${test_args[@]}"

  echo "Completed discriminator ablation test: $variant"
  echo "Metrics and predictions: $output_dir"
} 2>&1 | tee "$log_file"
