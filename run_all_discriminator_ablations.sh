#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

for variant in \
  none \
  mag phase radial band \
  mag_phase mag_radial mag_band phase_radial phase_band radial_band \
  wo_mag wo_phase wo_radial wo_band \
  all; do
  ./run_discriminator_ablation.sh "$variant"
done
