#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/datasets/lmdb" >&2
  exit 1
fi

source_dir="$(readlink -f "$1")"
if [[ ! -d "$source_dir" ]]; then
  echo "LMDB directory does not exist: $source_dir" >&2
  exit 1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root_dir/datasets"
if [[ -L "$root_dir/datasets/lmdb" ]]; then
  unlink "$root_dir/datasets/lmdb"
elif [[ -e "$root_dir/datasets/lmdb" ]]; then
  echo "$root_dir/datasets/lmdb already exists and is not a symlink." >&2
  exit 1
fi
ln -s "$source_dir" "$root_dir/datasets/lmdb"
echo "Configured dataset link: $root_dir/datasets/lmdb -> $source_dir"

