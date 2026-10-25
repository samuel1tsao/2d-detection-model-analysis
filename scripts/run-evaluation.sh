#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run-evaluation.sh [--data-root PATH] [--max-images N] [--skip-download] [--run-name NAME]

Run all three models on COCO and KITTI after preparing missing data.
Uses this workspace's .venv, regardless of the global Python selected by Git Bash.
EOF
}

workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="$workspace/data"
max_images=50
skip_download=0
run_name="$(date +%Y%m%d-%H%M%S)"

while (($#)); do
  case "$1" in
    --data-root) data_root="$(cd -- "$2" && pwd)"; shift 2 ;;
    --max-images) max_images="$2"; shift 2 ;;
    --skip-download) skip_download=1; shift ;;
    --run-name) run_name="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ "$max_images" =~ ^[0-9]+$ ]] || { echo '--max-images must be a nonnegative integer' >&2; exit 2; }
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo '--run-name may contain only letters, numbers, underscores, and hyphens' >&2; exit 2; }

python_bin="$workspace/.venv/Scripts/python.exe"
[[ -f "$python_bin" ]] || { echo "Workspace Python not found: $python_bin" >&2; exit 1; }
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("GPU:", torch.cuda.get_device_name(0))'

if (( ! skip_download )); then
  "$python_bin" "$workspace/scripts/download-data.py" --data-root "$data_root" --coco-images "$max_images" --kitti-images "$max_images"
fi

for path in "$data_root/coco/annotations/instances_val2017.json" "$data_root/coco/val2017" "$data_root/kitti/training/image_2" "$data_root/kitti/training/label_2"; do
  [[ -e "$path" ]] || { echo "Missing dataset path: $path" >&2; exit 1; }
done

output="$workspace/results/$run_name"
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
mkdir -p "$output"
repo="$workspace/DeepDataMiningLearning"
data_win="$(cygpath -m "$data_root")"
out_win="$(cygpath -m "$output")"
ann_win="$(cygpath -m "$data_root/coco/annotations/instances_val2017.json")"

"$python_bin" -m pip freeze > "$output/environment.txt"
git -C "$repo" rev-parse HEAD > "$output/repository-commit.txt"
[[ ! -f "$data_root/coco/downloaded-image-ids.json" ]] || cp "$data_root/coco/downloaded-image-ids.json" "$output/coco-downloaded-image-ids.json"

args=(
  -m DeepDataMiningLearning.ngdet.run_eval
  --models yolo:yolo11n.pt yolo:yolo11m.pt hf_detr:PekingU/rtdetr_r50vd
  --datasets coco kitti
  --roots "coco=${data_win}/coco/val2017" "kitti=${data_win}/kitti"
  --coco-ann "$ann_win"
  --taxonomy vehicle_person
  --max-images "$max_images"
  --score-thr 0.01
  --device cuda
  --out-dir "$out_win"
)
printf '%s\n' "${args[@]}" > "$output/arguments.txt"
(cd "$repo" && "$python_bin" "${args[@]}") 2>&1 | tee "$output/evaluation.log"
[[ ! -f "$data_root/kitti/ImageSets/train.txt" ]] || cp "$data_root/kitti/ImageSets/train.txt" "$output/kitti-train-ids.txt"
echo "Results saved under: $output"
