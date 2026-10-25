# 2D Detection Evaluation

Homework project comparing pretrained YOLO11n, YOLO11m, and RT-DETR ResNet-50 on COCO 2017 validation and KITTI's labeled training split. The models were evaluated without fine-tuning using the repository's two-class `vehicle_person` mapping.

## Results

The full-data accuracy run evaluated 5,000 COCO validation images and 7,481 KITTI labeled training images. The report contains mAP@[.50:.95], AP50/AP75, AP by object size, AR, and per-class AP:

- [Full accuracy report](results/full-dataset-run/report.md)
- [Raw metrics](results/full-dataset-run/metrics.json)
- [Run arguments, environment, code revision, logs, and evaluated image IDs](results/full-dataset-run/)

| Model | COCO mAP | KITTI mAP |
|---|---:|---:|
| YOLO11n | 0.471 | 0.303 |
| YOLO11m | 0.581 | 0.349 |
| RT-DETR ResNet-50 | 0.578 | 0.366 |

Separate FP32 latency benchmarks were run on one COCO image and one KITTI image each. They measure end-to-end prediction latency, including preprocessing and postprocessing; each result is a single-image benchmark and should be treated as indicative:

- [COCO-image latency report](results/latency-coco/latency_report.md)
- [KITTI-image latency report](results/latency-kitti/latency_report.md)

GPU memory was not measured. Fine-tuning has not been run. KITTI uses its labeled `training` partition as the evaluation data; these are not official KITTI benchmark test scores. Results use the unified two-class evaluation protocol and are not standard 80-class COCO scores.

## Reproduce

See the [Git Bash setup and hands-on commands](docs/hands-on-run.md), [experiment plan and reporting notes](docs/homework-experiment-plan.md), and [local environment and verification record](docs/setup-status.md). The workspace `.venv` and downloaded data are intentionally excluded from version control; follow the setup guide to recreate them.
