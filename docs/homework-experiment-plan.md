# 2D detection homework: experiment setup

Status: all six full-data zero-shot evaluations are complete, and separate one-image-per-dataset FP32 latency benchmarks are recorded. Fine-tuning and GPU memory measurement remain outstanding. The current accuracy results use all 5,000 COCO `val2017` images and all 7,481 labeled KITTI `training` images. Target environment: Windows with an RTX 2070.

See [local setup status and executable commands](setup-status.md) for the cloned revision, isolated CUDA environment, automatic data downloader, and current verification status.

## Experiment matrix

Selected model sizes: YOLO nano and medium, plus RT-DETR (interpreting “DETR-RT” as RT-DETR). Use YOLO11 nano (`yolo:yolo11n.pt`), YOLO11 medium (`yolo:yolo11m.pt`), and RT-DETR ResNet-50 (`hf_detr:PekingU/rtdetr_r50vd`) as provisional checkpoints. The two YOLO sizes share a generation, making their size/speed comparison easier to interpret. Confirm the installed adapters accept all three checkpoints. Start with batch one on the local 8 GB GPU and verify memory use with a smoke test.

Use COCO 2017 validation and the labeled KITTI object-detection training partition, or a documented validation subset of that partition. Both have native 2D annotations. KITTI's unlabeled test partition is unsuitable for local ground-truth evaluation. This gives six evaluations: every model on both datasets, with no fine-tuning. COCO is a same-domain reference for these COCO-pretrained checkpoints; KITTI probes domain transfer. Do not describe this as a training comparison or an official KITTI leaderboard result: this harness uses COCO-style metrics and unified categories.

Initial inspection on 2026-10-24 found Python 3.11, an NVIDIA RTX 2070 with 8192 MiB, and driver 610.88. The original global environment had CPU-only torch and lacked pycocotools. Setup has since created a dedicated CUDA environment with the required evaluation dependencies; see setup status for verified versions. The workspace initially contained no project or data files.

## Setup sequence

1. Clone [DeepDataMiningLearning](https://github.com/lkk688/DeepDataMiningLearning) into this workspace, and record `git rev-parse HEAD`.
2. Create a dedicated Python environment. Follow the repository's current installation instructions and install its tutorial dependencies: torch, torchvision, transformers, ultralytics, pycocotools, opencv-python, matplotlib, and pyarrow. Select a PyTorch build compatible with the actual GPU and verify CUDA before downloading datasets.
3. Run from the cloned repository root. Check `python -m DeepDataMiningLearning.ngdet.run_eval --help` and `python -m DeepDataMiningLearning.ngdet.latency --help`; commands below are tutorial-based templates, not locally validated commands.
4. Supply COCO `val2017` images and `annotations/instances_val2017.json`. Supply KITTI object-detection left color images and training labels, arranged as `C:/data/kitti/training/image_2/` and `C:/data/kitti/training/label_2/`. Record the release, split, and exact image IDs selected. Verify the adapter's handling of KITTI DontCare regions, ignored categories, truncation, and occlusion; disclose differences from official KITTI evaluation.
5. Run a 10-image smoke test, then a 50-image pilot per dataset. For the final submission, use the full chosen split if feasible, or a fixed larger subset with its selection method and limitations disclosed. A prefix of 50 frames is exploratory and may be unrepresentative.

## Accuracy command template (Git Bash)

Run this after activating the workspace `.venv` and changing to the cloned repository directory (`cd DeepDataMiningLearning` from the workspace root). Git Bash paths use forward slashes.

Replace the two dataset roots and annotation path. Verify all flags against the installed revision first.

```bash
python -m DeepDataMiningLearning.ngdet.run_eval \
  --models yolo:yolo11n.pt yolo:yolo11m.pt hf_detr:PekingU/rtdetr_r50vd \
  --datasets coco kitti \
  --roots coco=/d/data/coco/val2017 kitti=/d/data/kitti \
  --coco-ann /d/data/coco/annotations/instances_val2017.json \
  --taxonomy vehicle_person \
  --max-images 50 --score-thr 0.01 --device cuda \
  --out-dir ../results/my-run
```

The low score cutoff is a proposed AP protocol: AP needs a ranked set of predictions, and a high cutoff can remove detections before the precision-recall curve is computed. Check each adapter for additional internal score filtering and detection caps. Keep settings fixed for the main comparison; inspect sensitivity separately if needed. Use a higher visualization cutoff only for displayed boxes, without changing evaluation predictions.

`vehicle_person` avoids the tutorial's documented bicycle-versus-rider box mismatch. Results then represent two unified categories, not standard COCO 80-class benchmark mAP. If the instructor requires all native categories or cyclist AP, change the protocol explicitly and disclose the mapping limitations. Never compare a unified-taxonomy score directly with a published native COCO score.

## Metrics and measurement

| Required signal | Record |
|---|---|
| mAP | AP averaged over IoU 0.50:0.05:0.95, maxDets=100 |
| AP50 / AP75 | AP at each individual IoU threshold |
| Size AP | small, medium, large using the evaluator's COCO area bins; confirm adapter ground-truth areas and coordinate conventions |
| Per-class AP | Every evaluated unified category, with ground-truth counts |
| AR | AR@1, AR@10, AR@100 and size-specific AR where available |
| Latency / FPS | Warmed batch-one mean, p50, p90 latency; FPS=1000/mean milliseconds |
| GPU memory | Peak allocated and peak reserved MiB, or clearly labeled externally sampled process memory |

Use N/A for undefined metrics (for example, a class/size group with no ground truth), not zero. State whether AP is represented on a 0–1 or 0–100 scale.

The tutorial writes accuracy results to `metrics.json` and `report.md`. Inspect actual output fields before extracting a table. Preserve outputs separately for each run so later runs cannot overwrite the evidence.

Latency template:

```bash
python -m DeepDataMiningLearning.ngdet.latency \
  --models yolo:yolo11n.pt yolo:yolo11m.pt hf_detr:PekingU/rtdetr_r50vd \
  --accel fp32 --iters 40 --device cuda \
  --image ../data/kitti/training/image_2/000008.png
```

Inspect this benchmark's image source, warmup count, resize policy, and timing scope. If it uses a synthetic/shared image, label it a separate model benchmark; it does not establish model-by-dataset speed. For per-dataset timings, instrument the evaluation prediction calls on the same saved image subset. Synchronize CUDA before and after timing, exclude model loading and downloads, state whether decoding/preprocessing/postprocessing are included, and use the same hardware, precision, and batch size. Repeat three times if feasible. Do not infer GPU FPS from CPU timings.

For memory, reset PyTorch peak memory statistics after model loading and warmup, synchronize after inference, and capture `max_memory_allocated()` and `max_memory_reserved()`. State that this measures PyTorch allocator memory and excludes some driver/external allocations. Measure each model in a fresh process where possible. If measurement is unavailable, report N/A with the reason.

## Reproducibility record for every model/dataset pair

- Experiment ID, timestamp, repository commit, exact command, and saved configuration.
- Backend, model name, checkpoint file/revision and SHA-256 where available; resolve aliases such as default pretrained weights to a specific weight identifier.
- Dataset release and split, annotation checksum, selected image IDs, image count, category counts, taxonomy and mapping.
- Input resize policy and actual tensor dimensions: models may use different defaults. Do not claim all models use 640x640 unless configured and verified. Native preprocessing is acceptable if reported as part of each pipeline.
- Confidence cutoff at every filtering stage, NMS IoU threshold or N/A where no NMS is used, internal proposal/detection limits, evaluator maxDets.
- GPU/VRAM, CPU, RAM, OS, driver, Python, torch/CUDA, torchvision, transformers, ultralytics, and pycocotools versions. Save `python -m pip freeze`.
- Precision, batch size, seeds/subset selection, warmup/iteration counts, latency scope, and memory measurement method.
- Output paths, failures, missing metrics, and any deviations from the protocol.

## Windows and Colab execution

Prefer Windows initially because the local GPU is already identified. Verify CUDA, install missing dependencies in an isolated environment, and run all six pilot evaluations on the same device. A GPU's presence alone does not establish that the installed PyTorch build can use it.

The original global Python environment has `torch==2.10.0+cpu` and cannot use the RTX 2070. The workspace `.venv` now has CUDA-enabled PyTorch 2.10.0+cu126; CUDA execution and all three model predictions passed. Use that environment for experiments.

If Windows package compatibility or memory becomes a blocker, use a Colab GPU runtime. Clone the same repository commit, install dependencies, and stage the same dataset subset on the runtime's local disk. In notebooks, prefix shell commands with `!` (or use a `%%bash` cell), and replace dataset paths with the Colab paths. Save outputs and environment details to persistent storage before the session ends. GPU availability and hardware vary, so record the actual assigned GPU. Compare speed/memory only within one hardware environment; if switching to Colab, rerun all three models there for the performance comparison.

## Submission tables

Use six rows, one per model/dataset pair. The full-data accuracy results are in [full-dataset-run/report.md](../results/full-dataset-run/report.md), and the separate latency results are in [latency-coco](../results/latency-coco/latency_report.md) and [latency-kitti](../results/latency-kitti/latency_report.md). Latency is a single-image indicative measurement, not a dataset-wide timing estimate. GPU memory is unmeasured and should be marked N/A. Put per-class AP and the other AR rows in a second table. Attach the reproducibility records and exact commands.

Discuss accuracy versus speed, which classes/sizes are difficult, and how results differ across datasets. Inspect several false positives and missed detections. Treat cross-dataset differences cautiously: class prevalence, object size, annotation rules, and sample selection also change. The KITTI scores are from its labeled training partition, not a held-out official test split. Fine-tuning requires a separate, fixed KITTI holdout and a matching pretrained baseline on that same holdout; do not compare a fine-tuned score on training images to the existing full-train score.

Source: [ngdet tutorial](https://github.com/lkk688/DeepDataMiningLearning/blob/main/DeepDataMiningLearning/ngdet/TUTORIAL.md), reviewed 2026-10-24. Source code is now cloned and the accuracy CLI has been verified. See setup status for evaluator limitations and the script that saves run metadata. The separate latency command requires `--image` with a real local image path; its upstream default points to the author's machine.
