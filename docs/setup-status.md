# Local experiment setup

## Environment

Repository: `../DeepDataMiningLearning`, evaluated at commit `fa15745942ef5a27388a79f78c9c589e335189a1` (a local commit containing the Windows UTF-8 report fix).

Isolated environment: `../.venv`, Python 3.11. Installed CUDA PyTorch 2.10.0+cu126 and torchvision 0.25.0+cu126 using the [official PyTorch wheel index](https://pytorch.org/get-started/previous-versions/). CUDA availability and an actual GPU matrix multiplication passed on the RTX 2070. The original global Python environment remains CPU-only.

For a guided first run that explains each option and starts with one model on one dataset, see the [hands-on run guide](hands-on-run.md).

## Commands from the workspace root

Activate the environment in Git Bash from the workspace root. Then download missing data (safe to rerun; do not run concurrent download processes against the same folder):

```bash
source .venv/Scripts/activate
python scripts/download-data.py
```

This prepares 50 COCO validation images, the full COCO validation instance annotations, 50 KITTI training images, and KITTI training labels and calibration. Sources are COCO's `images.cocodataset.org` S3 bucket and the `avg-kitti` S3 URLs listed in upstream `Utils/kitti.py`. The COCO custom HTTPS hostname has a certificate mismatch, so the script uses the bucket's HTTPS S3 endpoint with certificate verification enabled.

Archives live in `data/downloads`; datasets live in `data/coco` and `data/kitti`. KITTI pilot mode reads the ZIP directory and selected images via HTTPS byte ranges, avoiding the full image archive. Use `--kitti-images 500` for a larger pilot or `--kitti-images 0` to download the full image archive and extract all training images. An interrupted full-archive attempt may leave a resumable `.part` file; subset mode does not need that file. Existing completed archives and extracted files are reused; `.part` downloads resume when the server supports byte ranges. KITTI manifests check file presence and sizes, and ZIP extraction checks member CRCs. These checks do not establish cryptographic authenticity or detect every same-size corruption.

To stage more COCO images, use `--coco-images 500`; use `0` for all 5,000 validation images. For KITTI, use `--kitti-images 500`; use `0` to fetch/extract all labeled training images. COCO selection uses sorted image IDs, matching torchvision's dataset ordering. A prefix is suitable for a smoke/pilot run, not a representative random sample. The [hands-on guide](hands-on-run.md) includes the full-data command and runtime estimate.

Verify model loading and GPU prediction (downloads weights on the first run):

```bash
python scripts/smoke-models.py
```

This uses a blank image to test execution only; detections and timing from it are not scientific results.

After learning the individual steps, run the six accuracy evaluations, downloading missing data automatically:

```bash
bash scripts/run-evaluation.sh --max-images 50
```

Use `--skip-download` when all required data is already staged, `--data-root /d/datasets` to choose another target, and `--run-name my-run` for a readable output directory. The runner saves the command arguments, dependency versions, repository commit, console log, and available dataset image lists in a new `results/<run-name>/` folder. It refuses to overwrite an existing run folder. For the Git Bash commands to explore models and datasets individually, use the [hands-on run guide](hands-on-run.md).

Upstream KITTI defaults to split `train` and would automatically generate an 80/20 split if `ImageSets` were absent. Our downloader instead writes `ImageSets/train.txt` with the selected sorted training image IDs, so all 50 downloaded images participate in the pilot. Report this as a fixed prefix subset of KITTI's labeled training partition. Increasing the requested image count updates this generated list. COCO uses val2017. No model fine-tuning occurs. Both subsets are exploratory rather than random or representative samples.

The accuracy runner does not measure inference latency or GPU peak memory. Separate single-image latency benchmarks were run and are saved under `../results/latency-coco` and `../results/latency-kitti`. These report end-to-end FP32 prediction timing for one image per dataset, not a dataset-wide latency distribution. GPU peak memory remains unmeasured. Do not use total evaluation runtime as inference latency.

## Verification status

- Repository clone completed; the local source checkout is one commit ahead of upstream due to the Windows UTF-8 report fix.
- CUDA environment and GPU operation passed.
- The interactive instructions and all-in-one launcher use Git Bash.
- Dependency installation and `pip check` passed; resolved versions are saved in `../requirements-lock.txt` (reinstall CUDA wheels from the CUDA index first).
- All three checkpoint loads and CUDA predictions passed; receipt: `../results/model-smoke.json`.
- Ten-image COCO evaluation passed for all three models and generated `../results/coco-smoke-verified/metrics.json`, `report.md`, and `heatmap_mAP.png`. These are execution checks, not final homework results.
- The first COCO attempt was interrupted when the sandbox blocked Hugging Face metadata checks. The successful run used cached weights with `HF_HUB_OFFLINE=1`. Normal network-enabled runs can use the default online mode.
- Downloader checks passed for existing-file skipping, selected extraction, repeated extraction, and rejection of archive traversal paths. Python compilation passed.
- Both datasets now have 50 pilot images. KITTI byte-range extraction passed; rerunning the final downloader completed without network access and skipped existing data. Resume and restart behavior were also checked with controlled responses.
- The previous all-in-one launcher completed six 50-image accuracy pilots successfully. Each result contains all 12 COCO statistics and per-class AP. See [pilot report](../results/pilot-50/report.md), with raw metrics and reproduction records in the same folder.

The full-data run is complete: 5,000 COCO `val2017` images and 7,481 labeled KITTI `training` images were evaluated for each model. See the [full report](../results/full-dataset-run/report.md), raw metrics, arguments, environment, evaluation log, code commit, and image ID lists in that folder. This is zero-shot evaluation; no fine-tuning was performed. The KITTI measurements use the labeled training partition and must not be described as official KITTI test scores.

| Model | COCO unified mAP | KITTI unified mAP |
|---|---:|---:|
| YOLO11n | 0.548 | 0.336 |
| YOLO11m | 0.622 | 0.445 |
| RT-DETR ResNet-50 | 0.630 | 0.449 |

These are historical 50-image prefix pilot results on the two-category `vehicle_person` taxonomy, not the full-data results.

Separate latency runs are available in `../results/latency-coco/latency_report.md` and `../results/latency-kitti/latency_report.md`. Each used one image and FP32 inference on the RTX 2070; treat the values as indicative. Peak GPU memory has not been measured. Fine-tuning has not been run.

Upstream evaluator inspection found that reconstructed ground truth uses box area and `iscrowd=0`. This drops native COCO segmentation areas/crowd semantics, in addition to the unified taxonomy changes. Label the results as the ngdet unified evaluation protocol; they are not official COCO or KITTI leaderboard scores.
