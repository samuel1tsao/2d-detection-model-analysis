# Run the detection experiments yourself in Git Bash

The workspace and 50-image pilot data are ready. Work through one model and one dataset at a time so you can see which settings produce each result. The earlier [50-image pilot report](../results/pilot-50/report.md) is a reference run; use your own output folders for your experiments.

## 1. Open Git Bash in the workspace

```bash
cd "/c/Users/YOUR_USERNAME/OneDrive/Documents/workspace/2D Detection Evaluation"
source .venv/Scripts/activate
```

Confirm Bash is using the workspace environment and that COCO's evaluator is installed:

```bash
python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python -c 'import pycocotools; print(pycocotools.__file__)'
```

The executable path must contain `/2D Detection Evaluation/.venv/Scripts/python.exe`, followed by `2.10.0+cu126`, `True`, and `NVIDIA GeForce RTX 2070`. `pycocotools` should also load from `.venv/Lib/site-packages`.

Your traceback showed `C:/Users/YOUR_USERNAME/AppData/Local/Programs/Python/Python311/python.exe`, which is the global Python. If the check still prints that path, invoke the workspace interpreter directly in the commands below: use `./.venv/Scripts/python.exe` from the workspace root and `../.venv/Scripts/python.exe` after changing into `DeepDataMiningLearning`. This uses the already-installed dependencies without changing the global Python.

## 2. Look at the data before running a model

COCO images are in `data/coco/val2017`, with annotations in `data/coco/annotations/instances_val2017.json`. KITTI images and labels are in `data/kitti/training/image_2` and `data/kitti/training/label_2`. The selected image IDs are saved in `data/coco/downloaded-image-ids.json` and `data/kitti/downloaded-image-ids.json`.

The current 50-image sets let you try a short run. For a final comparison, using all the labeled data is preferable when you have the disk space and time. COCO 2017 validation has 5,000 images. KITTI provides 7,481 labeled training images; its 7,518 official test images have no public ground truth for local scoring. The official KITTI benchmark describes these counts [here](https://www.cvlibs.net/datasets/kitti/eval_object.php); the COCO 2017 validation split is listed as 5,000 images in the [COCO dataset catalog](https://www.tensorflow.org/datasets/catalog/coco).

Downloading all KITTI object-detection data takes about 11.71 GiB according to the [TensorFlow Datasets KITTI catalog](https://www.tensorflow.org/datasets/catalog/kitti). The workspace had about 53.5 GB free when these instructions were updated. The KITTI image archive also contains unlabeled test images; the downloader extracts the labeled training images for this experiment. COCO validation images are fetched individually, so downloading all 5,000 takes longer than the 50-image pilot.

For a larger intermediate subset, inspect the options, then fetch 200 images per dataset:

```bash
./.venv/Scripts/python.exe scripts/download-data.py --help
./.venv/Scripts/python.exe scripts/download-data.py --coco-images 200 --kitti-images 200
```

Existing files are skipped. COCO uses ascending image IDs and KITTI uses ascending filenames. These prefixes make quick pilots, not random representative samples.

To download every COCO validation image and every labeled KITTI training image, use `0` for both counts:

```bash
./.venv/Scripts/python.exe scripts/download-data.py --coco-images 0 --kitti-images 0
```

This can take a while; the KITTI downloader gets all training images from its archive, and COCO fetches 5,000 files. It will skip the pilot files already on disk.

## 3. Run YOLO11 nano on COCO

Enter the cloned repository. The command below evaluates 20 images and saves output to a fresh folder:

```bash
cd DeepDataMiningLearning
../.venv/Scripts/python.exe -m DeepDataMiningLearning.ngdet.run_eval \
  --models yolo:yolo11n.pt \
  --datasets coco \
  --roots coco=../data/coco/val2017 \
  --coco-ann ../data/coco/annotations/instances_val2017.json \
  --taxonomy vehicle_person \
  --max-images 20 \
  --score-thr 0.01 \
  --device cuda \
  --out-dir ../results/your-yolo11n-coco
```

The weights download on first use. `--max-images 20` evaluates the first 20 entries in the dataset's order. `vehicle_person` combines recognized COCO categories into `vehicle` and `person`. Keep the score cutoff at `0.01` for the comparison; the harness applies it after each model's own post-processing.

Open `../results/your-yolo11n-coco/report.md` and `metrics.json`. Find mAP, AP50, AP75, size AP, AR, and per-class AP. Values are fractions from 0 to 1. In your homework notes, record the command and what the results tell you.

If evaluation finishes but report generation fails, first check whether `metrics.json` exists. You can regenerate the report and heatmap without rerunning the model:

```bash
../.venv/Scripts/python.exe -m DeepDataMiningLearning.ngdet.report \
  --metrics ../results/your-yolo11n-coco/metrics.json \
  --taxonomy vehicle_person
```

The report writer saves text as UTF-8, including on Windows.

## 4. Change the dataset, then the model

For the same model on KITTI, change the dataset, root, and output folder:

```bash
../.venv/Scripts/python.exe -m DeepDataMiningLearning.ngdet.run_eval \
  --models yolo:yolo11n.pt \
  --datasets kitti \
  --roots kitti=../data/kitti \
  --taxonomy vehicle_person \
  --max-images 20 \
  --score-thr 0.01 \
  --device cuda \
  --out-dir ../results/your-yolo11n-kitti
```

Then change the model and output folder to compare YOLO11 medium on COCO:

```bash
../.venv/Scripts/python.exe -m DeepDataMiningLearning.ngdet.run_eval \
  --models yolo:yolo11m.pt \
  --datasets coco \
  --roots coco=../data/coco/val2017 \
  --coco-ann ../data/coco/annotations/instances_val2017.json \
  --taxonomy vehicle_person \
  --max-images 20 \
  --score-thr 0.01 \
  --device cuda \
  --out-dir ../results/your-yolo11m-coco
```

Continue with these model/dataset pairs, keeping each output folder unique:

| Model | COCO output folder | KITTI output folder |
|---|---|---|
| YOLO11n | `your-yolo11n-coco` | `your-yolo11n-kitti` |
| YOLO11m | `your-yolo11m-coco` | `your-yolo11m-kitti` |
| RT-DETR R50 | `your-rtdetr-coco` | `your-rtdetr-kitti` |

For the remaining runs, reuse one of the commands above and update the model spec (`yolo:yolo11m.pt` or `hf_detr:PekingU/rtdetr_r50vd`), dataset/root arguments, and `--out-dir`. KITTI uses no `--coco-ann` argument. Use a new output folder each time.

## 5. Save the run details

From the repository directory, these commands record the environment and exact repository revision beside your results. Repeat with the output folder for each experiment:

```bash
../.venv/Scripts/python.exe -m pip freeze > ../results/your-yolo11n-coco/environment.txt
git rev-parse HEAD > ../results/your-yolo11n-coco/repository-commit.txt
```

The 20-image commands above are just a quick first run: they confirm that a model loads, the dataset paths work, and you can read its output. Twenty images are too few for a strong final comparison, and the selected prefix may overrepresent some scenes or classes. After you've understood the run, use the complete available labeled splits for your final measurements. In every model/dataset command, replace `--max-images 20` with `--max-images 0`. This evaluates all 5,000 COCO validation images and all 7,481 labeled KITTI training images. Use distinct output folders and keep the other settings fixed.

Expect full runs to take much longer than the pilot. In the 50-image pilot, each model/dataset run took about 3–14 seconds including setup. A rough linear estimate for all six full-dataset runs is around 85 minutes on this PC; actual time may differ because of startup, caching, and image sizes. You can first run one model on one full dataset and read its saved `seconds` value in `metrics.json` to estimate your own remaining runs.

The script can also run the full matrix after you have tried the individual commands:

```bash
bash scripts/run-evaluation.sh --max-images 0 --run-name full-dataset-run
```

For KITTI, this scores its labeled training images; the official test partition has no public labels for local evaluation. Record the split and image IDs.

The reports use ngdet's `vehicle_person` COCO-style evaluation. The values are not standard 80-class COCO or official KITTI benchmark scores. Native COCO crowd flags/segmentation areas and some KITTI annotation rules are simplified by this harness. Latency/FPS and peak GPU memory need a separate measurement; command total runtime is not inference latency.

## Optional: run all six from Git Bash

Once you've worked through the individual commands, the launcher can run every model on both datasets. It explicitly uses the workspace `.venv`, even if Git Bash's `python` points at your global installation:

```bash
cd "/c/Users/YOUR_USERNAME/OneDrive/Documents/workspace/2D Detection Evaluation"
source .venv/Scripts/activate
bash scripts/run-evaluation.sh --max-images 50 --run-name my-pilot-50
```
