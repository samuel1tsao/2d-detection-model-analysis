"""Download each selected checkpoint and verify one CUDA prediction; no accuracy claims."""
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DeepDataMiningLearning'))
from DeepDataMiningLearning.ngdet.detectors.base import build_detector
from DeepDataMiningLearning.ngdet.taxonomy import Taxonomy

if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this environment.')
tax = Taxonomy.from_preset('vehicle_person')
records = []
for spec in ('yolo:yolo11n.pt', 'yolo:yolo11m.pt', 'hf_detr:PekingU/rtdetr_r50vd'):
    detector = build_detector(spec, tax, device='cuda', score_thr=0.01)
    prediction = detector.predict(Image.new('RGB', (640, 480), (127, 127, 127)))
    torch.cuda.synchronize()
    records.append({'model': spec, 'detections': len(prediction), 'status': 'passed'})
    print(json.dumps(records[-1]), flush=True)
    del detector, prediction
    gc.collect()
    torch.cuda.empty_cache()
(ROOT / 'results').mkdir(exist_ok=True)
(ROOT / 'results' / 'model-smoke.json').write_text(json.dumps(records, indent=2))
