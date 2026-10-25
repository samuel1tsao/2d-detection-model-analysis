# ngdet evaluation report

Unified taxonomy: **vehicle_person** = `['vehicle', 'person']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco | kitti |
|---|---|---|
| yolo:yolo11n.pt | 0.471 | 0.303 |
| yolo:yolo11m.pt | 0.581 | 0.349 |
| hf_detr:PekingU/rtdetr_r50vd | 0.578 | 0.366 |

## Summary matrix — AP50

| model \ dataset | coco | kitti |
|---|---|---|
| yolo:yolo11n.pt | 0.686 | 0.593 |
| yolo:yolo11m.pt | 0.790 | 0.652 |
| hf_detr:PekingU/rtdetr_r50vd | 0.805 | 0.669 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model | kitti |
|---|---|
| yolo:yolo11n.pt | +0.168 |
| yolo:yolo11m.pt | +0.232 |
| hf_detr:PekingU/rtdetr_r50vd | +0.212 |

## Full COCO tables (per run)

### yolo:yolo11n.pt  ×  coco  (5000 imgs, 214.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.471 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.686 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.500 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.186 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.486 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.702 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.194 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.514 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.593 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.353 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.649 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.776 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.439 |
| person | 0.502 |

### yolo:yolo11n.pt  ×  kitti  (7481 imgs, 369.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.303 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.593 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.279 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.097 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.278 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.458 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.142 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.389 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.444 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.287 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.444 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.557 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.399 |
| person | 0.207 |

### yolo:yolo11m.pt  ×  coco  (5000 imgs, 182.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.581 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.790 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.630 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.317 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.615 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.771 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.217 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.611 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.682 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.477 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.739 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.829 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.561 |
| person | 0.601 |

### yolo:yolo11m.pt  ×  kitti  (7481 imgs, 336.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.349 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.652 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.341 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.172 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.334 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.466 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.156 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.446 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.477 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.388 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.474 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.554 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.460 |
| person | 0.239 |

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (5000 imgs, 514.9s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.578 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.805 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.620 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.327 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.605 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.768 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.217 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.611 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.710 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.529 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.746 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.855 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.563 |
| person | 0.593 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (7481 imgs, 809.9s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.366 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.669 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.357 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.198 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.350 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.488 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.159 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.457 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.511 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.448 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.502 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.579 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.492 |
| person | 0.241 |
