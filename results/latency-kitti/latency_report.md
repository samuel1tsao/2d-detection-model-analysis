# ngdet latency benchmark

Device: **NVIDIA GeForce RTX 2070**. End-to-end `predict()` latency (includes pre/post-processing).

| model | accel | mean ms | p50 ms | p90 ms | FPS | speedup |
|---|---|---|---|---|---|---|
| yolo:yolo11n.pt | fp32 | 20.2 | 19.0 | 25.6 | 49.6 | x1.00 |
| yolo:yolo11m.pt | fp32 | 22.2 | 21.5 | 26.4 | 45.1 | x1.00 |
| hf_detr:rtdetr_r50vd | fp32 | 69.9 | 69.3 | 73.7 | 14.3 | x1.00 |