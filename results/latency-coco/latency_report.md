# ngdet latency benchmark

Device: **NVIDIA GeForce RTX 2070**. End-to-end `predict()` latency (includes pre/post-processing).

| model | accel | mean ms | p50 ms | p90 ms | FPS | speedup |
|---|---|---|---|---|---|---|
| yolo:yolo11n.pt | fp32 | 27.1 | 25.8 | 32.5 | 37.0 | x1.00 |
| yolo:yolo11m.pt | fp32 | 24.6 | 23.3 | 30.0 | 40.6 | x1.00 |
| hf_detr:rtdetr_r50vd | fp32 | 64.4 | 64.1 | 69.0 | 15.5 | x1.00 |