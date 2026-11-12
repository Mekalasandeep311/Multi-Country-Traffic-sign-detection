import sys, time, numpy as np, torch
from ultralytics import YOLO
dev = sys.argv[1]
torch.set_num_threads(max(1, torch.get_num_threads()))
img = (np.random.rand(288, 512, 3) * 255).astype(np.uint8)
for name in ("yolo11n.yaml", "yolo11s.yaml", "yolo11s-p2.yaml", "yolo11m.yaml"):
    m = YOLO(name)
    m.model.nc = 159
    for _ in range(5): m.predict(img, imgsz=(288, 512), device=dev, verbose=False, half=(dev != "cpu"))
    t = []
    for _ in range(30):
        s = time.perf_counter(); m.predict(img, imgsz=(288, 512), device=dev, verbose=False, half=(dev != "cpu")); t.append(time.perf_counter() - s)
    print(f"{dev:>4} {name:<16} median {1000*np.median(t):6.1f} ms  p95 {1000*np.percentile(t,95):6.1f} ms", flush=True)
