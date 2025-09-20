#!/usr/bin/env python3
"""
Watch for best.pt from training and then run batch prediction on validation images to save YOLO .txt labels.
"""
import time
import subprocess
from pathlib import Path

BASE = Path(r"d:\SIH 2025\SIH DATASETS\model\Combined_Dataset")
WEIGHTS = BASE / "runs" / "segment" / "combined_dataset_training" / "weights" / "best.pt"
SOURCE = BASE / "valid" / "images"
OUT_DIR = BASE / "inference_valid"
SCRIPT = BASE / "predict_to_labels.py"

print("Watcher started. Waiting for best.pt to be created by training...")
print(f"Weights path: {WEIGHTS}")
print(f"Source images: {SOURCE}")

# Poll for best.pt
while not WEIGHTS.exists():
    time.sleep(10)

print("best.pt detected. Starting batch prediction on validation images...")
cmd = [
    "python",
    str(SCRIPT),
    "--weights", str(WEIGHTS),
    "--source", str(SOURCE),
    "--out_dir", str(OUT_DIR),
    "--imgsz", "640",
    "--conf", "0.25",
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=False)
print("Batch prediction complete with return code:", result.returncode)
print("Labels saved under:", OUT_DIR / "labels")
