#!/usr/bin/env python3
"""
Watch for best.pt and evaluate the model on the test split defined in data.yaml.
Saves metrics, plots, and summary in the Ultralytics runs directory.
"""
from pathlib import Path
import time
from ultralytics import YOLO

BASE = Path(r"d:\SIH 2025\SIH DATASETS\model\Combined_Dataset")
WEIGHTS = BASE / "runs" / "segment" / "combined_dataset_training" / "weights" / "best.pt"
DATA_YAML = BASE / "data.yaml"

print("Test evaluator started. Waiting for best.pt ...")
print(f"Weights: {WEIGHTS}")
print(f"Data: {DATA_YAML}")

while not WEIGHTS.exists():
    time.sleep(10)

print("best.pt found. Starting test evaluation...")
model = YOLO(str(WEIGHTS))
results = model.val(
    data=str(DATA_YAML),
    split='test',          # use the test images defined in data.yaml
    project='runs/segment',
    name='combined_dataset_test_eval',
    exist_ok=True,
    device='cpu',
    plots=True,
    save_json=True,
)

print("Test evaluation finished.")
print(f"Results saved to: {results.save_dir}")
