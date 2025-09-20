#!/usr/bin/env python3
"""
Run inference with a YOLOv8 segmentation model and save predictions as YOLO-format label .txt files.
- Input: folder of images
- Output: labels directory mirroring image filenames (without extension)

For YOLOv8 segmentation, each line is: class cx cy w h p1x p1y p2x p2y ... (normalized)
Ultralytics automatically saves TXT labels when save_txt=True.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def predict_to_labels(weights: Path, source: Path, out_dir: Path | None = None, imgsz: int = 640, conf: float = 0.25):
    model = YOLO(str(weights))

    # Set output directory
    project = out_dir if out_dir else (source / 'predictions')
    name = 'labels'

    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        save=True,           # save prediction images
        save_txt=True,       # save labels as .txt in YOLO format
        save_conf=False,
        project=str(project),
        name=name,
        exist_ok=True,
        stream=False,
        device='cpu',
        task='segment'
    )

    # When save_txt=True, Ultralytics writes labels under
    # {project}/{name}/labels/*.txt corresponding to images in {project}/{name}/
    labels_path = Path(project) / name / 'labels'
    print(f"Saved YOLO labels to: {labels_path}")
    return labels_path


def main():
    parser = argparse.ArgumentParser(description='Predict masks and save YOLO labels (.txt)')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights .pt file')
    parser.add_argument('--source', type=str, required=True, help='Path to an image or directory of images')
    parser.add_argument('--out_dir', type=str, default='', help='Optional output base directory')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    weights = Path(args.weights)
    source = Path(args.source)
    out_dir = Path(args.out_dir) if args.out_dir else None

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    labels_dir = predict_to_labels(weights, source, out_dir, imgsz=args.imgsz, conf=args.conf)
    print(f"Done. Labels directory: {labels_dir}")

if __name__ == '__main__':
    main()
