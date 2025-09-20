# Combined Dataset: Potholes and Trash Cans

## Dataset Information
- **Total Classes**: 4
- **Class 0**: Pothole
- **Class 1**: Trash Can - Closed
- **Class 2**: Trash Can - Empty  
- **Class 3**: Trash Can - Full

## Dataset Statistics
- **Training samples**: 2424
  - Pothole: 1440
  - Trash Cans: 984
- **Validation samples**: 284
  - Pothole: 120
  - Trash Cans: 164
- **Test samples**: 432

## Usage with YOLOv8
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(data='data.yaml', epochs=100, imgsz=640)
```

## Directory Structure
```
Combined_Dataset/
├── data.yaml
├── README.md
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Generated on: 2025-09-17 11:19:56
