#!/usr/bin/env python3
"""
YOLOv8 Training Script for Combined Pothole and Trash Cans Dataset
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_yolov8():
    """Train YOLOv8 model on the combined dataset"""
    
    print("=== YOLOv8 Training for Combined Dataset ===")
    print("Dataset: Potholes + Trash Cans")
    print("Classes: 4 (Pothole, Trash_Can_Closed, Trash_Can_Empty, Trash_Can_Full)")
    
    # Get the current directory (where this script is located)
    current_dir = Path(__file__).parent
    data_yaml = current_dir / 'data.yaml'
    
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at {data_yaml}")
        return
    
    print(f"Using data configuration: {data_yaml}")
    
    # Load a segmentation model - choose size
    model_options = {
        'n': 'yolov8n-seg.pt',  # Nano - fastest, least accurate (segmentation)
        's': 'yolov8s-seg.pt',  # Small - good balance (segmentation)
        'm': 'yolov8m-seg.pt',  # Medium - better accuracy (segmentation)
        'l': 'yolov8l-seg.pt',  # Large - high accuracy (segmentation)
        'x': 'yolov8x-seg.pt'   # Extra Large - highest accuracy, slowest (segmentation)
    }
    
    # Default to small model - good balance of speed and accuracy
    model_size = 's'
    model_path = model_options[model_size]
    
    print(f"Loading YOLOv8{model_size.upper()}-SEG model: {model_path}")
    model = YOLO(model_path)
    
    # Training parameters
    training_params = {
        'data': str(data_yaml),
        'epochs': 100,          # Number of training epochs
        'imgsz': 640,           # Image size
        'batch': 16,            # Batch size (adjust based on your GPU memory)
        'device': 'cpu',        # Force CPU since no CUDA device is available
        'workers': 8,           # Number of data loading workers
        'patience': 50,         # Early stopping patience
        'save': True,           # Save checkpoints
        'save_period': 10,      # Save checkpoint every N epochs
        'cache': False,         # Cache images for faster training (uses more RAM)
        'project': 'runs/segment',  # Project directory for segmentation
        'name': 'combined_dataset_training',  # Experiment name
        'exist_ok': True,       # Overwrite existing experiment
        'pretrained': True,     # Use pretrained weights
        'optimizer': 'auto',    # Optimizer (auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp)
        'verbose': True,        # Verbose output
        'seed': 0,              # Random seed for reproducibility
        'deterministic': True,  # Deterministic training
        'single_cls': False,    # Train as single class (set to True if you want to treat all as one class)
        'rect': False,          # Rectangular training
        'cos_lr': False,        # Cosine learning rate scheduler
        'close_mosaic': 10,     # Disable mosaic augmentation for final epochs
        'resume': False,        # Resume training from last checkpoint
        'amp': True,            # Automatic Mixed Precision training
        'fraction': 1.0,        # Dataset fraction to use
        'profile': False,       # Profile ONNX and TensorRT speeds during training
        'freeze': None,         # Freeze layers: backbone=10, first3=0,1,2
        'lr0': 0.01,            # Initial learning rate
        'lrf': 0.01,            # Final learning rate (lr0 * lrf)
        'momentum': 0.937,      # SGD momentum/Adam beta1
        'weight_decay': 0.0005, # Optimizer weight decay
        'warmup_epochs': 3.0,   # Warmup epochs
        'warmup_momentum': 0.8, # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        # Detection/pose-specific hyperparameters removed for segmentation task
        'label_smoothing': 0.0, # Label smoothing
        'nbs': 64,              # Nominal batch size
        'hsv_h': 0.015,         # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,           # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,           # Image HSV-Value augmentation (fraction)
        'degrees': 0.0,         # Image rotation (+/- deg)
        'translate': 0.1,       # Image translation (+/- fraction)
        'scale': 0.5,           # Image scale (+/- gain)
        'shear': 0.0,           # Image shear (+/- deg)
        'perspective': 0.0,     # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,          # Image flip up-down (probability)
        'fliplr': 0.5,          # Image flip left-right (probability)
        'mosaic': 1.0,          # Image mosaic (probability)
        'mixup': 0.0,           # Image mixup (probability)
        'copy_paste': 0.0,      # Segment copy-paste (probability)
    }
    
    print("\nTraining Parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    print("This may take a while depending on your hardware and dataset size.")
    print("Training progress will be saved in the 'runs/segment/combined_dataset_training' directory.")
    
    try:
        # Train the model
        results = model.train(**training_params)
        
        print("\n=== Training Complete! ===")
        print(f"Best model saved at: {results.save_dir}")
        print(f"Training results: {results}")
        
        # Validate the model
        print("\nRunning validation...")
        val_results = model.val()
        print(f"Validation results: {val_results}")
        
        # Export the model (optional)
        print("\nExporting model to ONNX format...")
        model.export(format='onnx')
        
        print("\n=== All Done! ===")
        print("Your trained model is ready to use!")
        print(f"Model location: {results.save_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your system requirements and dataset.")

def predict_sample():
    """Run prediction on a sample image (optional)"""
    print("\n=== Sample Prediction ===")
    
    # You can add sample prediction code here
    # model = YOLO('path/to/your/trained/model.pt')
    # results = model('path/to/test/image.jpg')
    # results[0].show()  # Display results
    
    print("To run predictions, load your trained model and use:")
    print("model = YOLO('runs/segment/combined_dataset_training/weights/best.pt')")
    print("results = model('path/to/your/image.jpg')")
    print("results[0].show()  # Display results")

if __name__ == "__main__":
    print("YOLOv8 Training Script")
    print("Make sure you have ultralytics installed: pip install ultralytics")
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        exit(1)
    
    # Start training
    train_yolov8()
    
    # Optional: Run sample prediction
    # predict_sample()
