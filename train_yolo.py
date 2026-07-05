"""
YOLO Object Detection Training Script
Detects: Helmets, No-Helmets, Vests, People
"""

from ultralytics import YOLO
import os
import torch

print("=" * 60)
print("🚀 YOLO OBJECT DETECTION TRAINING")
print("=" * 60)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📊 Using device: {device}")

# Load a pretrained YOLO model
print("📦 Loading YOLO model...")
model = YOLO('yolo11n.pt')  # Nano model - fast and efficient

# Model info
print(f"📋 Model: YOLO11 Nano")
print(f"📊 Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# Train settings
epochs = 10  # Start small for quick testing
imgsz = 640
batch = 8  # Adjust based on your RAM

print(f"\n🎯 Training Configuration:")
print(f"   Epochs: {epochs}")
print(f"   Image Size: {imgsz}")
print(f"   Batch Size: {batch}")
print(f"   Device: {device}")

# Check for dataset
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    print("\n⚠️ Dataset not found! Downloading sample dataset...")
    print("This will download a small safety equipment dataset.")
    
    # Download from Roboflow (Hard Hat Workers dataset)
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")  # Get free key from roboflow.com
    project = rf.workspace("roboflow-jvuqo").project("hard-hat-workers")
    dataset = project.version(6).download("yolov11")
    dataset_path = "Hard-Hat-Workers-6"
    print(f"✅ Dataset downloaded to: {dataset_path}")

print(f"\n📁 Dataset path: {dataset_path}")

# Train the model
print("\n🚀 Starting training...")
print("This will take 5-15 minutes depending on your system.")
print("=" * 60)

results = model.train(
    data=f'{dataset_path}/data.yaml' if os.path.exists(dataset_path) else 'dataset/data.yaml',
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    name='safety_detector',
    device=device,
    workers=4,
    patience=5,  # Early stopping
    verbose=True
)

print("\n" + "=" * 60)
print("🎉 Training Complete!")
print("=" * 60)

# Save the model
model_path = 'runs/detect/safety_detector/weights/best.pt'
if os.path.exists(model_path):
    print(f"✅ Model saved to: {model_path}")
    
    # Export to ONNX for faster inference
    print("📦 Exporting to ONNX format...")
    model.export(format='onnx', imgsz=imgsz)
    print("✅ ONNX export complete!")
else:
    print("⚠️ Warning: Model not saved. Check training logs.")

print("\n📊 Training summary:")
print(f"   Model: YOLO11 Nano")
print(f"   Classes: Helmet, No-Helmet, Vest, Person")
print(f"   Training complete! Ready for detection.")
