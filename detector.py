"""
Object Detection Engine using YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self, model_path='runs/detect/safety_detector/weights/best.pt'):
        """
        Initialize the YOLO detector
        """
        print(f"🔍 Loading model: {model_path}")
        
        # Check if model exists, use pretrained if not
        if not os.path.exists(model_path):
            print("⚠️ Trained model not found. Using pretrained YOLO11...")
            model_path = 'yolo11n.pt'
        
        self.model = YOLO(model_path)
        
        # Class names (from Hard Hat Workers dataset)
        self.class_names = {
            0: 'Helmet',
            1: 'No-Helmet',
            2: 'Vest',
            3: 'Person'
        }
        
        # Colors for each class (BGR)
        self.colors = {
            'Helmet': (0, 255, 0),     # Green
            'No-Helmet': (0, 0, 255),  # Red
            'Vest': (255, 200, 0),     # Yellow
            'Person': (255, 0, 0)      # Blue
        }
        
        print(f"✅ Detector initialized with {len(self.class_names)} classes")
    
    def detect_frame(self, frame):
        """
        Detect objects in a single frame
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Only include confident detections
                    if conf > 0.3:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'class': self.class_names.get(cls, 'Unknown'),
                            'class_id': cls,
                            'confidence': conf
                        })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']
            color = self.colors.get(cls_name, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_h - 8),
                (x1 + label_w + 8, y1),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def process_image(self, image_path):
        """
        Process an image file
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        detections = self.detect_frame(frame)
        annotated = self.draw_detections(frame, detections)
        
        return annotated, detections
    
    def get_stats(self, detections):
        """
        Get statistics from detections
        """
        stats = {}
        for det in detections:
            cls_name = det['class']
            stats[cls_name] = stats.get(cls_name, 0) + 1
        return stats

# Test the detector
if __name__ == "__main__":
    print("🧪 Testing ObjectDetector...")
    detector = ObjectDetector()
    print("✅ ObjectDetector initialized successfully!")
