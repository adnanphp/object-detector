"""
Gradio Web Interface for Object Detection
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from detector import ObjectDetector
from PIL import Image

# Initialize detector
print("🔍 Initializing Object Detector...")
detector = ObjectDetector()
print("✅ Detector ready!")

def process_image(image):
    """
    Process uploaded image and return annotated image with stats
    """
    if image is None:
        return None, "Please upload an image."
    
    # Convert to BGR (OpenCV format)
    if isinstance(image, np.ndarray):
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Detect objects
    detections = detector.detect_frame(frame)
    
    # Draw annotations
    annotated = detector.draw_detections(frame, detections)
    
    # Generate statistics
    stats = detector.get_stats(detections)
    stats_text = "📊 Detection Summary:\n" + "\n".join([
        f"• {cls}: {count}" for cls, count in stats.items()
    ])
    
    if not stats:
        stats_text = "🔍 No objects detected."
    
    # Convert back to RGB for Gradio
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return annotated_rgb, stats_text

def process_video(video):
    """
    Process uploaded video and return annotated video
    """
    if video is None:
        return None
    
    # Create temporary output
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    
    # Open video
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and annotate
        detections = detector.detect_frame(frame)
        annotated = detector.draw_detections(frame, detections)
        
        # Write frame
        out.write(annotated)
        processed += 1
        
        # Progress indicator (every 30 frames)
        if processed % 30 == 0:
            print(f"Processing: {processed}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"✅ Video processed: {processed} frames")
    return output_path

# Create Gradio Interface
with gr.Blocks(
    title="Safety Equipment Detection",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🦺 Safety Equipment Detection System
    
    Upload an **image** or **video** to detect:
    - 🪖 **Helmets** (Green)
    - 🚫 **No-Helmet** (Red) - Safety violation!
    - 🦺 **Safety Vests** (Yellow)
    - 👷 **People** (Blue)
    
    *Powered by YOLO11 Object Detection*
    """)
    
    with gr.Tab("📸 Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil"
                )
                image_button = gr.Button("🔍 Detect Objects", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(
                    label="Detected Objects",
                    type="numpy"
                )
                stats_output = gr.Textbox(
                    label="Detection Summary",
                    lines=8,
                    interactive=False
                )
        
        image_button.click(
            process_image,
            inputs=[image_input],
            outputs=[image_output, stats_output]
        )
    
    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            video_input = gr.Video(
                label="Upload Video"
            )
            video_output = gr.Video(
                label="Processed Video"
            )
        
        video_button = gr.Button("🎬 Process Video", variant="primary")
        video_button.click(
            process_video,
            inputs=[video_input],
            outputs=[video_output]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ### About This Project
        
        **Object Detection System** using YOLO (You Only Look Once)
        
        **Features:**
        - Real-time object detection
        - Safety equipment monitoring
        - Support for images and videos
        - Visual bounding boxes with labels
        
        **Classes:** Helmet, No-Helmet, Vest, Person
        
        **Tech Stack:**
        - Ultralytics YOLO11
        - OpenCV
        - Gradio
        - Python
        """)

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=False
    )
