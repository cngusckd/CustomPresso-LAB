import torch
import cv2
import numpy as np
from models.yolo_v12n import YOLOv12n
import os

def load_model(model_name, device='cpu'):
    """
    Loads the requested model.
    For Phase 5 Demo, we primarily use the Baseline model for visual inference 
    as Int8/TRT engines might be platform specific or mocked.
    """
    print(f"Loading model: {model_name} on {device}")
    
    # In a real scenario, we would load state_dict or engine here.
    # For this demo, we initialize the architecture.
    model = YOLOv12n(nc=20)
    
    # Load weights if available (e.g., 'best.pt')
    # if os.path.exists('best.pt'):
    #     model.load_state_dict(torch.load('best.pt', map_location=device))
        
    model.to(device)
    model.eval()
    return model

def draw_boxes(image, detections, class_names=None):
    """
    Draws bounding boxes on the image.
    detections: list of [x1, y1, x2, y2, conf, cls] (from NMS)
    image: PIL Image or Numpy array (RGB)
    """
    img = np.array(image)
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    
    # Ensure detections is numpy array from the start
    if isinstance(detections, torch.Tensor):
        detections = detections.detach().cpu().numpy()
    
    for det in detections:
        # det is now numpy array row
        
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Color
        color = (0, 255, 0) # Green
        
        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{int(cls_id)}: {conf:.2f}"
        if class_names and int(cls_id) < len(class_names):
             label = f"{class_names[int(cls_id)]}: {conf:.2f}"
             
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    # Back to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
