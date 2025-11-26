import os
import platform

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Detect if running on Jetson device
    IS_JETSON = os.path.exists('/etc/nv_tegra_release') or 'jetson' in platform.processor().lower()
    
    # Pothole detection model paths (platform-specific)
    if IS_JETSON:
        # TensorRT engine for Jetson (optimized)
        MODEL_PATH = os.environ.get('MODEL_PATH') or '/home/cdac/Desktop/mlmodels/best_robust_model.pth'
    else:
        # PyTorch model for Windows/Linux
        MODEL_PATH = os.environ.get('MODEL_PATH') or r"C:\Users\ripunjay\Desktop\ml_models\best_robust_model.pth"
    
    # YOLO model paths for blind spot detection (platform-specific)
    if IS_JETSON:
        # TensorRT engine for Jetson (optimized)
        YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH') or 'yolov8n.pt'
    else:
        # PyTorch model for Windows/Linux
        YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH') or 'yolov8n.pt'  # Will use from Backend/ folder
    
    # Camera configuration
    CAMERA_ID = int(os.environ.get('CAMERA_ID', 0))  # Camera 0 for pothole detection
    
    # Blind spot detection camera IDs
    LEFT_CAMERA_ID = int(os.environ.get('LEFT_CAMERA_ID', 1))   # Camera 1 for left blind spot
    RIGHT_CAMERA_ID = int(os.environ.get('RIGHT_CAMERA_ID', 2))  # Camera 2 for right blind spot
