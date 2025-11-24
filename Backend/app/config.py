import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MODEL_PATH = os.environ.get('MODEL_PATH') or r"C:\Users\ripunjay\Desktop\ml_models\best_robust_model.pth"
    
    # Camera configuration
    CAMERA_ID = int(os.environ.get('CAMERA_ID', 0))  # Camera 0 for pothole detection
    
    # Blind spot detection camera IDs
    LEFT_CAMERA_ID = int(os.environ.get('LEFT_CAMERA_ID', 0))   # Camera 1 for left blind spot
    RIGHT_CAMERA_ID = int(os.environ.get('RIGHT_CAMERA_ID', 1))  # Camera 2 for right blind spot
