import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    MODEL_PATH = os.environ.get('MODEL_PATH') or r"C:\Users\ripunjay\Desktop\ml_models\best_robust_model.pth"
    CAMERA_ID = int(os.environ.get('CAMERA_ID', 0))
