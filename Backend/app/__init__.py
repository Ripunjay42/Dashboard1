from flask import Flask
from flask_cors import CORS
from app.config import Config
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from app.routes.hello_routes import hello_bp
    from app.routes.pothole_routes import pothole_bp
    
    app.register_blueprint(hello_bp)
    app.register_blueprint(pothole_bp, url_prefix='/api/pothole')
    
    # Preload model on startup to reduce first-time initialization delay
    model_path = app.config.get('MODEL_PATH')
    camera_id = app.config.get('CAMERA_ID', 0)
    
    if model_path and os.path.exists(model_path):
        print("Preloading pothole detection model...")
        from app.services.pothole_detector import get_global_detector
        try:
            get_global_detector(model_path)
            print("✓ Model preloaded successfully!")
            
            # Pre-initialize camera for instant startup
            print("\nPre-initializing camera for instant detection...")
            from app.controllers.pothole_controller import pre_initialize_camera
            if pre_initialize_camera(model_path, camera_id):
                print("✓ Camera ready! Detection will start instantly when requested.\n")
            else:
                print("⚠ Camera pre-initialization failed. Will initialize on first use.\n")
                
        except Exception as e:
            print(f"Warning: Could not preload model/camera: {e}")
    else:
        print(f"Warning: Model file not found at {model_path}")
    
    return app
