from flask import Flask
from flask_cors import CORS
from app.config import Config
import os

# Global flag to prevent re-initialization on Flask reloader
_initialized = False

def create_app():
    global _initialized
    
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    from app.routes.hello_routes import hello_bp
    from app.routes.pothole_routes import pothole_bp
    from app.routes.blindspot_routes import blindspot_bp
    
    app.register_blueprint(hello_bp)
    app.register_blueprint(pothole_bp, url_prefix='/api/pothole')
    app.register_blueprint(blindspot_bp, url_prefix='/api/blindspot')
    
    # Models will be loaded lazily on first use (when user clicks feature)
    # No pre-initialization - cameras start only when user clicks Pothole/Blind Spot
    
    return app
