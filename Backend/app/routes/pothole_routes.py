from flask import Blueprint, jsonify, request, current_app
from app.controllers import pothole_controller

pothole_bp = Blueprint('pothole', __name__)


@pothole_bp.route('/preload', methods=['POST'])
def preload_model():
    """Preload the model (called by frontend on app start)"""
    try:
        model_path = current_app.config['MODEL_PATH']
        from app.services.pothole_detector import get_global_detector
        get_global_detector(model_path)
        return jsonify({'status': 'success', 'message': 'Model preloaded'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@pothole_bp.route('/start', methods=['POST'])
def start_detection():
    """Start pothole detection"""
    model_path = current_app.config['MODEL_PATH']
    camera_id = current_app.config['CAMERA_ID']
    
    result = pothole_controller.start_detection(model_path, camera_id)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@pothole_bp.route('/stop', methods=['POST'])
def stop_detection():
    """Stop pothole detection"""
    result = pothole_controller.stop_detection()
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@pothole_bp.route('/status', methods=['GET'])
def get_status():
    """Get detection stream status"""
    result = pothole_controller.get_stream_status()
    return jsonify(result)


@pothole_bp.route('/video_feed', methods=['GET'])
def video_feed():
    """Video streaming endpoint"""
    return pothole_controller.video_feed()
