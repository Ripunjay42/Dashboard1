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
    """Start pothole detection with optional camera mode"""
    top_cam_id = current_app.config.get('TOP_CAMERA_ID', 0)
    bottom_cam_id = current_app.config.get('BOTTOM_CAMERA_ID', 2)
    
    # Get camera mode from request body (default: 'top' for less Jetson load)
    data = request.get_json() or {}
    camera_mode = data.get('camera', 'top')  # 'top', 'bottom', or 'both'
    
    result = pothole_controller.start_detection(top_cam_id, bottom_cam_id, camera_mode)
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
    """Video streaming endpoint (legacy - uses top camera)"""
    return pothole_controller.video_feed()


@pothole_bp.route('/top_feed', methods=['GET'])
def top_feed():
    """Top camera video streaming endpoint"""
    return pothole_controller.top_video_feed()


@pothole_bp.route('/bottom_feed', methods=['GET'])
def bottom_feed():
    """Bottom camera video streaming endpoint"""
    return pothole_controller.bottom_video_feed()
