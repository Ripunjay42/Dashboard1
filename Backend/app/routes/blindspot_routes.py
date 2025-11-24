from flask import Blueprint, jsonify, request, current_app
from app.controllers import blindspot_controller

blindspot_bp = Blueprint('blindspot', __name__)


@blindspot_bp.route('/preload', methods=['POST'])
def preload_models():
    """Preload the blind spot detection models (called by frontend on app start)"""
    try:
        from app.services.blindspot_detector import get_global_detector
        get_global_detector()
        return jsonify({'status': 'success', 'message': 'Models preloaded'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@blindspot_bp.route('/start', methods=['POST'])
def start_detection():
    """Start blind spot detection"""
    left_cam_id = current_app.config.get('LEFT_CAMERA_ID', 0)
    right_cam_id = current_app.config.get('RIGHT_CAMERA_ID', 1)
    
    result = blindspot_controller.start_detection(left_cam_id, right_cam_id)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@blindspot_bp.route('/stop', methods=['POST'])
def stop_detection():
    """Stop blind spot detection"""
    result = blindspot_controller.stop_detection()
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@blindspot_bp.route('/status', methods=['GET'])
def get_status():
    """Get detection status"""
    result = blindspot_controller.get_stream_status()
    return jsonify(result)


@blindspot_bp.route('/left_feed', methods=['GET'])
def left_video_feed():
    """Left camera video streaming endpoint"""
    return blindspot_controller.left_video_feed()


@blindspot_bp.route('/right_feed', methods=['GET'])
def right_video_feed():
    """Right camera video streaming endpoint"""
    return blindspot_controller.right_video_feed()
