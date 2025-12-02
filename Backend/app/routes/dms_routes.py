from flask import Blueprint, jsonify, request, current_app
from app.controllers import dms_controller

dms_bp = Blueprint('dms', __name__)


@dms_bp.route('/preload', methods=['POST'])
def preload_model():
    """Preload the DMS model (called by frontend on app start)"""
    try:
        from app.services.dms_detector import get_global_detector
        get_global_detector()
        return jsonify({'status': 'success', 'message': 'DMS model preloaded'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@dms_bp.route('/start', methods=['POST'])
def start_detection():
    """Start DMS detection"""
    # Use camera ID 0 for DMS (front-facing driver camera)
    # Can be configured in config.py if needed
    camera_id = current_app.config.get('DMS_CAMERA_ID', 6)
    
    result = dms_controller.start_detection(camera_id)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@dms_bp.route('/stop', methods=['POST'])
def stop_detection():
    """Stop DMS detection"""
    result = dms_controller.stop_detection()
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@dms_bp.route('/status', methods=['GET'])
def get_status():
    """Get DMS detection status"""
    result = dms_controller.get_stream_status()
    return jsonify(result)


@dms_bp.route('/video_feed', methods=['GET'])
def video_feed():
    """Video streaming endpoint"""
    return dms_controller.video_feed()
