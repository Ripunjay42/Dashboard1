from flask import Blueprint, jsonify

hello_bp = Blueprint('hello', __name__)


@hello_bp.route('/hello', methods=['GET'])
def hello():
    """Simple hello endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Hello from Experience Centre Dashboard API!'
    })


@hello_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Experience Centre Backend'
    })


@hello_bp.route('/api/cleanup', methods=['POST'])
def cleanup_cameras():
    """Emergency cleanup endpoint - call if camera switching gets stuck"""
    try:
        from app.services.camera_manager import cleanup_all_cameras
        cleanup_all_cameras()
        
        # Also reset all controllers
        from app.controllers import pothole_controller, blindspot_controller, dms_controller
        
        if pothole_controller.video_manager:
            try:
                pothole_controller.video_manager.stop()
            except:
                pass
            pothole_controller.video_manager = None
        
        if blindspot_controller.camera_manager:
            try:
                blindspot_controller.camera_manager.stop()
            except:
                pass
            blindspot_controller.camera_manager = None
        
        if dms_controller.dms_manager:
            try:
                dms_controller.dms_manager.stop()
            except:
                pass
            dms_controller.dms_manager = None
        
        return jsonify({
            'status': 'success',
            'message': 'Emergency camera cleanup completed'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
