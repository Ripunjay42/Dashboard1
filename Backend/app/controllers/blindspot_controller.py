from flask import Response, jsonify
from app.services.blindspot_detector import DualCameraManager

# Global camera manager instance (dual camera mode)
camera_manager = None


def initialize_camera(left_cam_id=2, right_cam_id=4):
    """Initialize the dual camera manager"""
    global camera_manager
    
    if camera_manager is not None:
        camera_manager.stop()
        camera_manager = None
    
    camera_manager = DualCameraManager(left_cam_id, right_cam_id)
    return camera_manager


def start_detection(left_cam_id=2, right_cam_id=4, camera=None):
    """Start the blind spot detection for both cameras"""
    try:
        # Stop any existing detection first
        stop_detection()
        
        manager = initialize_camera(left_cam_id, right_cam_id)
        
        # Load detector model first (singleton, fast if already loaded)
        from app.services.blindspot_detector import get_global_detector
        get_global_detector()
        
        if manager.start():
            return {'status': 'success', 'message': 'Blind spot detection started (dual camera mode)'}
        else:
            return {'status': 'error', 'message': 'Failed to open cameras'}, 500
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


def stop_detection():
    """Stop the blind spot detection"""
    global camera_manager
    if camera_manager is not None:
        camera_manager.stop()
        camera_manager = None
        return {'status': 'success', 'message': 'Detection stopped'}
    return {'status': 'success', 'message': 'No active detection'}


def get_stream_status():
    """Get the current status of the blind spot detection"""
    global camera_manager
    if camera_manager is None:
        return {
            'status': 'inactive', 
            'running': False, 
            'left_danger': False,
            'right_danger': False
        }
    
    is_active = camera_manager.is_active()
    left_danger, right_danger = camera_manager.get_danger_status() if is_active else (False, False)
    
    return {
        'status': 'active' if is_active else 'inactive', 
        'running': bool(is_active),
        'left_danger': bool(left_danger),
        'right_danger': bool(right_danger)
    }


def generate_left_frames():
    """Generator function for left camera video streaming"""
    global camera_manager
    if camera_manager is None or not camera_manager.is_active():
        return
    
    while camera_manager.is_active():
        frame_bytes = camera_manager.get_left_frame()
        if frame_bytes is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_right_frames():
    """Generator function for right camera video streaming"""
    global camera_manager
    if camera_manager is None or not camera_manager.is_active():
        return
    
    while camera_manager.is_active():
        frame_bytes = camera_manager.get_right_frame()
        if frame_bytes is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def left_video_feed():
    """Left camera video streaming route"""
    return Response(generate_left_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def right_video_feed():
    """Right camera video streaming route"""
    return Response(generate_right_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
