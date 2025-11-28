from flask import Response, jsonify
from app.services.blindspot_detector import SingleCameraManager

# Global camera manager instance (single camera mode)
camera_manager = None
active_camera = None  # 'left' or 'right'


def initialize_camera(camera_id, camera_side='left'):
    """Initialize the single camera manager"""
    global camera_manager, active_camera
    
    # If switching cameras or first time, create new manager
    if camera_manager is not None:
        camera_manager.stop()
        camera_manager = None
    
    camera_manager = SingleCameraManager(camera_id, camera_side)
    active_camera = camera_side
    return camera_manager


def start_detection(left_cam_id=2, right_cam_id=4, camera='left'):
    """Start the blind spot detection for a single camera"""
    global active_camera
    try:
        # Stop any existing detection first
        stop_detection()
        
        # Choose camera based on selection
        camera_id = left_cam_id if camera == 'left' else right_cam_id
        
        manager = initialize_camera(camera_id, camera)
        
        # Load detector model first (singleton, fast if already loaded)
        from app.services.blindspot_detector import get_global_detector
        get_global_detector()
        
        if manager.start():
            return {'status': 'success', 'message': f'{camera.capitalize()} blind spot detection started'}
        else:
            return {'status': 'error', 'message': f'Failed to open {camera} camera'}, 500
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


def stop_detection():
    """Stop the blind spot detection"""
    global camera_manager, active_camera
    if camera_manager is not None:
        camera_manager.stop()
        camera_manager = None
        active_camera = None
        return {'status': 'success', 'message': 'Detection stopped'}
    return {'status': 'success', 'message': 'No active detection'}


def get_stream_status():
    """Get the current status of the blind spot detection"""
    global camera_manager, active_camera
    if camera_manager is None:
        return {
            'status': 'inactive', 
            'running': False, 
            'left_danger': False,
            'right_danger': False,
            'active_camera': None
        }
    
    is_active = camera_manager.is_active()
    danger = camera_manager.is_danger() if is_active else False
    
    return {
        'status': 'active' if is_active else 'inactive', 
        'running': bool(is_active),
        'left_danger': bool(danger) if active_camera == 'left' else False,
        'right_danger': bool(danger) if active_camera == 'right' else False,
        'active_camera': active_camera
    }


def generate_frames():
    """Generator function for single camera video streaming"""
    global camera_manager
    if camera_manager is None or not camera_manager.is_active():
        return
    
    while camera_manager.is_active():
        frame_bytes = camera_manager.get_frame()
        if frame_bytes is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_left_frames():
    """Generator function for left camera video streaming"""
    global camera_manager, active_camera
    if camera_manager is None or not camera_manager.is_active() or active_camera != 'left':
        return
    
    return generate_frames()


def generate_right_frames():
    """Generator function for right camera video streaming"""
    global camera_manager, active_camera
    if camera_manager is None or not camera_manager.is_active() or active_camera != 'right':
        return
    
    return generate_frames()


def left_video_feed():
    """Left camera video streaming route"""
    global active_camera
    if active_camera == 'left':
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(status=204)  # No content if not active


def right_video_feed():
    """Right camera video streaming route"""
    global active_camera
    if active_camera == 'right':
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(status=204)  # No content if not active
