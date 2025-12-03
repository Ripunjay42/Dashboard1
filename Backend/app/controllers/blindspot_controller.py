from flask import Response, jsonify
from app.services.blindspot_detector import DualCameraManager

# Global dual camera manager instance
camera_manager = None


def initialize_cameras(left_cam_id=0, right_cam_id=1):
    """Initialize the dual camera manager"""
    global camera_manager
    if camera_manager is None:
        camera_manager = DualCameraManager(left_cam_id, right_cam_id)
    return camera_manager


def start_detection(left_cam_id=0, right_cam_id=1):
    """Start the blind spot detection"""
    try:
        manager = initialize_cameras(left_cam_id, right_cam_id)
        if manager.is_active():
            return {'status': 'success', 'message': 'Detection already running'}
        
        # Load detector model first (singleton, fast if already loaded)
        from app.services.blindspot_detector import get_global_detector
        get_global_detector()
        
        if manager.start():
            return {'status': 'success', 'message': 'Blind spot detection started'}
        else:
            return {'status': 'error', 'message': 'Failed to open cameras'}, 500
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


def stop_detection():
    """Stop the blind spot detection and reset manager for clean restart"""
    global camera_manager
    if camera_manager is not None:
        camera_manager.stop()
        camera_manager = None  # Reset for fresh initialization on next start
        return {'status': 'success', 'message': 'Detection stopped'}
    return {'status': 'success', 'message': 'Detection was not running'}


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
    left_danger = camera_manager.is_left_danger() if is_active else False
    right_danger = camera_manager.is_right_danger() if is_active else False
    
    return {
        'status': 'active' if is_active else 'inactive', 
        'running': bool(is_active),
        'left_danger': bool(left_danger),
        'right_danger': bool(right_danger)
    }


def generate_left_frames():
    """Generator function for left camera video streaming - JETSON OPTIMIZED"""
    global camera_manager
    import time
    
    if camera_manager is None or not camera_manager.is_active():
        return
    
    consecutive_failures = 0
    max_failures = 30  # After ~1 second of failures, exit gracefully
    last_frame_time = time.time()
    frame_timeout = 5.0  # Exit if no new frame for 5 seconds
    
    while True:
        # Check if manager is still active (quick exit on stop)
        if camera_manager is None or not camera_manager.is_active():
            break
        
        # Check for frame timeout (feed stuck)
        if time.time() - last_frame_time > frame_timeout:
            print("⚠️ Blindspot left feed timeout - no new frames")
            break
            
        frame_bytes = camera_manager.get_left_frame()
        if frame_bytes is None:
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                # Too many failures, exit to prevent browser hang
                break
            time.sleep(0.033)  # ~30fps rate limiting, prevents CPU spin
            continue
        
        consecutive_failures = 0  # Reset on success
        last_frame_time = time.time()  # Update last frame time
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_right_frames():
    """Generator function for right camera video streaming - JETSON OPTIMIZED"""
    global camera_manager
    import time
    
    if camera_manager is None or not camera_manager.is_active():
        return
    
    consecutive_failures = 0
    max_failures = 30  # After ~1 second of failures, exit gracefully
    last_frame_time = time.time()
    frame_timeout = 5.0  # Exit if no new frame for 5 seconds
    
    while True:
        # Check if manager is still active (quick exit on stop)
        if camera_manager is None or not camera_manager.is_active():
            break
        
        # Check for frame timeout (feed stuck)
        if time.time() - last_frame_time > frame_timeout:
            print("⚠️ Blindspot right feed timeout - no new frames")
            break
            
        frame_bytes = camera_manager.get_right_frame()
        if frame_bytes is None:
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                # Too many failures, exit to prevent browser hang
                break
            time.sleep(0.033)  # ~30fps rate limiting, prevents CPU spin
            continue
        
        consecutive_failures = 0  # Reset on success
        last_frame_time = time.time()  # Update last frame time
        
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
