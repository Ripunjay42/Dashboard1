from flask import Response, jsonify
from app.services.dms_detector import DMSStreamManager

# Global DMS stream manager instance
dms_manager = None


def initialize_stream(camera_id):
    """Initialize the DMS stream manager"""
    global dms_manager
    if dms_manager is None:
        dms_manager = DMSStreamManager(camera_id)
    return dms_manager


def start_detection(camera_id):
    """Start the DMS detection stream"""
    try:
        manager = initialize_stream(camera_id)
        if manager.is_active():
            return {'status': 'success', 'message': 'DMS already running'}
        
        if manager.start():
            return {'status': 'success', 'message': 'DMS detection started'}
        else:
            return {'status': 'error', 'message': 'Failed to open camera'}, 500
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


def stop_detection():
    """Stop the DMS detection stream and reset manager for clean restart"""
    global dms_manager
    if dms_manager is not None:
        dms_manager.stop()
        dms_manager = None  # Reset for fresh initialization on next start
        return {'status': 'success', 'message': 'DMS detection stopped'}
    return {'status': 'success', 'message': 'DMS was not running'}


def get_stream_status():
    """Get the current status of the DMS stream"""
    global dms_manager
    if dms_manager is None:
        return {
            'status': 'inactive',
            'running': False,
            'is_drowsy': False,
            'is_yawning': False,
            'ear_value': 0.0,
            'yawn_value': 0.0
        }
    
    is_active = dms_manager.is_active()
    
    if is_active:
        dms_status = dms_manager.get_status()
        return {
            'status': 'active',
            'running': True,
            'is_drowsy': bool(dms_status['is_drowsy']),
            'is_yawning': bool(dms_status['is_yawning']),
            'ear_value': float(dms_status['ear_value']),
            'yawn_value': float(dms_status['yawn_value'])
        }
    else:
        return {
            'status': 'inactive',
            'running': False,
            'is_drowsy': False,
            'is_yawning': False,
            'ear_value': 0.0,
            'yawn_value': 0.0
        }


def generate_frames():
    """Generator function for video streaming"""
    global dms_manager
    if dms_manager is None or not dms_manager.is_active():
        return
    
    while dms_manager.is_active():
        frame_bytes = dms_manager.get_encoded_frame()
        if frame_bytes is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
