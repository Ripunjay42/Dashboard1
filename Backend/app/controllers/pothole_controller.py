from flask import Response, jsonify
from app.services.pothole_detector import VideoStreamManager

# Global video stream manager instance
video_manager = None


def initialize_stream(model_path, camera_id):
    """Initialize the video stream manager"""
    global video_manager
    if video_manager is None:
        video_manager = VideoStreamManager(model_path, camera_id)
    return video_manager


def start_detection(model_path, camera_id):
    """Start the pothole detection stream"""
    try:
        manager = initialize_stream(model_path, camera_id)
        if manager.is_active():
            return {'status': 'success', 'message': 'Stream already running'}
        
        if manager.start():
            return {'status': 'success', 'message': 'Detection started'}
        else:
            return {'status': 'error', 'message': 'Failed to open camera'}, 500
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


def stop_detection():
    """Stop the pothole detection stream"""
    global video_manager
    if video_manager is not None:
        video_manager.stop()
        return {'status': 'success', 'message': 'Detection stopped'}
    return {'status': 'error', 'message': 'No active stream'}, 400


def get_stream_status():
    """Get the current status of the video stream"""
    global video_manager
    if video_manager is None:
        return {'status': 'inactive', 'running': False, 'pothole_detected': False}
    
    is_active = video_manager.is_active()
    pothole_detected = video_manager.is_pothole_detected() if is_active else False
    
    # Ensure all values are JSON serializable (Python bool, not NumPy bool_)
    return {
        'status': 'active' if is_active else 'inactive', 
        'running': bool(is_active),
        'pothole_detected': bool(pothole_detected)
    }


def generate_frames():
    """Generator function for video streaming - optimized for smooth playback"""
    global video_manager
    if video_manager is None or not video_manager.is_active():
        return
    
    while video_manager.is_active():
        # Get pre-encoded frame for better performance
        frame_bytes = video_manager.get_encoded_frame()
        if frame_bytes is None:
            continue
        
        # Stream the pre-encoded frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
