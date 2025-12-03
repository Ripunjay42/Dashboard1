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
        import time
        manager = initialize_stream(model_path, camera_id)
        if manager.is_active():
            return {'status': 'error', 'message': 'Detection already running'}, 400
        
        # JETSON: Small delay after previous cleanup to ensure camera release
        time.sleep(0.5)
        
        if manager.start():
            return {'status': 'success', 'message': 'Detection started'}
        else:
            return {'status': 'error', 'message': 'Failed to start stream'}, 500
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


def stop_detection():
    """Stop the pothole detection stream and reset manager for clean restart"""
    global video_manager
    import gc
    import torch
    
    if video_manager is not None:
        print("🧹 CLEANUP: Stopping pothole detection with aggressive cleanup...")
        video_manager.stop()  # This handles camera release and lock release internally
        video_manager = None  # Reset for fresh initialization on next start
        
        # NOTE: Don't call release_camera_lock or cleanup_all_cameras here!
        # VideoStreamManager.stop() already handles this properly.
        # Double-releasing corrupts the lock state and breaks subsequent starts.
        
        # Unload pothole model from memory
        from app.services.pothole_detector import unload_global_detector
        unload_global_detector()
        
        # JETSON: Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # JETSON: Aggressive garbage collection (3 passes)
        for i in range(3):
            gc.collect()
        
        print("✅ CLEANUP: Pothole cleanup complete")
        return {'status': 'success', 'message': 'Detection stopped'}
    return {'status': 'success', 'message': 'Detection was not running'}


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
    """Generator function for video streaming - JETSON OPTIMIZED"""
    global video_manager
    import time
    
    if video_manager is None or not video_manager.is_active():
        return
    
    consecutive_failures = 0
    max_failures = 30  # After ~1 second of failures, exit gracefully
    last_frame_time = time.time()
    frame_timeout = 5.0  # Exit if no new frame for 5 seconds
    
    while True:
        # Check if manager is still active (quick exit on stop)
        if video_manager is None or not video_manager.is_active():
            break
        
        # Check for frame timeout (feed stuck)
        if time.time() - last_frame_time > frame_timeout:
            print("⚠️ Pothole feed timeout - no new frames")
            break
            
        # Get pre-encoded frame for better performance
        frame_bytes = video_manager.get_encoded_frame()
        if frame_bytes is None:
            consecutive_failures += 1
            if consecutive_failures > max_failures:
                # Too many failures, exit to prevent browser hang
                break
            time.sleep(0.033)  # ~30fps rate limiting, prevents CPU spin
            continue
        
        consecutive_failures = 0  # Reset on success
        last_frame_time = time.time()  # Update last frame time
        
        # Stream the pre-encoded frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
