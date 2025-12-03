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
        import time
        manager = initialize_stream(camera_id)
        if manager.is_active():
            return {'status': 'success', 'message': 'DMS already running'}
        
        # JETSON: Small delay after previous cleanup to ensure camera release
        time.sleep(0.5)
        
        if manager.start():
            return {'status': 'success', 'message': 'DMS detection started'}
        else:
            return {'status': 'error', 'message': 'Failed to open camera'}, 500
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500


def stop_detection():
    """Stop the DMS detection stream and reset manager for clean restart"""
    global dms_manager
    import gc
    import torch
    
    if dms_manager is not None:
        print("🧹 CLEANUP: Stopping DMS detection with aggressive cleanup...")
        dms_manager.stop()  # This handles camera release and lock release internally
        dms_manager = None  # Reset for fresh initialization on next start
        
        # NOTE: Don't call release_camera_lock or cleanup_all_cameras here!
        # DMSStreamManager.stop() already handles this properly.
        # Double-releasing corrupts the lock state and breaks subsequent starts.
        
        # JETSON: Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # JETSON: Aggressive garbage collection (3 passes)
        for i in range(3):
            gc.collect()
        
        print("✅ CLEANUP: DMS cleanup complete")
        return {'status': 'success', 'message': 'Detection stopped'}
    return {'status': 'success', 'message': 'Detection was not running'}


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
    """Generator function for video streaming - JETSON OPTIMIZED"""
    global dms_manager
    import time
    
    if dms_manager is None or not dms_manager.is_active():
        return
    
    consecutive_failures = 0
    max_failures = 30  # After ~1 second of failures, exit gracefully
    last_frame_time = time.time()
    frame_timeout = 5.0  # Exit if no new frame for 5 seconds
    
    while True:
        # Check if manager is still active (quick exit on stop)
        if dms_manager is None or not dms_manager.is_active():
            break
        
        # Check for frame timeout (feed stuck)
        if time.time() - last_frame_time > frame_timeout:
            print("⚠️ DMS feed timeout - no new frames")
            break
            
        frame_bytes = dms_manager.get_encoded_frame()
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


def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
