from flask import Response, jsonify
from app.services.pothole_detector import VideoStreamManager, DualCameraManager
import threading
import time

# Global video stream manager instance (for backward compatibility with single camera)
video_manager = None
# Global dual camera manager instance
camera_manager = None
_operation_lock = threading.Lock()  # Prevent concurrent start/stop operations


def initialize_cameras(top_cam_id=0, bottom_cam_id=2, camera_mode='top'):
    """Initialize the dual camera manager with optional single camera mode"""
    global camera_manager
    if camera_manager is None:
        camera_manager = DualCameraManager(top_cam_id, bottom_cam_id, camera_mode)
    return camera_manager


def start_detection(top_cam_id=0, bottom_cam_id=2, camera_mode='top'):
    """Start the pothole detection with single or dual camera mode"""
    global camera_manager
    
    # Prevent concurrent start/stop operations
    if not _operation_lock.acquire(blocking=False):
        return {'status': 'error', 'message': 'Operation in progress, please wait'}, 409
    
    try:
        print(f"🚀 Starting pothole detection - camera_mode: {camera_mode}")
        
        # Reset manager if it exists but isn't active (clean slate)
        if camera_manager is not None and not camera_manager.is_active():
            print("   Resetting inactive manager...")
            camera_manager = None
        
        manager = initialize_cameras(top_cam_id, bottom_cam_id, camera_mode)
        if manager.is_active():
            print("   Detection already running")
            return {'status': 'success', 'message': 'Detection already running'}
        
        # JETSON: Small delay after previous cleanup to ensure camera release
        time.sleep(0.5)
        
        if manager.start():
            print(f"✓ Pothole detection started in {camera_mode} mode")
            return {'status': 'success', 'message': f'Detection started in {camera_mode} mode'}
        else:
            camera_manager = None
            print("❌ Failed to start stream")
            return {'status': 'error', 'message': 'Failed to start stream'}, 500
    except Exception as e:
        print(f"❌ Error in start_detection: {e}")
        camera_manager = None
        return {'status': 'error', 'message': str(e)}, 500
    finally:
        _operation_lock.release()


def stop_detection():
    """Stop the pothole detection stream and reset manager for clean restart"""
    global camera_manager, video_manager
    import gc
    import torch
    
    print("🛑 Stopping pothole detection...")
    
    # Prevent concurrent start/stop operations
    if not _operation_lock.acquire(blocking=False):
        return {'status': 'success', 'message': 'Stop already in progress'}
    
    try:
        # Stop dual camera manager if active
        if camera_manager is not None:
            print("   Stopping dual camera manager...")
            camera_manager.stop()
            camera_manager = None
            print("   ✓ Dual camera manager stopped")
        
        # Stop old single camera manager if active (backward compatibility)
        if video_manager is not None:
            print("   Stopping single camera manager...")
            video_manager.stop()
            video_manager = None
            print("   ✓ Single camera manager stopped")
        
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
        
        print("✅ Pothole cleanup complete")
        return {'status': 'success', 'message': 'Detection stopped'}
    finally:
        _operation_lock.release()


def get_stream_status():
    """Get the current status of the pothole detection"""
    global camera_manager, video_manager
    
    # Check dual camera manager first
    if camera_manager is not None:
        is_active = camera_manager.is_active()
        top_danger = camera_manager.is_top_danger() if is_active else False
        bottom_danger = camera_manager.is_bottom_danger() if is_active else False
        
        return {
            'status': 'active' if is_active else 'inactive',
            'running': bool(is_active),
            'top_pothole_detected': bool(top_danger),
            'bottom_pothole_detected': bool(bottom_danger),
            'pothole_detected': bool(top_danger or bottom_danger),  # Legacy compatibility
            'camera_mode': camera_manager.camera_mode
        }
    
    # Fallback to old single camera manager
    if video_manager is not None:
        is_active = video_manager.is_active()
        pothole_detected = video_manager.is_pothole_detected() if is_active else False
        
        return {
            'status': 'active' if is_active else 'inactive',
            'running': bool(is_active),
            'pothole_detected': bool(pothole_detected),
            'camera_mode': 'single'
        }
    
    # Nothing running
    return {
        'status': 'inactive',
        'running': False,
        'pothole_detected': False,
        'top_pothole_detected': False,
        'bottom_pothole_detected': False,
        'camera_mode': 'none'
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


def generate_top_frames():
    """Generator function for top camera video streaming"""
    global camera_manager
    import time
    
    if camera_manager is None or not camera_manager.is_active():
        return
    
    consecutive_failures = 0
    max_failures = 30
    last_frame_time = time.time()
    frame_timeout = 5.0
    
    while True:
        if camera_manager is None or not camera_manager.is_active():
            break
        
        if time.time() - last_frame_time > frame_timeout:
            break
        
        frame_bytes = camera_manager.get_top_encoded_frame()
        
        if frame_bytes is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break
            time.sleep(0.033)
            continue
        
        consecutive_failures = 0
        last_frame_time = time.time()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_bottom_frames():
    """Generator function for bottom camera video streaming"""
    global camera_manager
    import time
    
    if camera_manager is None or not camera_manager.is_active():
        return
    
    consecutive_failures = 0
    max_failures = 30
    last_frame_time = time.time()
    frame_timeout = 5.0
    
    while True:
        if camera_manager is None or not camera_manager.is_active():
            break
        
        if time.time() - last_frame_time > frame_timeout:
            break
        
        frame_bytes = camera_manager.get_bottom_encoded_frame()
        
        if frame_bytes is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                break
            time.sleep(0.033)
            continue
        
        consecutive_failures = 0
        last_frame_time = time.time()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def top_video_feed():
    """Top camera video streaming route"""
    return Response(generate_top_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def bottom_video_feed():
    """Bottom camera video streaming route"""
    return Response(generate_bottom_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def video_feed():
    """Video streaming route - LEGACY (uses top camera if dual mode)"""
    # If dual camera manager is active, use top camera
    if camera_manager is not None and camera_manager.is_active():
        return Response(generate_top_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Otherwise use old single camera feed
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
