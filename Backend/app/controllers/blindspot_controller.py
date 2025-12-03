from flask import Response, jsonify
from app.services.blindspot_detector import DualCameraManager
import threading

# Global dual camera manager instance
camera_manager = None
_operation_lock = threading.Lock()  # Prevent concurrent start/stop operations


def initialize_cameras(left_cam_id=0, right_cam_id=1, camera_mode='left'):
    """Initialize the dual camera manager with optional single camera mode"""
    global camera_manager
    if camera_manager is None:
        camera_manager = DualCameraManager(left_cam_id, right_cam_id, camera_mode)
    return camera_manager


def start_detection(left_cam_id=0, right_cam_id=1, camera_mode='left'):
    """Start the blind spot detection with single or dual camera mode"""
    global camera_manager
    
    # Prevent concurrent start/stop operations
    if not _operation_lock.acquire(blocking=False):
        return {'status': 'error', 'message': 'Operation in progress, please wait'}, 409
    
    try:
        import time
        from app.services.camera_manager import acquire_camera_lock, release_camera_lock
        
        # Reset manager if it exists but isn't active (clean slate)
        if camera_manager is not None and not camera_manager.is_active():
            camera_manager = None
        
        manager = initialize_cameras(left_cam_id, right_cam_id, camera_mode)
        if manager.is_active():
            return {'status': 'success', 'message': 'Detection already running'}
        
        # JETSON: Acquire camera lock FIRST to prevent race conditions
        if not acquire_camera_lock('blindspot', timeout=5.0):
            print("❌ Could not acquire camera lock - another service may be using the camera")
            return {'status': 'error', 'message': 'Failed to acquire camera lock (another service is using cameras)'}, 500
        
        # JETSON: Wait for cameras to be fully released by previous service
        time.sleep(0.8)
        
        # Load detector model first (singleton, fast if already loaded)
        from app.services.blindspot_detector import get_global_detector
        get_global_detector()
        
        if manager.start():
            return {'status': 'success', 'message': 'Detection started'}
        else:
            # IMPORTANT: Release lock if start failed
            release_camera_lock('blindspot')
            camera_manager = None
            return {'status': 'error', 'message': 'Failed to start detection'}, 500
    except Exception as e:
        # IMPORTANT: Release lock on any exception
        try:
            from app.services.camera_manager import release_camera_lock
            release_camera_lock('blindspot')
        except:
            pass
        camera_manager = None
        return {'status': 'error', 'message': str(e)}, 500
    finally:
        _operation_lock.release()


def stop_detection():
    """Stop the blind spot detection and reset manager for clean restart"""
    global camera_manager
    import gc
    import torch
    
    # Prevent concurrent start/stop operations
    if not _operation_lock.acquire(blocking=False):
        return {'status': 'success', 'message': 'Stop already in progress'}
    
    try:
        if camera_manager is not None:
            print("🧹 CLEANUP: Stopping blindspot detection with aggressive cleanup...")
            camera_manager.stop()  # This handles camera release and lock release internally
            camera_manager = None  # Reset for fresh initialization on next start
            
            # JETSON: Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # JETSON: Aggressive garbage collection (3 passes)
            for i in range(3):
                gc.collect()
            
            print("✅ CLEANUP: Blindspot cleanup complete")
            return {'status': 'success', 'message': 'Detection stopped'}
        return {'status': 'success', 'message': 'Detection was not running'}
    finally:
        _operation_lock.release()


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
        try:
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
        except (GeneratorExit, StopIteration):
            # Client disconnected or generator stopped
            break
        except Exception as e:
            print(f"⚠️ Left feed error: {e}")
            break


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
        try:
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
        except (GeneratorExit, StopIteration):
            # Client disconnected or generator stopped
            break
        except Exception as e:
            print(f"⚠️ Right feed error: {e}")
            break


def left_video_feed():
    """Left camera video streaming route"""
    return Response(generate_left_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def right_video_feed():
    """Right camera video streaming route"""
    return Response(generate_right_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
