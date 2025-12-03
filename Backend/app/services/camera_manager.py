"""
Global Camera Manager for Jetson
================================
Ensures clean camera switching between different detection services.
Prevents camera resource conflicts that cause freezing/lag on Jetson.
"""

import threading
import time
import gc
import os
import cv2

# Global lock for camera operations - prevents multiple services from 
# fighting over camera resources
_camera_lock = threading.Lock()
_active_service = None  # Track which service is using cameras
_lock_holder_lock = threading.Lock()  # Protect _active_service reads/writes

# Detect Jetson
IS_JETSON = os.path.exists('/etc/nv_tegra_release')


def acquire_camera_lock(service_name, timeout=5.0):
    """
    Acquire the global camera lock before opening any camera.
    
    Args:
        service_name: Name of the service requesting cameras (e.g., 'pothole', 'blindspot', 'dms')
        timeout: Maximum time to wait for lock
    
    Returns:
        True if lock acquired, False otherwise
    """
    global _active_service
    
    acquired = _camera_lock.acquire(timeout=timeout)
    if acquired:
        with _lock_holder_lock:
            _active_service = service_name
        print(f"🔒 Camera lock acquired by: {service_name}")
    else:
        with _lock_holder_lock:
            current_holder = _active_service
        print(f"⚠️ Failed to acquire camera lock for {service_name} (held by {current_holder})")
    
    return acquired


def release_camera_lock(service_name):
    """
    Release the global camera lock after closing cameras.
    This is idempotent - safe to call multiple times.
    
    Args:
        service_name: Name of the service releasing cameras
    """
    global _active_service
    
    with _lock_holder_lock:
        current_holder = _active_service
        
        # Only release if this service holds the lock
        if current_holder == service_name:
            _active_service = None
            try:
                _camera_lock.release()
                print(f"🔓 Camera lock released by: {service_name}")
            except RuntimeError:
                # Lock wasn't held - shouldn't happen but handle gracefully
                print(f"⚠️ Lock already released when {service_name} tried to release")
        elif current_holder is None:
            # Lock already released - this is OK (idempotent)
            pass
        else:
            # Different service holds the lock - this is a bug
            print(f"⚠️ {service_name} tried to release lock held by {current_holder}")


def force_release_camera(cap, service_name="unknown"):
    """
    Force release a camera with proper cleanup for Jetson.
    
    Args:
        cap: OpenCV VideoCapture object
        service_name: Name of the service for logging
    
    Returns:
        None (cap is released and should be set to None by caller)
    """
    if cap is None:
        return
    
    try:
        # Try to read any remaining frames to clear buffer
        for _ in range(3):
            cap.grab()
    except:
        pass
    
    try:
        cap.release()
    except:
        pass
    
    # Force garbage collection on Jetson
    if IS_JETSON:
        gc.collect()
        time.sleep(0.05)  # Small delay for camera driver to fully release
    
    print(f"📷 Camera released by: {service_name}")


def wait_for_camera_ready(camera_id, max_wait=2.0):
    """
    Wait for a camera to become available after another service released it.
    
    Args:
        camera_id: Camera index to check
        max_wait: Maximum time to wait in seconds
    
    Returns:
        True if camera is ready, False if timeout
    """
    if not IS_JETSON:
        return True  # Windows/Linux usually don't need this
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            # Quick test open
            test_cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            if test_cap.isOpened():
                test_cap.release()
                gc.collect()
                time.sleep(0.1)
                return True
            test_cap.release()
        except:
            pass
        time.sleep(0.1)
    
    return False


def cleanup_all_cameras():
    """
    Emergency cleanup - release all camera resources.
    Call this if switching gets stuck.
    This is idempotent - safe to call multiple times.
    """
    global _active_service
    
    print("🧹 Emergency camera cleanup...")
    
    # Safely release the lock if held
    with _lock_holder_lock:
        if _active_service is not None:
            previous_holder = _active_service
            _active_service = None
            try:
                _camera_lock.release()
                print(f"🔓 Emergency: Camera lock released (was held by {previous_holder})")
            except RuntimeError:
                print("⚠️ Lock release failed in emergency cleanup (already released)")
        else:
            # Check if lock is held without _active_service being set (corrupted state)
            if _camera_lock.locked():
                try:
                    _camera_lock.release()
                    print("🔓 Emergency: Released orphaned camera lock")
                except RuntimeError:
                    pass
            else:
                print("ℹ️ Camera lock already released")
    
    # Force garbage collection
    gc.collect()
    
    if IS_JETSON:
        time.sleep(0.2)
        gc.collect()
    
    print("✓ Emergency cleanup complete")
