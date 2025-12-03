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
        _active_service = service_name
        print(f"🔒 Camera lock acquired by: {service_name}")
    else:
        print(f"⚠️ Failed to acquire camera lock for {service_name} (held by {_active_service})")
    
    return acquired


def release_camera_lock(service_name):
    """
    Release the global camera lock after closing cameras.
    
    Args:
        service_name: Name of the service releasing cameras
    """
    global _active_service
    
    try:
        if _active_service == service_name:
            _active_service = None
            _camera_lock.release()
            print(f"🔓 Camera lock released by: {service_name}")
        else:
            print(f"⚠️ {service_name} tried to release lock held by {_active_service}")
    except RuntimeError:
        # Lock wasn't held
        pass


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
    """
    global _active_service
    
    print("🧹 Emergency camera cleanup...")
    
    # Only try to release lock if it's actually held
    try:
        if _camera_lock.locked():
            _active_service = None
            _camera_lock.release()
            print("🔓 Emergency: Camera lock released")
        else:
            print("ℹ️ Camera lock already released")
    except RuntimeError:
        print("⚠️ Lock release failed in emergency cleanup (already released)")
    
    # Force garbage collection
    gc.collect()
    
    if IS_JETSON:
        time.sleep(0.2)
        gc.collect()
    
    print("✓ Emergency cleanup complete")
