#!/usr/bin/env python3
"""
Quick script to check all available cameras on Jetson
Run this to see which cameras are actually connected
"""

import cv2
import os

print("=" * 60)
print("CAMERA DETECTION TEST - Jetson Orin Nano")
print("=" * 60)

# Check if running on Jetson
is_jetson = os.path.exists('/etc/nv_tegra_release')
print(f"\nRunning on Jetson: {is_jetson}")

# Test cameras 0-4
available_cameras = []

for camera_id in range(5):
    print(f"\n📹 Testing Camera {camera_id}...")
    
    # Try V4L2 backend (works on Jetson)
    try:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"   ✅ Camera {camera_id} AVAILABLE (V4L2)")
                print(f"      Resolution: {w}x{h}")
                available_cameras.append(camera_id)
                cap.release()
                continue
        cap.release()
    except Exception as e:
        print(f"   ❌ V4L2 failed: {e}")
    
    # Try CAP_ANY backend
    try:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"   ✅ Camera {camera_id} AVAILABLE (CAP_ANY)")
                print(f"      Resolution: {w}x{h}")
                if camera_id not in available_cameras:
                    available_cameras.append(camera_id)
                cap.release()
                continue
        cap.release()
    except Exception as e:
        print(f"   ❌ CAP_ANY failed: {e}")
    
    # Try direct index
    try:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"   ✅ Camera {camera_id} AVAILABLE (Direct)")
                print(f"      Resolution: {w}x{h}")
                if camera_id not in available_cameras:
                    available_cameras.append(camera_id)
                cap.release()
                continue
        cap.release()
    except Exception as e:
        print(f"   ❌ Direct failed: {e}")
    
    print(f"   ❌ Camera {camera_id} NOT AVAILABLE")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n✅ Available cameras: {available_cameras}")
print(f"   Total: {len(available_cameras)} camera(s)")

print("\n📋 Recommended Configuration:")
if len(available_cameras) >= 3:
    print(f"   - Pothole Detection: Camera {available_cameras[0]}")
    print(f"   - Blind Spot Left:   Camera {available_cameras[1]}")
    print(f"   - Blind Spot Right:  Camera {available_cameras[2]}")
elif len(available_cameras) >= 2:
    print(f"   - Pothole Detection: Camera {available_cameras[0]}")
    print(f"   - Blind Spot (both): Cameras {available_cameras[0]} and {available_cameras[1]}")
elif len(available_cameras) >= 1:
    print(f"   ⚠️  Only 1 camera found - will use same camera for all features")
    print(f"   - All features: Camera {available_cameras[0]}")
else:
    print("   ❌ No cameras detected!")

print("\n" + "=" * 60)

# Also check /dev/video* devices
print("\n🔍 Checking /dev/video* devices...")
import glob
video_devices = glob.glob('/dev/video*')
if video_devices:
    print(f"   Found devices: {video_devices}")
else:
    print("   No /dev/video* devices found")

print("\n" + "=" * 60)
