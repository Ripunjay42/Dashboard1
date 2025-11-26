#!/usr/bin/env python3
"""
OpenCV GStreamer Support Diagnostic
Tests if OpenCV was compiled with GStreamer support
"""

import cv2
import sys

print("=" * 60)
print("OpenCV GStreamer Support Diagnostic")
print("=" * 60)
print()

# Check OpenCV version
print(f"✓ OpenCV Version: {cv2.__version__}")
print()

# Check build information
build_info = cv2.getBuildInformation()

# Check for GStreamer in build info
has_gstreamer = 'GStreamer' in build_info and 'YES' in build_info

print("📹 Video I/O Backends:")
print("-" * 60)

# Extract relevant lines from build info
for line in build_info.split('\n'):
    line_lower = line.lower()
    if any(keyword in line_lower for keyword in ['gstreamer', 'v4l', 'video', 'ffmpeg', 'backend']):
        if 'gstreamer' in line_lower:
            if 'yes' in line_lower:
                print(f"  ✅ {line.strip()}")
            else:
                print(f"  ❌ {line.strip()}")
        else:
            print(f"     {line.strip()}")

print()
print("=" * 60)

# Try to open a test GStreamer pipeline
print("\n🧪 Testing GStreamer Pipeline:")
print("-" * 60)

test_pipeline = "videotestsrc num-buffers=1 ! video/x-raw,width=640,height=480 ! appsink"

try:
    cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"  ✅ GStreamer pipeline opened successfully!")
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Frame read successfully: {frame.shape}")
        else:
            print(f"  ⚠️  Pipeline opened but couldn't read frame")
        cap.release()
    else:
        print(f"  ❌ Failed to open GStreamer pipeline")
        print(f"  💡 OpenCV may not be compiled with GStreamer support")
except Exception as e:
    print(f"  ❌ Exception: {e}")
    print(f"  💡 OpenCV not compiled with GStreamer support")

print()
print("=" * 60)
print("\n🧪 Testing V4L2 Backend:")
print("-" * 60)

try:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"  ✅ V4L2 backend opened camera 0")
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Frame read successfully: {frame.shape}")
        else:
            print(f"  ⚠️  Camera opened but couldn't read frame")
        cap.release()
    else:
        print(f"  ❌ Failed to open camera with V4L2")
except Exception as e:
    print(f"  ❌ Exception: {e}")

print()
print("=" * 60)
print("\n🧪 Testing CAP_ANY Backend:")
print("-" * 60)

try:
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if cap.isOpened():
        print(f"  ✅ CAP_ANY backend opened camera 0")
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Frame read successfully: {frame.shape}")
        else:
            print(f"  ⚠️  Camera opened but couldn't read frame")
        cap.release()
    else:
        print(f"  ❌ Failed to open camera with CAP_ANY")
except Exception as e:
    print(f"  ❌ Exception: {e}")

print()
print("=" * 60)
print("\n🧪 Testing Direct Camera Index:")
print("-" * 60)

try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print(f"  ✅ Direct index opened camera 0")
        ret, frame = cap.read()
        if ret:
            print(f"  ✅ Frame read successfully: {frame.shape}")
        else:
            print(f"  ⚠️  Camera opened but couldn't read frame")
        cap.release()
    else:
        print(f"  ❌ Failed to open camera with direct index")
except Exception as e:
    print(f"  ❌ Exception: {e}")

print()
print("=" * 60)
print("\n📋 Summary:")
print("-" * 60)

if has_gstreamer:
    print("✅ OpenCV compiled WITH GStreamer support")
    print("💡 Use GStreamer pipelines for best performance")
else:
    print("❌ OpenCV compiled WITHOUT GStreamer support")
    print("💡 Recommendation: Install opencv-python from source with GStreamer")
    print("   Or use pre-built Jetson packages:")
    print("   sudo apt-get install python3-opencv")

print()
print("🔧 Recommended backend for your system:")
print("   Use: cv2.VideoCapture(0, cv2.CAP_ANY)")
print("   Or:  cv2.VideoCapture(0)")
print()
