#!/usr/bin/env python3
"""
Minimal test script for Jetson Orin Nano
Tests each component separately to identify segfault source
"""

import sys
import os

print("=" * 60)
print("JETSON ORIN NANO - COMPONENT TEST")
print("=" * 60)

# Test 1: Basic imports
print("\n[Test 1] Basic Python imports...")
try:
    import numpy as np
    import time
    import threading
    print("✓ Basic imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    sys.exit(1)

# Test 2: OpenCV
print("\n[Test 2] OpenCV import...")
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} imported successfully")
except Exception as e:
    print(f"❌ OpenCV import failed: {e}")
    sys.exit(1)

# Test 3: PyTorch
print("\n[Test 3] PyTorch import...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

# Test 4: Camera detection
print("\n[Test 4] Camera detection...")
try:
    # Check video devices
    video_devices = []
    for i in range(4):
        if os.path.exists(f'/dev/video{i}'):
            video_devices.append(i)
    
    if video_devices:
        print(f"✓ Found video devices: {video_devices}")
    else:
        print("⚠️  No video devices found")
except Exception as e:
    print(f"❌ Camera detection failed: {e}")

# Test 5: GStreamer
print("\n[Test 5] GStreamer support...")
try:
    # Test if GStreamer backend is available
    gst_available = cv2.getBuildInformation().find('GStreamer') != -1
    if gst_available:
        print("✓ OpenCV built with GStreamer support")
    else:
        print("⚠️  OpenCV not built with GStreamer (USB cameras may not work optimally)")
except Exception as e:
    print(f"⚠️  Could not check GStreamer: {e}")

# Test 6: Simple camera open
print("\n[Test 6] Camera open test...")
camera_opened = False
for cam_id in [0, 1]:
    try:
        print(f"  Trying camera {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        time.sleep(0.5)  # Give it time to initialize
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ Camera {cam_id} working: {frame.shape}")
                camera_opened = True
                cap.release()
                break
            else:
                print(f"  ⚠️  Camera {cam_id} opened but can't read frames")
        else:
            print(f"  ❌ Camera {cam_id} failed to open")
        
        cap.release()
    except Exception as e:
        print(f"  ❌ Camera {cam_id} error: {e}")

if not camera_opened:
    print("\n⚠️  Warning: No working cameras found!")

# Test 7: YOLO model loading
print("\n[Test 7] YOLO model loading...")
try:
    from ultralytics import YOLO
    print("  Importing YOLO...")
    
    if not os.path.exists('yolov8n.pt'):
        print("  ⚠️  yolov8n.pt not found, downloading...")
    
    model = YOLO("yolov8n.pt")
    print(f"  ✓ YOLO model loaded")
    
    # Test inference
    print("  Testing inference...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model.predict(dummy_frame, imgsz=256, verbose=False)
    print("  ✓ YOLO inference successful")
    
except Exception as e:
    print(f"  ❌ YOLO test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Segmentation model (if model file exists)
print("\n[Test 8] Segmentation model loading...")
try:
    # Check if model file exists
    model_files = [
        'pothole_segmentation_model.pth',
        'best_model.pth',
        'model.pth'
    ]
    
    model_path = None
    for mf in model_files:
        if os.path.exists(mf):
            model_path = mf
            break
    
    if model_path:
        import segmentation_models_pytorch as smp
        print(f"  Loading model from {model_path}...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print(f"  ✓ Segmentation model loaded on {device}")
        
        # Test inference
        print("  Testing inference...")
        dummy_input = torch.randn(1, 3, 96, 96).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print("  ✓ Segmentation inference successful")
        
    else:
        print("  ⚠️  No segmentation model file found (this is OK for testing)")
        
except Exception as e:
    print(f"  ❌ Segmentation model test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Memory check
print("\n[Test 9] System memory...")
try:
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
    
    for line in meminfo.split('\n'):
        if any(x in line for x in ['MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree']):
            print(f"  {line}")
    
    # CUDA memory if available
    if torch.cuda.is_available():
        print(f"\n  CUDA Memory:")
        print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"    Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
        
except Exception as e:
    print(f"  ⚠️  Could not check memory: {e}")

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("\nIf all tests passed, the system is ready!")
print("\nNext steps:")
print("1. cd Backend")
print("2. python3 run.py")
print("\nIf you get a segfault, note which test failed above.")
print("=" * 60)
