import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
import os
import gc
from queue import Queue
import platform

# Jetson Orin Nano: Prevent segmentation faults
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
# Disable problematic CUDA optimizations on Jetson
if 'jetson' in platform.processor().lower() or os.path.exists('/etc/nv_tegra_release'):
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False  # Disable for stability on Jetson
    torch.backends.cudnn.deterministic = True
else:
    # Enable CUDA optimizations for other platforms
    torch.backends.cudnn.benchmark = True

# Global model instances for reuse (singleton pattern)
_global_detector = None
_detector_lock = threading.Lock()

# Configuration - OPTIMIZED FOR WINDOWS CPU
FRAME_W, FRAME_H = 480, 270  # Higher resolution for better quality (Windows can handle this)
YOLO_INPUT_W, YOLO_INPUT_H = 320, 180  # Small input for YOLO inference (CPU optimization)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, truck, bus, motorbike
DEPTH_SCALE = 38
NEAR_THRESHOLD = 3.0


def get_global_detector(device=None):
    """Get or create the global detector instance (singleton pattern for fast loading)"""
    global _global_detector
    with _detector_lock:
        if _global_detector is None:
            _global_detector = BlindSpotDetector(device)
    return _global_detector


class BlindSpotDetector:
    """Blind Spot Detection using YOLO and MiDaS depth estimation"""
    
    def __init__(self, device=None):
        print(f"🚀 Loading Blind Spot Detection models (JETSON ORIN OPTIMIZED)...")
        start_time = time.time()
        
        # Detect if running on Jetson device
        self.is_jetson = self._detect_jetson()
        
        # Set device with proper error handling
        if device:
            self.device = device
        elif torch.cuda.is_available() and self.is_jetson:
            self.device = 'cuda'
            print("✓ CUDA available on Jetson")
        else:
            self.device = 'cpu'
            print("⚠️ Using CPU mode")
        
        try:
            # Load YOLO model path from config
            from app.config import Config
            yolo_model_path = Config.YOLO_MODEL_PATH
            
            # Check if model file exists
            if not os.path.exists(yolo_model_path):
                print(f"⚠️ Model file not found: {yolo_model_path}")
                print(f"   Falling back to default yolov8n.pt")
                yolo_model_path = "yolov8n.pt"
            
            # Determine model type from file extension
            model_extension = os.path.splitext(yolo_model_path)[1].lower()
            
            # Load YOLO with optimizations - ADAPTIVE FOR CUDA/CPU
            if self.is_jetson:
                print(f"   Jetson detected - Loading YOLO model: {yolo_model_path}")
                
                if model_extension == '.engine':
                    # TensorRT engine for maximum Jetson performance
                    print("   ✓ Loading TensorRT engine (optimized for Jetson)")
                    self.yolo = YOLO(yolo_model_path, task='detect')
                else:
                    # PyTorch model fallback
                    print("   ⚠️ Loading PyTorch model (consider converting to .engine for better performance)")
                    self.yolo = YOLO(yolo_model_path)
                
                # Adaptive Jetson optimizations based on CUDA availability
                if self.device == 'cuda':
                    # CUDA MODE: Higher quality, GPU can handle larger inputs
                    self.yolo_img_size = 416  # Larger for better detection quality
                    self.conf_threshold = 0.35
                    self.iou_threshold = 0.7
                    self.frame_skip = 2  # Process every 2nd frame (15 FPS AI)
                    print(f"   ✓ CUDA MODE: {self.yolo_img_size}px input, conf={self.conf_threshold}, skip={self.frame_skip}")
                else:
                    # CPU MODE: Optimized for speed
                    self.yolo_img_size = 256  # Smaller for stability
                    self.conf_threshold = 0.35
                    self.iou_threshold = 0.7
                    self.frame_skip = 5  # Process every 5th frame (6 FPS AI)
                    print(f"   ✓ CPU MODE: {self.yolo_img_size}px input, conf={self.conf_threshold}, skip={self.frame_skip}")
            else:
                # Windows/Linux CPU optimizations
                print(f"   Loading YOLO model for CPU: {yolo_model_path}")
                self.yolo = YOLO(yolo_model_path)
                
                self.yolo_img_size = YOLO_INPUT_W  # Use 320px for CPU
                self.conf_threshold = 0.3
                self.iou_threshold = 0.7
                self.frame_skip = 3  # Process every 3rd frame
                
                # Fuse model for CPU optimization (only for PyTorch models)
                if model_extension == '.pt':
                    try:
                        self.yolo.fuse()
                        print("   ✓ YOLO model fused for CPU optimization")
                    except:
                        print("   ⚠️ Model fusion not available")
                
                print(f"   ✓ PC mode: {self.yolo_img_size}px input, conf={self.conf_threshold}")
            
            # Warm-up YOLO (prevents first-frame crash on Jetson)
            if self.is_jetson:
                try:
                    print("🔥 Warming up YOLO model...")
                    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    _ = self.detect_vehicles_only(dummy_frame)
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    print("✓ YOLO warmed up successfully")
                except Exception as e:
                    print(f"⚠️ Warm-up failed: {e}")
            
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            raise
        
        # SKIP MiDaS for performance - use YOLO-only detection
        self.use_depth = False
        
        load_time = time.time() - start_time
        print(f"✓ Models loaded in {load_time:.2f}s on {self.device}")
    
    def _detect_jetson(self):
        """Detect if running on Jetson device (Nano, Orin, Xavier, etc.)"""
        try:
            # Primary method: Check device tree model
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'jetson' in model or 'tegra' in model:
                        return True
            
            # Secondary method: Check for Tegra release file
            if os.path.exists('/etc/nv_tegra_release'):
                return True
            
            # Tertiary method: Check for NVIDIA SoC
            try:
                import subprocess
                result = subprocess.check_output(['cat', '/proc/cpuinfo'], 
                                               stderr=subprocess.DEVNULL, text=True)
                if 'tegra' in result.lower() or 'nvidia' in result.lower():
                    return True
            except:
                pass
            
            return False
        except:
            return False
    
    def detect_vehicles_only(self, frame):
        """Optimized YOLO inference - MEMORY LEAK PREVENTION"""
        try:
            # YOLO inference with optimized parameters
            with torch.no_grad():  # Prevent gradient accumulation
                # Determine if we can use FP16 (only on CUDA, not CPU)
                use_half = self.is_jetson and self.device == 'cuda'
                
                if self.is_jetson:
                    # Jetson: Use FP16 ONLY if CUDA is available
                    results = self.yolo.predict(
                        frame, 
                        imgsz=self.yolo_img_size,
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False,
                        half=use_half,  # FP16 only on CUDA
                        device=self.device,
                        save=False,  # Don't save results to disk
                        stream=True,  # Stream results for memory efficiency
                        classes=VEHICLE_CLASSES,  # Only detect vehicles
                        agnostic_nms=True  # Faster NMS
                    )
                    results = next(results)  # Get first result from stream
                else:
                    # Windows/Linux CPU: Optimized for maximum speed
                    results = self.yolo.predict(
                        frame,
                        imgsz=self.yolo_img_size, 
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        verbose=False,
                        half=False,  # CPU doesn't support FP16
                        device='cpu',  # Force CPU for consistency
                        classes=VEHICLE_CLASSES,  # Only detect vehicles (faster)
                        save=False,  # Don't save results to disk
                        stream=True,  # Stream results for memory efficiency
                        agnostic_nms=True  # Faster NMS
                    )
                    results = next(results)  # Get first result from stream
            
            # Extract vehicle detections only
            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls in VEHICLE_CLASSES:
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = float(box.conf[0])
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': confidence,
                            'class': cls
                        })
            
            # Explicit cleanup (critical for Jetson stability)
            del results
            if self.is_jetson and self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return detections
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            return []
    
    def draw_side_mirror_grid(self, frame, color, is_left=True):
        """ULTRA SIMPLE blind spot detection area - MAXIMUM SPEED"""
        h, w = frame.shape[:2]
        
        # Simple rectangular detection zones 
        if is_left:
            # Left blind spot area
            x1, y1 = int(w * 0.1), int(h * 0.3)
            x2, y2 = int(w * 0.6), int(h * 0.8)
        else:
            # Right blind spot area  
            x1, y1 = int(w * 0.4), int(h * 0.3)
            x2, y2 = int(w * 0.9), int(h * 0.8)
        
        # Draw only simple rectangle (no text for speed)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Return simple polygon for detection
        polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return polygon
    
    def inside(self, poly, cx, cy):
        """Check if point is inside blind-spot cage"""
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
    
    def process_frame_fast(self, frame, is_left=True):
        """ULTRA FAST processing - returns only danger status"""
        
        # Resize for YOLO inference based on platform (CRITICAL CPU OPTIMIZATION)
        if self.is_jetson:
            # Jetson: Balanced size (320x240)
            frame_small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
            scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 240
        else:
            # Windows CPU: Use optimized size (320x180 for best CPU performance)
            frame_small = cv2.resize(frame, (YOLO_INPUT_W, YOLO_INPUT_H), interpolation=cv2.INTER_LINEAR)
            scale_x, scale_y = frame.shape[1] / YOLO_INPUT_W, frame.shape[0] / YOLO_INPUT_H
        
        # Get detections (no drawing, just data)
        detections = self.detect_vehicles_only(frame_small)
        
        danger = False
        detected_vehicles = []
        
        # Check if any vehicle is in blind spot zone
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Scale back to original frame size
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Check blind spot zones on original frame
            h, w = frame.shape[:2]
            if is_left:
                in_zone = (w * 0.1 < cx < w * 0.6) and (h * 0.3 < cy < h * 0.8)
            else:
                in_zone = (w * 0.4 < cx < w * 0.9) and (h * 0.3 < cy < h * 0.8)
            
            if in_zone:
                danger = True
                detected_vehicles.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': det['confidence'],
                    'in_blind_spot': True
                })
        
        return danger, detected_vehicles


class DualCameraManager:
    """
    DUAL CAMERA OPTIMIZED: Camera 0 (Left) + Camera 1 (Right) with CPU Optimizations
    
    ULTRA-OPTIMIZED FOR WINDOWS CPU + DUAL CAMERAS:
    - Two camera threads: Read + Encode frames for each camera (FAST)
    - Two AI threads: YOLO detection for each side (Background)
    - Frame skipping: Process every 3rd frame (10 FPS AI vs 30 FPS video)
    - Small YOLO input: 320x180 for maximum CPU performance
    - MJPEG streaming: Direct binary, no base64 conversion
    - Smart fallback: If dual cameras fail, use single camera
    
    Result: Smooth dual camera performance on Windows CPU!
    """
    
    def __init__(self, left_cam_id=0, right_cam_id=1):
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        
        # Dual cameras (camera 0 for left, camera 1 for right)
        self.left_cap = None
        self.right_cap = None
        
        self.detector = None
        self.is_jetson = False  # Will be detected
        self.dual_camera_mode = True  # Will be set by _open_cameras()
        
        # Dual camera frame buffers
        self.left_latest_frame = None
        self.right_latest_frame = None
        self.left_encoded = None
        self.right_encoded = None
        self.left_danger = False
        self.right_danger = False
        
        self.running = False
        self.active = False
        
        # 4 threads total (2 per camera) for dual camera setup
        self.left_video_thread = None
        self.left_ai_thread = None
        self.right_video_thread = None
        self.right_ai_thread = None
        
        self.left_lock = threading.Lock()
        self.right_lock = threading.Lock()
        
        # Performance monitoring
        self.start_time = None
        self.frame_count = 0
        self.last_fps_report = 0
    
    def _detect_jetson_nano(self):
        """Detect if running on Jetson Nano"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                return 'jetson' in model
        except:
            try:
                with open('/etc/nv_tegra_release', 'r') as f:
                    return True
            except:
                return False
    
    def _get_jetson_camera_pipeline(self, camera_id=0):
        """Get optimized camera pipeline for Jetson - V4L2 PRIORITIZED (TESTED & WORKING)"""
        # Since diagnostic shows V4L2 works but GStreamer doesn't, prioritize V4L2
        
        # Priority 1: V4L2 Direct (TESTED - WORKS!)
        # Priority 2: CAP_ANY (TESTED - WORKS!)
        # Priority 3-6: GStreamer pipelines (if available)
        # Priority 7: Direct index (last resort)
        
        # GStreamer pipelines (kept for future if GStreamer gets installed)
        gst_csi = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1,format=NV12 ! "
            "nvvidconv ! "
            "video/x-raw,width=480,height=270,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        gst_usb_mjpeg_hw = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "image/jpeg,width=640,height=480,framerate=30/1 ! "
            "nvjpegdec ! "
            "videoscale ! "
            "video/x-raw,width=480,height=270 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        gst_usb_mjpeg_sw = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "image/jpeg,width=640,height=480,framerate=30/1 ! "
            "jpegdec ! "
            "videoscale ! "
            "video/x-raw,width=480,height=270 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        gst_usb_yuyv = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "videoscale ! "
            "video/x-raw,width=480,height=270 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        # OPTIMIZED ORDER: V4L2 first (proven to work), then GStreamer
        return [
            (camera_id, cv2.CAP_V4L2),               # Priority 1: V4L2 (WORKS!)
            (camera_id, cv2.CAP_ANY),                # Priority 2: CAP_ANY (WORKS!)
            (gst_csi, cv2.CAP_GSTREAMER),            # Priority 3: CSI camera
            (gst_usb_mjpeg_hw, cv2.CAP_GSTREAMER),   # Priority 4: GStreamer MJPEG HW
            (gst_usb_mjpeg_sw, cv2.CAP_GSTREAMER),   # Priority 5: GStreamer MJPEG SW
            (gst_usb_yuyv, cv2.CAP_GSTREAMER),       # Priority 6: GStreamer YUYV
            (camera_id, None)                        # Priority 7: Direct index
        ]
    
    def _detect_available_cameras(self):
        """Detect available cameras - Use requested camera IDs"""
        available_cameras = []
        
        print("🔍 Detecting available cameras...")
        
        # Check if running on Jetson Nano
        self.is_jetson = self._detect_jetson_nano()
        
        # Use the camera IDs that were requested (left_cam_id and right_cam_id)
        requested_cameras = [self.left_cam_id, self.right_cam_id]
        print(f"    📷 Testing cameras: {requested_cameras}")
        
        if self.is_jetson:
            print("    📹 Jetson detected - Testing V4L2 backends (proven to work)")
            
            # Test cameras with working backends: V4L2 and CAP_ANY
            for camera_id in requested_cameras:
                camera_opened = False
                
                # Try V4L2 first (TESTED - WORKS!)
                try:
                    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            available_cameras.append(camera_id)
                            print(f"   ✓ Camera {camera_id} available (V4L2)")
                            cap.release()
                            camera_opened = True
                        else:
                            cap.release()
                except Exception as e:
                    pass
                
                # Try CAP_ANY as backup (TESTED - WORKS!)
                if not camera_opened:
                    try:
                        cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                available_cameras.append(camera_id)
                                print(f"   ✓ Camera {camera_id} available (CAP_ANY)")
                                cap.release()
                                camera_opened = True
                        else:
                            cap.release()
                    except Exception as e:
                        pass
                
                # Try direct index as last resort (TESTED - WORKS!)
                if not camera_opened:
                    try:
                        cap = cv2.VideoCapture(camera_id)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                available_cameras.append(camera_id)
                                print(f"   ✓ Camera {camera_id} available (direct index)")
                                cap.release()
                        else:
                            cap.release()
                    except Exception as e:
                        pass
        else:
            # Windows/Linux - test requested cameras
            print("   💻 Testing requested camera IDs")
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
            
            for camera_id in requested_cameras:
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(camera_id, backend)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                if camera_id not in available_cameras:
                                    available_cameras.append(camera_id)
                                    print(f"   ✓ Camera {camera_id} available")
                                cap.release()
                                break
                        cap.release()
                    except:
                        pass
        
        print(f"   Found {len(available_cameras)} camera(s): {available_cameras}")
        return available_cameras
    
    def _open_cameras(self):
        """Open dual cameras with smart fallback to single camera"""
        if self.left_cap is not None and self.right_cap is not None:
            if self.left_cap.isOpened() and self.right_cap.isOpened():
                return True
        
        # Detect available cameras first
        available_cameras = self._detect_available_cameras()
        
        if len(available_cameras) == 0:
            print(" No physical cameras found!")
            print("   Tip: Make sure cameras are connected and not used by other apps")
            return False
        
        start_time = time.time()
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        
        if len(available_cameras) >= 2:
            # Try to use two separate cameras (ideal setup)
            print(f"📹 Opening dual cameras {available_cameras[0]} and {available_cameras[1]}...")
            
            try:
                self.left_cap = cv2.VideoCapture(available_cameras[0], backend)
                self.right_cap = cv2.VideoCapture(available_cameras[1], backend)
                
                if self.left_cap.isOpened() and self.right_cap.isOpened():
                    self.dual_camera_mode = True
                    print("   Dual camera mode: Left and Right cameras")
                else:
                    # Fallback to single camera
                    if self.left_cap: self.left_cap.release()
                    if self.right_cap: self.right_cap.release()
                    raise Exception("Dual camera initialization failed")
                    
            except Exception as e:
                print(f"   Dual camera failed: {e}")
                print("   Falling back to single camera mode...")
                
                # Single camera fallback
                self.left_cap = cv2.VideoCapture(available_cameras[0], backend)
                self.right_cap = cv2.VideoCapture(available_cameras[0], backend)
                self.dual_camera_mode = False
                print("   Single camera mode: Same feed for both sides")
        else:
            # Only one camera available - use it for both sides
            print(f"📹 Opening single camera {available_cameras[0]} for both sides...")
            
            # Try to open camera with working backends
            if self.is_jetson:
                # On Jetson, use proven backends: V4L2, CAP_ANY, then GStreamer
                pipelines = self._get_jetson_camera_pipeline(available_cameras[0])
                camera_opened = False
                
                for idx, (pipeline, backend_type) in enumerate(pipelines):
                    try:
                        # Handle None backend (direct index)
                        if backend_type is None:
                            test_cap = cv2.VideoCapture(pipeline)
                        else:
                            test_cap = cv2.VideoCapture(pipeline, backend_type)
                        
                        if test_cap.isOpened():
                            ret, frame = test_cap.read()
                            if ret and frame is not None:
                                # Success! Share this camera for both left and right
                                self.left_cap = test_cap
                                self.right_cap = self.left_cap  # Share the same object
                                camera_opened = True
                                
                                backend_name = "V4L2" if backend_type == cv2.CAP_V4L2 else \
                                              "CAP_ANY" if backend_type == cv2.CAP_ANY else \
                                              "GStreamer" if backend_type == cv2.CAP_GSTREAMER else \
                                              "Direct"
                                print(f"   ✓ Camera opened successfully (backend: {backend_name})")
                                break
                        test_cap.release()
                    except Exception as e:
                        pass
                
                if not camera_opened:
                    print(f"   ❌ Failed to open camera with any backend")
            else:
                # Windows/Linux
                self.left_cap = cv2.VideoCapture(available_cameras[0], backend)
                if self.left_cap.isOpened():
                    # Share the same capture object for single camera
                    self.right_cap = self.left_cap
                    camera_opened = True
                else:
                    self.right_cap = None
            
            self.dual_camera_mode = False
            print("   Single camera mode: Same feed for both sides")
        
        # Configure both cameras with ANTI-LAG optimizations
        for cap in [self.left_cap, self.right_cap]:
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CRITICAL: No buffering = no lag accumulation
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)  # 480x270 for good quality
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try MJPEG for better performance
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except:
                    pass
                
                # Additional Windows-specific anti-lag settings
                if platform.system() == "Windows":
                    try:
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto-exposure for consistent frame timing
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable autofocus to prevent processing delays
                    except:
                        pass  # Some cameras don't support these properties
        
        # Verify cameras are working
        if not self.left_cap or not self.left_cap.isOpened() or not self.right_cap or not self.right_cap.isOpened():
            print("Failed to open cameras")
            return False
        
        open_time = time.time() - start_time
        print(f"Cameras opened in {open_time:.2f}s")
        return True
    
    def start(self):
        """Start dual camera blind spot detection with CPU optimizations"""
        if self.active:
            return True
        
        if not self._open_cameras():
            return False
        
        # Load detector
        if self.detector is None:
            self.detector = get_global_detector()
        
        self.running = True
        self.active = True
        
        # Clear any GPU/CPU cache before starting (anti-lag optimization)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU cache cleared")
            import gc
            gc.collect()
            print("Memory garbage collected")
        except:
            pass
        
        # Start 4 threads for dual camera setup
        self.left_video_thread = threading.Thread(target=self._left_video_loop, daemon=True)
        self.left_ai_thread = threading.Thread(target=self._left_ai_loop, daemon=True)
        self.right_video_thread = threading.Thread(target=self._right_video_loop, daemon=True)
        self.right_ai_thread = threading.Thread(target=self._right_ai_loop, daemon=True)
        
        # Start all threads
        self.left_video_thread.start()
        self.left_ai_thread.start()
        self.right_video_thread.start()
        self.right_ai_thread.start()
        
        mode_text = "Dual camera mode" if self.dual_camera_mode else "Single camera mode (both sides)"
        print(f"Blind spot detection started with 4 threads! ({mode_text})")
        print("   Left camera: Video + AI threads")  
        print("   Right camera: Video + AI threads")
        print("   Expected performance: 15-30 FPS video + 10 FPS AI per camera")
        return True
    
    def _left_video_loop(self):
        """Left camera video thread - OPTIMIZED FOR JETSON"""
        frame_counter = 0
        
        while self.running:
            if not self.left_cap or not self.left_cap.isOpened():
                time.sleep(0.01)
                continue
            
            # Read latest frame (buffer=1 prevents accumulation)
            ret, frame = self.left_cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            
            frame_counter += 1
            
            # CRITICAL: Frame skipping for Jetson to reduce encoding load
            # Skip every other frame = 15 FPS (still smooth, much less lag)
            if self.is_jetson and frame_counter % 2 != 0:
                del frame
                continue
            
            # Prevent counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Store frame for AI thread (minimal lock time)
            with self.left_lock:
                if self.left_latest_frame is not None:
                    del self.left_latest_frame
                self.left_latest_frame = frame.copy()
                danger = self.left_danger
            
            # Draw grid overlay
            color = (0, 0, 255) if danger else (0, 255, 0)
            self.detector.draw_side_mirror_grid(frame, color, is_left=True)
            
            # AGGRESSIVE Jetson optimization - Lower resolution for smooth streaming
            if self.is_jetson:
                encode_size = (320, 180)   # Much smaller for Jetson (was 480x270)
                encode_quality = 50        # Lower quality for maximum speed
            else:
                encode_size = (400, 225)   # Smaller for CPU
                encode_quality = 50
            
            frame_resized = cv2.resize(frame, encode_size, interpolation=cv2.INTER_LINEAR)
            ret_encode, buffer = cv2.imencode('.jpg', frame_resized, 
                                              [cv2.IMWRITE_JPEG_QUALITY, encode_quality,
                                               cv2.IMWRITE_JPEG_OPTIMIZE, 1])  # Fast encoding
            
            del frame_resized
            
            if ret_encode:
                with self.left_lock:
                    if self.left_encoded is not None:
                        del self.left_encoded
                    self.left_encoded = buffer.tobytes()
                del buffer
            
            del frame
            
            # Small sleep to prevent CPU spinning (Jetson optimization)
            time.sleep(0.001)
    
    def _right_video_loop(self):
        """Right camera video thread - OPTIMIZED FOR JETSON"""
        frame_counter = 0
        
        while self.running:
            if not self.right_cap or not self.right_cap.isOpened():
                time.sleep(0.01)
                continue
            
            # Read latest frame (buffer=1 prevents accumulation)
            ret, frame = self.right_cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            
            frame_counter += 1
            
            # CRITICAL: Frame skipping for Jetson to reduce encoding load
            # Skip every other frame = 15 FPS (still smooth, much less lag)
            if self.is_jetson and frame_counter % 2 != 0:
                del frame
                continue
            
            # Prevent counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Store frame for AI thread (minimal lock time)
            with self.right_lock:
                if self.right_latest_frame is not None:
                    del self.right_latest_frame
                self.right_latest_frame = frame.copy()
                danger = self.right_danger
            
            # Draw grid overlay
            color = (0, 0, 255) if danger else (0, 255, 0)
            self.detector.draw_side_mirror_grid(frame, color, is_left=False)
            
            # AGGRESSIVE Jetson optimization - Lower resolution for smooth streaming
            if self.is_jetson:
                encode_size = (320, 180)   # Much smaller for Jetson (was 480x270)
                encode_quality = 50        # Lower quality for maximum speed
            else:
                encode_size = (400, 225)   # Smaller for CPU
                encode_quality = 50
            
            frame_resized = cv2.resize(frame, encode_size, interpolation=cv2.INTER_LINEAR)
            ret_encode, buffer = cv2.imencode('.jpg', frame_resized, 
                                              [cv2.IMWRITE_JPEG_QUALITY, encode_quality,
                                               cv2.IMWRITE_JPEG_OPTIMIZE, 1])  # Fast encoding
            
            del frame_resized
            
            if ret_encode:
                with self.right_lock:
                    if self.right_encoded is not None:
                        del self.right_encoded
                    self.right_encoded = buffer.tobytes()
                del buffer
            
            del frame
            
            # Small sleep to prevent CPU spinning (Jetson optimization)
            time.sleep(0.001)
    
    def _left_ai_loop(self):
        """Left AI thread - MEMORY LEAK PREVENTION & CPU OPTIMIZED"""
        frame_counter = 0
        last_gc = time.time()
        
        while self.running:
            # Frame skipping for Windows CPU (process every 3rd frame = 10 FPS AI)
            frame_counter += 1
            if frame_counter % 3 != 0:
                time.sleep(0.02)  # Small delay when skipping
                continue
            
            # Periodic garbage collection to prevent memory accumulation
            current_time = time.time()
            if current_time - last_gc > 60:  # Every 60 seconds
                import gc
                gc.collect()
                last_gc = current_time
                print("🗑️ Left AI: Garbage collection performed")
            
            # Prevent frame counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Get latest frame
            with self.left_lock:
                if self.left_latest_frame is None:
                    time.sleep(0.01)
                    continue
                # Process smaller frame for maximum speed (320x180 for CPU)
                frame_small = cv2.resize(self.left_latest_frame, (YOLO_INPUT_W, YOLO_INPUT_H), 
                                       interpolation=cv2.INTER_LINEAR)
            
            try:
                # Run AI detection (ultra fast on small frame)
                danger, detected_vehicles = self.detector.process_frame_fast(frame_small, is_left=True)
                
                # Explicitly delete variables to free memory
                del detected_vehicles
                del frame_small
                
                # Update danger status
                with self.left_lock:
                    self.left_danger = danger
                    
            except Exception as e:
                # Graceful error handling with memory cleanup
                print(f"Left AI error: {e}")
                # Clean up any partial variables
                if 'frame_small' in locals():
                    del frame_small
                time.sleep(0.1)
    
    def _right_ai_loop(self):
        """Right AI thread - MEMORY LEAK PREVENTION & CPU OPTIMIZED"""
        frame_counter = 0
        last_gc = time.time()
        
        while self.running:
            # Frame skipping for Windows CPU (process every 3rd frame = 10 FPS AI)
            frame_counter += 1
            if frame_counter % 3 != 0:
                time.sleep(0.02)  # Small delay when skipping
                continue
            
            # Periodic garbage collection to prevent memory accumulation
            current_time = time.time()
            if current_time - last_gc > 60:  # Every 60 seconds
                import gc
                gc.collect()
                last_gc = current_time
                print("Right AI: Garbage collection performed")
            
            # Prevent frame counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Get latest frame
            with self.right_lock:
                if self.right_latest_frame is None:
                    time.sleep(0.01)
                    continue
                # Process smaller frame for maximum speed (320x180 for CPU)
                frame_small = cv2.resize(self.right_latest_frame, (YOLO_INPUT_W, YOLO_INPUT_H), 
                                       interpolation=cv2.INTER_LINEAR)
            
            try:
                # Run AI detection (ultra fast on small frame)
                danger, detected_vehicles = self.detector.process_frame_fast(frame_small, is_left=False)
                
                # Explicitly delete variables to free memory
                del detected_vehicles
                del frame_small
                
                # Update danger status
                with self.right_lock:
                    self.right_danger = danger
                    
            except Exception as e:
                # Graceful error handling with memory cleanup
                print(f"Right AI error: {e}")
                # Clean up any partial variables
                if 'frame_small' in locals():
                    del frame_small
                time.sleep(0.1)
    
    def get_left_frame(self):
        """Get encoded left camera frame"""
        with self.left_lock:
            return self.left_encoded
    
    def get_right_frame(self):
        """Get encoded right camera frame"""
        with self.right_lock:
            return self.right_encoded
    
    def is_left_danger(self):
        """Check if left blind spot has danger"""
        with self.left_lock:
            return bool(self.left_danger)
    
    def is_right_danger(self):
        """Check if right blind spot has danger"""
        with self.right_lock:
            return bool(self.right_danger)
    
    def stop(self):
        """Stop dual camera blind spot detection (4 threads)"""
        self.running = False
        self.active = False
        
        # Wait for all threads to finish
        if self.left_video_thread:
            self.left_video_thread.join(timeout=2)
        if self.left_ai_thread:
            self.left_ai_thread.join(timeout=2)
        if self.right_video_thread:
            self.right_video_thread.join(timeout=2)
        if self.right_ai_thread:
            self.right_ai_thread.join(timeout=2)
        
        # Release cameras
        if self.left_cap:
            self.left_cap.release()
            self.left_cap = None
        if self.right_cap:
            self.right_cap.release()
            self.right_cap = None
        
        print("Dual camera blind spot detection stopped (Windows CPU optimized)")
    
    def is_active(self):
        """Check if detection is active"""
        return self.active
    
    def _monitor_performance(self):
        """Monitor system performance and memory usage"""
        try:
            import psutil
            import os
            
            # Get current process
            process = psutil.Process(os.getpid())
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Calculate FPS
            current_time = time.time()
            if self.start_time is None:
                self.start_time = current_time
                self.last_fps_report = current_time
                return
            
            self.frame_count += 1
            
            # Report every 30 seconds
            if current_time - self.last_fps_report > 30:
                elapsed_time = current_time - self.start_time
                avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                print(f"Performance Report:")
                print(f"   Memory: {memory_mb:.1f} MB")
                print(f"   CPU: {cpu_percent:.1f}%")
                print(f"   Avg FPS: {avg_fps:.1f}")
                print(f"   Uptime: {elapsed_time/60:.1f} min")
                
                # Reset FPS counter to get recent FPS
                self.frame_count = 0
                self.start_time = current_time
                self.last_fps_report = current_time
                
                # Memory warning
                if memory_mb > 500:  # More than 500MB
                    print("High memory usage detected! Consider restarting.")
                    
        except ImportError:
            # psutil not available, skip monitoring
            pass
        except Exception as e:
            print(f"Performance monitoring error: {e}")
