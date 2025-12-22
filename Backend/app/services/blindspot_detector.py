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

# Configuration - OPTIMIZED FOR JETSON
# Detect platform once at module level
_IS_JETSON = os.path.exists('/etc/nv_tegra_release')

if _IS_JETSON:
    FRAME_W, FRAME_H = 400, 225  # Smaller resolution for Jetson (less memory, faster encoding)
    YOLO_INPUT_W, YOLO_INPUT_H = 256, 144  # Very small input for YOLO on Jetson
    ENCODE_QUALITY = 50  # Lower quality = faster encoding on Jetson
    AI_FRAME_SKIP = 4  # Process every 4th frame on Jetson (~7.5 FPS AI)
else:
    FRAME_W, FRAME_H = 480, 270  # Higher resolution for PC
    YOLO_INPUT_W, YOLO_INPUT_H = 320, 180  # Standard input for YOLO
    ENCODE_QUALITY = 60  # Better quality for PC
    AI_FRAME_SKIP = 3  # Process every 3rd frame on PC (~10 FPS AI)

VEHICLE_CLASSES = [0, 1, 2, 3, 5, 7]  # car, truck, bus, motorbike
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
    - Single camera mode: Only opens one camera for less Jetson load
    
    Result: Smooth dual camera performance on Windows CPU!
    """
    
    def __init__(self, left_cam_id=0, right_cam_id=1, camera_mode='left'):
        """
        Initialize the DualCameraManager.
        
        Args:
            left_cam_id: Camera ID for left blind spot
            right_cam_id: Camera ID for right blind spot
            camera_mode: 'left', 'right', or 'both' - which camera(s) to open
        """
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.camera_mode = camera_mode  # 'left', 'right', or 'both'
        
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
        self._stop_event = threading.Event()  # More reliable stop signal
        
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
        """Open cameras based on camera_mode: 'left', 'right', or 'both'"""
        # Check if required cameras are already open
        if self.camera_mode == 'left':
            if self.left_cap is not None and self.left_cap.isOpened():
                return True
        elif self.camera_mode == 'right':
            if self.right_cap is not None and self.right_cap.isOpened():
                return True
        elif self.camera_mode == 'both':
            if self.left_cap is not None and self.right_cap is not None:
                if self.left_cap.isOpened() and self.right_cap.isOpened():
                    return True
        
        # Detect Jetson
        self.is_jetson = self._detect_jetson_nano()
        
        start_time = time.time()
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        
        print(f"📹 Opening camera(s) in '{self.camera_mode}' mode...")
        
        # Single camera mode - only open one camera (less Jetson load)
        if self.camera_mode == 'left':
            print(f"   Opening LEFT camera only (ID: {self.left_cam_id})...")
            if self.is_jetson:
                pipelines = self._get_jetson_camera_pipeline(self.left_cam_id)
                for pipeline, backend_type in pipelines:
                    try:
                        if backend_type is None:
                            self.left_cap = cv2.VideoCapture(pipeline)
                        else:
                            self.left_cap = cv2.VideoCapture(pipeline, backend_type)
                        if self.left_cap.isOpened():
                            ret, frame = self.left_cap.read()
                            if ret and frame is not None:
                                print(f"   ✓ Left camera opened successfully")
                                break
                        self.left_cap.release()
                    except:
                        pass
            else:
                self.left_cap = cv2.VideoCapture(self.left_cam_id, backend)
            
            self.right_cap = None  # No right camera in left-only mode
            self.dual_camera_mode = False
            
        elif self.camera_mode == 'right':
            print(f"   Opening RIGHT camera only (ID: {self.right_cam_id})...")
            if self.is_jetson:
                pipelines = self._get_jetson_camera_pipeline(self.right_cam_id)
                for pipeline, backend_type in pipelines:
                    try:
                        if backend_type is None:
                            self.right_cap = cv2.VideoCapture(pipeline)
                        else:
                            self.right_cap = cv2.VideoCapture(pipeline, backend_type)
                        if self.right_cap.isOpened():
                            ret, frame = self.right_cap.read()
                            if ret and frame is not None:
                                print(f"   ✓ Right camera opened successfully")
                                break
                        self.right_cap.release()
                    except:
                        pass
            else:
                self.right_cap = cv2.VideoCapture(self.right_cam_id, backend)
            
            self.left_cap = None  # No left camera in right-only mode
            self.dual_camera_mode = False
            
        else:  # 'both' mode
            print(f"   Opening BOTH cameras (Left: {self.left_cam_id}, Right: {self.right_cam_id})...")
            if self.is_jetson:
                # Open left camera
                pipelines = self._get_jetson_camera_pipeline(self.left_cam_id)
                for pipeline, backend_type in pipelines:
                    try:
                        if backend_type is None:
                            self.left_cap = cv2.VideoCapture(pipeline)
                        else:
                            self.left_cap = cv2.VideoCapture(pipeline, backend_type)
                        if self.left_cap.isOpened():
                            ret, frame = self.left_cap.read()
                            if ret and frame is not None:
                                print(f"   ✓ Left camera opened")
                                break
                        self.left_cap.release()
                    except:
                        pass
                
                # Open right camera
                pipelines = self._get_jetson_camera_pipeline(self.right_cam_id)
                for pipeline, backend_type in pipelines:
                    try:
                        if backend_type is None:
                            self.right_cap = cv2.VideoCapture(pipeline)
                        else:
                            self.right_cap = cv2.VideoCapture(pipeline, backend_type)
                        if self.right_cap.isOpened():
                            ret, frame = self.right_cap.read()
                            if ret and frame is not None:
                                print(f"   ✓ Right camera opened")
                                break
                        self.right_cap.release()
                    except:
                        pass
            else:
                self.left_cap = cv2.VideoCapture(self.left_cam_id, backend)
                self.right_cap = cv2.VideoCapture(self.right_cam_id, backend)
            
            self.dual_camera_mode = True
        
        # Configure opened cameras with ANTI-LAG optimizations
        for cap in [self.left_cap, self.right_cap]:
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                cap.set(cv2.CAP_PROP_FPS, 30)
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except:
                    pass
        
        # Verify required cameras are working
        if self.camera_mode == 'left':
            if not self.left_cap or not self.left_cap.isOpened():
                print("❌ Failed to open left camera")
                return False
        elif self.camera_mode == 'right':
            if not self.right_cap or not self.right_cap.isOpened():
                print("❌ Failed to open right camera")
                return False
        else:  # both
            if not self.left_cap or not self.left_cap.isOpened() or not self.right_cap or not self.right_cap.isOpened():
                print("❌ Failed to open both cameras")
                return False
        
        open_time = time.time() - start_time
        print(f"✅ Camera(s) opened in {open_time:.2f}s (mode: {self.camera_mode})")
        return True
    
    def start(self):
        """Start blind spot detection with single or dual camera mode"""
        if self.active:
            return True
        
        # NOTE: Camera lock is acquired by the controller, NOT here
        # This prevents double-lock acquisition issues
        
        # Clear stop event before starting new threads
        self._stop_event.clear()
        
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
        
        # Start threads based on camera_mode (less threads = less Jetson load)
        if self.camera_mode == 'left':
            # Only start left camera threads
            self.left_video_thread = threading.Thread(target=self._left_video_loop, daemon=True)
            self.left_ai_thread = threading.Thread(target=self._left_ai_loop, daemon=True)
            self.left_video_thread.start()
            self.left_ai_thread.start()
            print(f"✓ Blind spot detection started - LEFT camera only (2 threads)")
            
        elif self.camera_mode == 'right':
            # Only start right camera threads
            self.right_video_thread = threading.Thread(target=self._right_video_loop, daemon=True)
            self.right_ai_thread = threading.Thread(target=self._right_ai_loop, daemon=True)
            self.right_video_thread.start()
            self.right_ai_thread.start()
            print(f"✓ Blind spot detection started - RIGHT camera only (2 threads)")
            
        else:  # 'both' mode
            # Start all 4 threads for dual camera
            self.left_video_thread = threading.Thread(target=self._left_video_loop, daemon=True)
            self.left_ai_thread = threading.Thread(target=self._left_ai_loop, daemon=True)
            self.right_video_thread = threading.Thread(target=self._right_video_loop, daemon=True)
            self.right_ai_thread = threading.Thread(target=self._right_ai_loop, daemon=True)
            self.left_video_thread.start()
            self.left_ai_thread.start()
            self.right_video_thread.start()
            self.right_ai_thread.start()
            print(f"✓ Blind spot detection started - BOTH cameras (4 threads)")
        
        return True
    
    def _left_video_loop(self):
        """Left camera video thread - SMOOTH LAG-FREE FOR JETSON"""
        frame_counter = 0
        last_gc = time.time()
        
        while self.running and not self._stop_event.is_set():
            if not self.left_cap or not self.left_cap.isOpened():
                # Use event wait for faster response to stop signal
                if self._stop_event.wait(timeout=0.01):
                    break
                continue
            
            # Quick check for stop signal
            if self._stop_event.is_set():
                break
            
            # Periodic garbage collection
            current_time = time.time()
            if current_time - last_gc > 10.0:
                gc.collect()
                last_gc = current_time
                print(f"🗑️ Blindspot Left: GC performed (frame {frame_counter})")
            
            # SMOOTH APPROACH: Always grab twice - first discards buffered, second is fresh
            # This prevents lag WITHOUT causing stutter
            self.left_cap.grab()  # Discard potentially stale frame
            
            # Now grab and retrieve the fresh frame
            if not self.left_cap.grab():
                if self._stop_event.wait(timeout=0.001):
                    break
                continue
            
            ret, frame = self.left_cap.retrieve()
            if not ret or frame is None:
                if self._stop_event.wait(timeout=0.001):
                    break
                continue
            
            frame_counter += 1
            
            # Prevent counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Store frame for AI thread (minimal lock time)
            with self.left_lock:
                if self.left_latest_frame is not None:
                    del self.left_latest_frame
                self.left_latest_frame = frame.copy()
                danger = self.left_danger
            
            # Quick check for stop signal before heavy operations
            if self._stop_event.is_set():
                del frame
                break
            
            # Draw grid overlay
            color = (0, 0, 255) if danger else (0, 255, 0)
            self.detector.draw_side_mirror_grid(frame, color, is_left=True)
            
            # Use module-level encoding settings for Jetson optimization
            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            ret_encode, buffer = cv2.imencode('.jpg', frame_resized, 
                                              [cv2.IMWRITE_JPEG_QUALITY, ENCODE_QUALITY,
                                               cv2.IMWRITE_JPEG_OPTIMIZE, 1])  # Fast encoding
            
            del frame, frame_resized  # Clean up immediately
            
            if ret_encode:
                frame_bytes = buffer.tobytes()
                del buffer  # Free buffer immediately
                
                with self.left_lock:
                    if self.left_encoded is not None:
                        del self.left_encoded
                    self.left_encoded = frame_bytes
        
        print("🛑 Left video thread exiting")
    
    def _right_video_loop(self):
        """Right camera video thread - SMOOTH LAG-FREE FOR JETSON"""
        frame_counter = 0
        last_gc = time.time()
        
        while self.running and not self._stop_event.is_set():
            if not self.right_cap or not self.right_cap.isOpened():
                # Use event wait for faster response to stop signal
                if self._stop_event.wait(timeout=0.01):
                    break
                continue
            
            # Quick check for stop signal
            if self._stop_event.is_set():
                break
            
            # Periodic garbage collection
            current_time = time.time()
            if current_time - last_gc > 10.0:
                gc.collect()
                last_gc = current_time
                print(f"🗑️ Blindspot Right: GC performed (frame {frame_counter})")
            
            # SMOOTH APPROACH: Always grab twice - first discards buffered, second is fresh
            # This prevents lag WITHOUT causing stutter
            self.right_cap.grab()  # Discard potentially stale frame
            
            # Now grab and retrieve the fresh frame
            if not self.right_cap.grab():
                if self._stop_event.wait(timeout=0.001):
                    break
                continue
            
            ret, frame = self.right_cap.retrieve()
            if not ret or frame is None:
                if self._stop_event.wait(timeout=0.001):
                    break
                continue
            
            frame_counter += 1
            
            # Prevent counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Store frame for AI thread (minimal lock time)
            with self.right_lock:
                if self.right_latest_frame is not None:
                    del self.right_latest_frame
                self.right_latest_frame = frame.copy()
                danger = self.right_danger
            
            # Quick check for stop signal before heavy operations
            if self._stop_event.is_set():
                del frame
                break
            
            # Draw grid overlay
            color = (0, 0, 255) if danger else (0, 255, 0)
            self.detector.draw_side_mirror_grid(frame, color, is_left=False)
            
            # Use module-level encoding settings for Jetson optimization
            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
            ret_encode, buffer = cv2.imencode('.jpg', frame_resized, 
                                              [cv2.IMWRITE_JPEG_QUALITY, ENCODE_QUALITY,
                                               cv2.IMWRITE_JPEG_OPTIMIZE, 1])  # Fast encoding
            
            del frame, frame_resized  # Clean up immediately
            
            if ret_encode:
                frame_bytes = buffer.tobytes()
                del buffer  # Free buffer immediately
                
                with self.right_lock:
                    if self.right_encoded is not None:
                        del self.right_encoded
                    self.right_encoded = frame_bytes
        
        print("🛑 Right video thread exiting")
    
    def _left_ai_loop(self):
        """Left AI thread - JETSON OPTIMIZED"""
        frame_counter = 0
        last_gc = time.time()
        
        # Use module-level frame skip setting
        skip_count = AI_FRAME_SKIP
        
        while self.running and not self._stop_event.is_set():
            # Quick exit check at start of each iteration
            if self._stop_event.is_set():
                break
                
            # Frame skipping (Jetson: every 4th, PC: every 3rd)
            frame_counter += 1
            if frame_counter % skip_count != 0:
                # Use event wait instead of sleep for faster response
                if self._stop_event.wait(timeout=0.015):
                    break
                continue
            
            # Check again after wait
            if self._stop_event.is_set():
                break
            
            # Periodic garbage collection to prevent memory accumulation (more frequent on Jetson)
            current_time = time.time()
            gc_interval = 30 if self.is_jetson else 60  # Every 30 seconds on Jetson
            if current_time - last_gc > gc_interval:
                gc.collect()
                if self.is_jetson and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_gc = current_time
                print("🗑️ Left AI: Garbage collection performed")
            
            # Prevent frame counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Get latest frame
            with self.left_lock:
                if self.left_latest_frame is None:
                    pass  # Will sleep below
                else:
                    # Process smaller frame for maximum speed (320x180 for CPU)
                    frame_small = cv2.resize(self.left_latest_frame, (YOLO_INPUT_W, YOLO_INPUT_H), 
                                           interpolation=cv2.INTER_LINEAR)
            
            if self.left_latest_frame is None:
                if self._stop_event.wait(timeout=0.01):
                    break
                continue
            
            try:
                # Quick exit check before heavy AI operation
                if self._stop_event.is_set():
                    del frame_small
                    break
                    
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
                if self._stop_event.wait(timeout=0.1):
                    break
        
        print("🛑 Left AI thread exiting")
    
    def _right_ai_loop(self):
        """Right AI thread - JETSON OPTIMIZED"""
        frame_counter = 0
        last_gc = time.time()
        
        # Use module-level frame skip setting
        skip_count = AI_FRAME_SKIP
        
        while self.running and not self._stop_event.is_set():
            # Quick exit check at start of each iteration
            if self._stop_event.is_set():
                break
                
            # Frame skipping (Jetson: every 4th, PC: every 3rd)
            frame_counter += 1
            if frame_counter % skip_count != 0:
                # Use event wait instead of sleep for faster response
                if self._stop_event.wait(timeout=0.015):
                    break
                continue
            
            # Check again after wait
            if self._stop_event.is_set():
                break
            
            # Periodic garbage collection to prevent memory accumulation (more frequent on Jetson)
            current_time = time.time()
            gc_interval = 30 if self.is_jetson else 60  # Every 30 seconds on Jetson
            if current_time - last_gc > gc_interval:
                gc.collect()
                if self.is_jetson and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                last_gc = current_time
                print("🗑️ Right AI: Garbage collection performed")
            
            # Prevent frame counter overflow
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Get latest frame
            with self.right_lock:
                if self.right_latest_frame is None:
                    pass  # Will sleep below
                else:
                    # Process smaller frame for maximum speed (320x180 for CPU)
                    frame_small = cv2.resize(self.right_latest_frame, (YOLO_INPUT_W, YOLO_INPUT_H), 
                                           interpolation=cv2.INTER_LINEAR)
            
            if self.right_latest_frame is None:
                if self._stop_event.wait(timeout=0.01):
                    break
                continue
            
            try:
                # Quick exit check before heavy AI operation
                if self._stop_event.is_set():
                    del frame_small
                    break
                    
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
                if self._stop_event.wait(timeout=0.1):
                    break
        
        print("🛑 Right AI thread exiting")
    
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
        """Stop dual camera blind spot detection - ROBUST cleanup for Jetson"""
        print("🛑 Stopping blind spot detection...")
        
        # Signal threads to stop FIRST using both mechanisms for reliability
        self.running = False
        self.active = False
        self._stop_event.set()  # Signal threads via event (faster response)
        
        # Import camera manager for proper cleanup
        from app.services.camera_manager import force_release_camera, release_camera_lock
        
        # Wait for threads to notice the stop signal and exit gracefully
        # Longer timeout on Jetson due to potential blocking in camera reads
        thread_timeout = 2.0 if self.is_jetson else 1.0
        
        threads = [
            ('left_video', self.left_video_thread),
            ('left_ai', self.left_ai_thread),
            ('right_video', self.right_video_thread),
            ('right_ai', self.right_ai_thread)
        ]
        
        # First pass: try to join all threads gracefully
        for name, thread in threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=thread_timeout)
                if thread.is_alive():
                    print(f"⚠️ Thread {name} did not stop after {thread_timeout}s, continuing cleanup...")
        
        # Second pass: check again after a small delay (threads may still be exiting)
        time.sleep(0.1)
        for name, thread in threads:
            if thread is not None and thread.is_alive():
                # Final attempt with shorter timeout
                thread.join(timeout=0.5)
                if thread.is_alive():
                    print(f"⚠️ Thread {name} still running - will be abandoned (daemon thread)")
        
        # Clear threads references
        self.left_video_thread = None
        self.left_ai_thread = None
        self.right_video_thread = None
        self.right_ai_thread = None
        
        # Release cameras with proper Jetson cleanup
        # IMPORTANT: Check if cameras are same object BEFORE releasing
        same_camera = (self.right_cap is not None and self.left_cap is not None 
                       and self.right_cap is self.left_cap)
        
        if self.left_cap:
            force_release_camera(self.left_cap, "blindspot-left")
            self.left_cap = None
        
        if self.right_cap and not same_camera:
            # Only release right if it's a different camera
            force_release_camera(self.right_cap, "blindspot-right")
        self.right_cap = None
        
        # Release camera lock
        release_camera_lock("blindspot")
        
        # Clear buffers
        with self.left_lock:
            self.left_latest_frame = None
            self.left_encoded = None
            self.left_danger = False
        with self.right_lock:
            self.right_latest_frame = None
            self.right_encoded = None
            self.right_danger = False
        
        # Force garbage collection on Jetson
        gc.collect()
        if self.is_jetson and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Additional delay on Jetson for camera driver to fully release
        if self.is_jetson:
            time.sleep(0.3)  # Increased delay for dual cameras
        
        print("✓ Blind spot detection stopped")
    
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
