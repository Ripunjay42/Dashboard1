import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
from queue import Queue
import platform

# Enable CUDA optimizations
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
        print(f" Loading Blind Spot Detection models (JETSON NANO OPTIMIZED)...")
        start_time = time.time()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Detect if running on Jetson Nano
        self.is_jetson = self._detect_jetson()
        
        # Load YOLO with optimizations for Jetson Nano
        self.yolo = YOLO("yolov8n.pt")  # Nano model - fastest
        
        if self.is_jetson:
            # Jetson Nano optimizations
            print("    Jetson Nano detected - Ultra optimized settings")
            self.yolo_img_size = 320  # Good balance for Jetson
            self.conf_threshold = 0.3  # Lower confidence for faster NMS
            self.iou_threshold = 0.7   # Higher IoU for faster NMS
            self.frame_skip = 4        # Process every 4th frame (7.5 FPS AI)
        else:
            # Windows CPU optimizations - AGGRESSIVE FOR SMOOTH PERFORMANCE
            print("    Windows CPU detected - Maximum optimization settings")
            self.yolo_img_size = YOLO_INPUT_W  # Use 320px for CPU (good balance)
            self.conf_threshold = 0.3  # Lower confidence for faster detection
            self.iou_threshold = 0.7   # Higher IoU for faster NMS
            self.frame_skip = 3        # Process every 3rd frame (10 FPS AI)
            
            # Fuse model for CPU optimization
            try:
                self.yolo.fuse()  # Optimize model for inference speed
                print("   YOLO model fused for CPU optimization")
            except:
                print("   Model fusion not available")
        
        # SKIP MiDaS for performance - use YOLO-only detection
        self.use_depth = False
        
        load_time = time.time() - start_time
        print(f"Models loaded in {load_time:.2f}s on {self.device}")
        if self.is_jetson:
            print(f"   Jetson Nano ready: {self.yolo_img_size}px input, conf={self.conf_threshold}")
        else:
            print(f"   💻 PC ready: {self.yolo_img_size}px input, conf={self.conf_threshold}")
    
    def _detect_jetson(self):
        """Detect if running on Jetson Nano"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                return 'jetson' in model or 'tegra' in model
        except:
            # Alternative detection methods
            try:
                import subprocess
                result = subprocess.check_output(['cat', '/etc/nv_tegra_release'], 
                                               stderr=subprocess.DEVNULL, text=True)
                return 'tegra' in result.lower()
            except:
                return False
    
    def detect_vehicles_only(self, frame):
        """Optimized YOLO inference - MEMORY LEAK PREVENTION"""
        # YOLO inference with optimized parameters
        with torch.no_grad():  # Prevent gradient accumulation
            if self.is_jetson:
                # Jetson Nano: Use half precision for speed
                results = self.yolo.predict(
                    frame, 
                    imgsz=self.yolo_img_size,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    half=True,  # FP16 for Jetson
                    device=self.device,
                    save=False,  # Don't save results to disk
                    stream=True  # Stream results for memory efficiency
                )
                results = next(results)  # Get first result from stream
            else:
                # Windows CPU: Optimized for maximum speed
                results = self.yolo.predict(
                    frame,
                    imgsz=self.yolo_img_size, 
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    half=False,  # CPU doesn't benefit from FP16
                    device='cpu',  # Force CPU for consistency
                    classes=VEHICLE_CLASSES,  # Only detect vehicles (faster)
                    save=False,  # Don't save results to disk
                    stream=True  # Stream results for memory efficiency
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
        
        # Explicit cleanup
        del results
        
        return detections
    
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
        """Get optimized GStreamer pipeline for Jetson Nano"""
        # CSI Camera (Raspberry Pi Camera Module)
        csi_pipeline = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            f"video/x-raw(memory:NVMM), width={FRAME_W}, height={FRAME_H}, framerate=30/1, format=NV12 ! "
            "nvvidconv flip-method=0 ! "
            f"video/x-raw, width={FRAME_W}, height={FRAME_H}, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
        
        # USB Camera fallback
        usb_pipeline = (
            f"v4l2src device=/dev/video{camera_id} ! "
            f"video/x-raw, width={FRAME_W}, height={FRAME_H}, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
        
        return [(csi_pipeline, cv2.CAP_GSTREAMER), (usb_pipeline, cv2.CAP_GSTREAMER)]
    
    def _detect_available_cameras(self):
        """Detect available cameras - Windows CPU optimized (single camera preferred)"""
        available_cameras = []
        
        print("🔍 Detecting available cameras for single-camera mode...")
        
        # Check if running on Jetson Nano
        self.is_jetson = self._detect_jetson_nano()
        
        if self.is_jetson:
            print("    Jetson Nano detected - Testing GStreamer pipelines")
            
            # Test CSI and USB cameras on Jetson
            for camera_id in range(2):
                pipelines = self._get_jetson_camera_pipeline(camera_id)
                
                for pipeline, backend in pipelines:
                    try:
                        cap = cv2.VideoCapture(pipeline, backend)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                available_cameras.append(camera_id)
                                print(f"   Jetson Camera {camera_id} available (GStreamer)")
                                cap.release()
                                break
                        cap.release()
                    except Exception as e:
                        print(f"   Jetson Camera {camera_id} failed: {e}")
        else:
            # Windows CPU detection - prefer single high-quality camera
            print("   💻 Windows CPU mode - Looking for best single camera")
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
            
            for camera_id in range(4):
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(camera_id, backend)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                available_cameras.append(camera_id)
                                print(f"   Camera {camera_id} available (will be used for both sides)")
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
            self.left_cap = cv2.VideoCapture(available_cameras[0], backend)
            self.right_cap = cv2.VideoCapture(available_cameras[0], backend)
            self.dual_camera_mode = False
            print("   Single camera mode: Same feed for both sides")
        
        # Configure both cameras with ANTI-LAG optimizations
        for cap in [self.left_cap, self.right_cap]:
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CRITICAL: No buffering = no lag accumulation
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)  # 480x270 for good quality
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
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
        """Left camera video thread - MEMORY LEAK PREVENTION"""
        frame_counter = 0
        last_cleanup = time.time()
        
        while self.running:
            if not self.left_cap or not self.left_cap.isOpened():
                time.sleep(0.01)
                continue
            
            # Clear camera buffer periodically to prevent lag accumulation
            current_time = time.time()
            if current_time - last_cleanup > 30:  # Every 30 seconds
                # Flush camera buffer
                for _ in range(3):
                    self.left_cap.read()
                last_cleanup = current_time
                print("Left camera buffer flushed")
            
            ret, frame = self.left_cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            # Frame skipping for Windows CPU (every other frame = 15 FPS)
            frame_counter += 1
            if frame_counter % 2 != 0:
                # Explicitly delete frame to free memory immediately
                del frame
                continue
            
            # Prevent frame counter overflow (memory optimization)
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Store frame for AI thread + get current danger status (minimal lock time)
            with self.left_lock:
                # Delete old frame before storing new one
                if self.left_latest_frame is not None:
                    del self.left_latest_frame
                self.left_latest_frame = frame.copy()
                danger = self.left_danger
            
            # Draw grid overlay based on current danger status
            color = (0, 0, 255) if danger else (0, 255, 0)
            self.detector.draw_side_mirror_grid(frame, color, is_left=True)
            
            # Windows CPU optimized encoding
            encode_size = (400, 225) if not self.is_jetson else (480, 270)
            encode_quality = 50 if not self.is_jetson else 70
            
            frame_resized = cv2.resize(frame, encode_size, interpolation=cv2.INTER_LINEAR)
            ret_encode, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
            
            # Explicit memory cleanup
            del frame_resized
            
            if ret_encode:
                with self.left_lock:
                    # Delete old encoded frame before storing new one
                    if self.left_encoded is not None:
                        del self.left_encoded
                    self.left_encoded = buffer.tobytes()
                # Delete buffer immediately after use
                del buffer
            
            # Performance monitoring (every 1000 frames to avoid overhead)
            if frame_counter % 1000 == 0:
                self._monitor_performance()
            
            # Delete original frame
            del frame
    
    def _right_video_loop(self):
        """Right camera video thread - MEMORY LEAK PREVENTION"""
        frame_counter = 0
        last_cleanup = time.time()
        
        while self.running:
            if not self.right_cap or not self.right_cap.isOpened():
                time.sleep(0.01)
                continue
            
            # Clear camera buffer periodically to prevent lag accumulation
            current_time = time.time()
            if current_time - last_cleanup > 30:  # Every 30 seconds
                # Flush camera buffer
                for _ in range(3):
                    self.right_cap.read()
                last_cleanup = current_time
                print("Right camera buffer flushed")
            
            ret, frame = self.right_cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            # Frame skipping for Windows CPU (every other frame = 15 FPS)
            frame_counter += 1
            if frame_counter % 2 != 0:
                # Explicitly delete frame to free memory immediately
                del frame
                continue
            
            # Prevent frame counter overflow (memory optimization)
            if frame_counter > 1000000:
                frame_counter = 0
            
            # Store frame for AI thread + get current danger status (minimal lock time)
            with self.right_lock:
                # Delete old frame before storing new one
                if self.right_latest_frame is not None:
                    del self.right_latest_frame
                self.right_latest_frame = frame.copy()
                danger = self.right_danger
            
            # Draw grid overlay based on current danger status
            color = (0, 0, 255) if danger else (0, 255, 0)
            self.detector.draw_side_mirror_grid(frame, color, is_left=False)
            
            # Windows CPU optimized encoding
            encode_size = (400, 225) if not self.is_jetson else (480, 270)
            encode_quality = 50 if not self.is_jetson else 70
            
            frame_resized = cv2.resize(frame, encode_size, interpolation=cv2.INTER_LINEAR)
            ret_encode, buffer = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, encode_quality])
            
            # Explicit memory cleanup
            del frame_resized
            
            if ret_encode:
                with self.right_lock:
                    # Delete old encoded frame before storing new one
                    if self.right_encoded is not None:
                        del self.right_encoded
                    self.right_encoded = buffer.tobytes()
                # Delete buffer immediately after use
                del buffer
            
            # Delete original frame
            del frame
    
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
