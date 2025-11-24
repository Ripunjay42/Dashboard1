import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import threading
import time
import platform
from queue import Queue

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True

# Global model instance for reuse
_global_detector = None
_detector_lock = threading.Lock()


class PotholeDetector:
    
    def __init__(self, model_path, device=None):
        print(f" Loading model (ULTRA FAST MODE)...")
        start_time = time.time()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = (128, 128)  # ULTRA FAST: 256→128 = 4x fewer pixels = 3-4x faster
        self.threshold = 0.45  # Slightly lower for faster detection trigger

        # Build model (same architecture as training)
        self.model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        ).to(self.device)

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Preprocessing pipeline - OPTIMIZED
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s on {self.device}")

    def preprocess(self, frame):
        """Optimized preprocessing"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def postprocess(self, output, original_size):
        """Optimized postprocessing with better mask generation"""
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (prob_mask > self.threshold).astype(np.uint8)
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        return mask_resized * 255

    def detect(self, frame):
        """Run inference on a single frame with CUDA acceleration if available"""
        original_size = frame.shape[:2]
        tensor = self.preprocess(frame)
        with torch.no_grad():
            # Use mixed precision for better performance on CUDA
            if self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.model(tensor)
            else:
                output = self.model(tensor)
        return self.postprocess(output, original_size)


def get_global_detector(model_path):
    """Get or create the global detector instance (singleton pattern)"""
    global _global_detector
    with _detector_lock:
        if _global_detector is None:
            _global_detector = PotholeDetector(model_path)
    return _global_detector


def get_camera_pipeline(camera_id=0):
    """
    Get optimized camera pipeline based on platform
    Returns: [(pipeline, backend), ...]
    """
    system = platform.system()
    
    # Detect if running on Jetson Nano
    is_jetson = False
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            is_jetson = True
    except:
        pass
    
    if is_jetson:
        print("Jetson Nano detected - Using GStreamer pipeline")
        
        # Try CSI camera first (Raspberry Pi Camera)
        gst_csi = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1, format=NV12 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=480, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
        
        # USB camera fallback
        gst_usb = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
        
        return [(gst_csi, cv2.CAP_GSTREAMER), (gst_usb, cv2.CAP_GSTREAMER)]
    
    elif system == "Linux":
        print("Linux detected - Using V4L2 GStreamer")
        gst = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=1"
        )
        return [(gst, cv2.CAP_GSTREAMER), (camera_id, cv2.CAP_V4L2)]
    
    else:
        print("🪟 Windows detected - Using DirectShow")
        # Windows - use DirectShow (default) with optimizations
        return [(camera_id, cv2.CAP_DSHOW), (camera_id, cv2.CAP_ANY)]


class VideoStreamManager:
    """
    HIGH PERFORMANCE ARCHITECTURE: Two Parallel Threads
    
    Thread 1 (Video): Camera → Encode → Stream (FAST, no AI)
    Thread 2 (AI): Copy frame → Detect → Update status (SLOW, independent)
    
    Result: Smooth 30 FPS video + accurate detection, NO LAG!
    """
    
    def __init__(self, model_path, camera_id=0):
        # Use global detector instance
        self.detector = get_global_detector(model_path)
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Shared buffers (thread-safe)
        self.latest_frame = None  # Raw frame from camera (for AI thread)
        self.encoded_frame = None  # JPEG frame (for streaming)
        self.pothole_detected = False  # Detection result (from AI thread)
        self.current_mask = None  # Detection mask (from AI thread)
        
        # Detection smoothing (INSTANT MODE - NO DELAY)
        self.detection_history = []
        self.detection_history_size = 3  # Absolute minimum: 3 frames @ 30fps = ~100ms
        self.min_detections_to_start = 2  # 2/3 = 67% to START (instant trigger)
        self.min_detections_to_continue = 1  # 1/3 = 33% to CONTINUE (ultra responsive)
        self.is_currently_detecting = False
        self.frames_since_detection = 0
        self.detection_persistence = 5  # Instant clear (5 frames = ~167ms)
        
        # Performance settings (ULTRA FAST)
        self.jpeg_quality = 55  # Lower quality = faster encoding (was 65)
    
    def _open_camera(self):
        """Open camera with platform-specific optimizations and better error handling"""
        if self.cap is not None and self.cap.isOpened():
            return True
        
        print(f"📹 Opening camera {self.camera_id}...")
        start_time = time.time()
        
        # Try multiple pipelines based on platform
        pipelines = get_camera_pipeline(self.camera_id)
        
        for pipeline, backend in pipelines:
            try:
                print(f"   Trying: {pipeline if isinstance(pipeline, str) else f'Camera {pipeline}'}")
                self.cap = cv2.VideoCapture(pipeline, backend)
                
                if not self.cap.isOpened():
                    print(f"    Failed to open")
                    continue
                
                # Set properties BEFORE testing read
                if backend != cv2.CAP_GSTREAMER:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # No buffering for low latency
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG for fast capture
                    # Add small delay for Windows MSMF to stabilize
                    time.sleep(0.5)
                
                # Test read with retries
                success = False
                for attempt in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        success = True
                        print(f"    Success on attempt {attempt + 1}")
                        break
                    time.sleep(0.2)
                
                if success:
                    open_time = time.time() - start_time
                    print(f" Camera opened in {open_time:.2f}s\n")
                    return True
                else:
                    print(f"    Could not read frames")
                    self.cap.release()
                    
            except Exception as e:
                print(f"    Failed: {e}")
                if self.cap:
                    self.cap.release()
                continue
        
        print("Failed to open camera with any method")
        print("    Tip: Close other apps using the camera (Zoom, Teams, etc.)")
        self.cap = None
        return False
    
    def start(self):
        """Start both parallel threads: Video + AI"""
        if self.is_running:
            return True
        
        # Open camera if not already open
        if self.cap is None or not self.cap.isOpened():
            if not self._open_camera():
                return False
        else:
            print("Camera already initialized, starting threads...")
        
        self.is_running = True
        
        # Thread 1: FAST video streaming (no AI)
        video_thread = threading.Thread(target=self._video_stream_loop, daemon=True)
        video_thread.start()
        
        # Thread 2: SLOW AI inference (independent)
        ai_thread = threading.Thread(target=self._ai_inference_loop, daemon=True)
        ai_thread.start()
        
        print(" Two parallel threads started:")
        print("  - Thread 1: Video streaming (30 FPS)")
        print("  - Thread 2: AI inference (REAL-TIME, adaptive)")
        
        return True
    
    def _video_stream_loop(self):
        """
        THREAD 1: FAST Video Streaming
        Applies latest detection overlay and streams at 30 FPS
        """
        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                break
            
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            # Save raw frame for AI thread + get latest mask (single lock)
            with self.lock:
                self.latest_frame = frame.copy()
                current_mask = self.current_mask
            
            # Apply overlay immediately if we have detection
            if current_mask is not None:
                color_mask = np.zeros_like(frame)
                color_mask[current_mask > 0] = [0, 0, 255]  # Red
                overlay_frame = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
            else:
                overlay_frame = frame
            
            # Fast JPEG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ret, buffer = cv2.imencode('.jpg', overlay_frame, encode_param)
            
            if ret:
                with self.lock:
                    self.encoded_frame = buffer.tobytes()
    
    def _ai_inference_loop(self):
        """
        THREAD 2: INSTANT AI Inference
        Optimized for absolute minimum latency
        """
        while self.is_running:
            # Get latest frame copy (non-blocking)
            with self.lock:
                if self.latest_frame is None:
                    time.sleep(0.005)
                    continue
                # OPTIMIZATION: Resize before copy to reduce memory transfer
                frame_small = cv2.resize(self.latest_frame, (320, 240), interpolation=cv2.INTER_LINEAR)
            
            # Run AI inference immediately (smaller frame = faster inference!)
            mask = self.detector.detect(frame_small)
            
            # Resize mask back to original size for overlay
            mask_full = cv2.resize(mask, (self.latest_frame.shape[1], self.latest_frame.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            # Resize mask back to original size for overlay
            mask_full = cv2.resize(mask, (self.latest_frame.shape[1], self.latest_frame.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            
            # Check if pothole detected
            pothole_pixels = np.sum(mask_full > 0)
            total_pixels = mask_full.shape[0] * mask_full.shape[1]
            is_detected = (pothole_pixels / total_pixels) > 0.005
            
            # Smoothing with history
            self.detection_history.append(is_detected)
            if len(self.detection_history) > self.detection_history_size:
                self.detection_history.pop(0)
            
            detections_count = sum(self.detection_history)
            
            # Hysteresis logic for stable detection
            if not self.is_currently_detecting:
                if detections_count >= self.min_detections_to_start:
                    self.is_currently_detecting = True
                    self.frames_since_detection = 0
                    with self.lock:
                        self.current_mask = mask_full
                        self.pothole_detected = True
            else:
                if detections_count >= self.min_detections_to_continue:
                    self.frames_since_detection = 0
                    with self.lock:
                        self.current_mask = mask_full
                        self.pothole_detected = True
                else:
                    self.frames_since_detection += 1
                    if self.frames_since_detection > self.detection_persistence:
                        self.is_currently_detecting = False
                        with self.lock:
                            self.current_mask = None
                            self.pothole_detected = False
            
            # NO artificial sleep - run as fast as model allows!
    
    def get_encoded_frame(self):
        """Get pre-encoded JPEG frame for MJPEG streaming"""
        with self.lock:
            if self.encoded_frame is None:
                return None
            return self.encoded_frame
    
    def is_pothole_detected(self):
        """Check if pothole is currently detected"""
        with self.lock:
            return self.pothole_detected
    
    def stop(self):
        """Stop both video and AI threads"""
        self.is_running = False
        time.sleep(0.3)  # Give threads time to exit
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.latest_frame = None
            self.encoded_frame = None
            self.current_mask = None
            self.pothole_detected = False
        
        print(" Video and AI threads stopped")
    
    def is_active(self):
        """Check if stream is active"""
        return self.is_running and self.cap is not None and self.cap.isOpened()
