import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import threading
import time
import platform
import os
import gc
import warnings
from queue import Queue

# Suppress JPEG corruption warnings (common with USB cameras on Jetson)
warnings.filterwarnings('ignore', message='Corrupt JPEG data')

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

# Global model instance for reuse
_global_detector = None
_detector_lock = threading.Lock()


# ===============================================
# MODEL ARCHITECTURE (MATCHES YOUR TRAINING CODE)
# ===============================================
class DecoderBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", weights=None)

        self.encoder = torch.nn.ModuleDict({
            "conv1": resnet.conv1,
            "bn1":   resnet.bn1,
            "relu":  resnet.relu,
            "maxpool": resnet.maxpool,
            "layer1": resnet.layer1,
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        })

        self.center = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 1024, 3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
        )

        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.dec0 = DecoderBlock(64, 32)

        self.seg_head = torch.nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x0 = self.encoder["conv1"](x)
        x0 = self.encoder["bn1"](x0)
        x0 = self.encoder["relu"](x0)
        x1 = self.encoder["maxpool"](x0)
        x1 = self.encoder["layer1"](x1)
        x2 = self.encoder["layer2"](x1)
        x3 = self.encoder["layer3"](x2)
        x4 = self.encoder["layer4"](x3)

        c = self.center(x4)
        d4 = self.dec4(c)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        d0 = self.dec0(d1)

        return self.seg_head(d0)


class PotholeDetector:
    
    def __init__(self, model_path, device=None):
        print(f"Loading model (JETSON ORIN OPTIMIZED)...")
        start_time = time.time()
        
        # Detect Jetson device
        self.is_jetson = self._detect_jetson()
        
        # Set device with proper error handling and memory checks
        if device:
            self.device = device
        elif torch.cuda.is_available() and self.is_jetson:
            # Check if CUDA has enough memory before attempting to use it
            try:
                # Clear any cached memory first
                torch.cuda.empty_cache()
                
                # Try to allocate a small test tensor to verify CUDA works
                test_tensor = torch.zeros((1, 3, 64, 64), device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                
                self.device = 'cuda'
                print("✓ CUDA available on Jetson Orin Nano")
                
                # Set memory fraction to prevent OOM (use max 80% of GPU memory)
                torch.cuda.set_per_process_memory_fraction(0.8, 0)
            except Exception as e:
                print(f"⚠️ CUDA memory allocation failed: {e}")
                print("   Falling back to CPU mode")
                self.device = 'cpu'
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("✓ CUDA available")
        else:
            self.device = 'cpu'
            print("⚠️ Using CPU mode")
        
        # Input size MUST match training (512x512 for UNetResNet50)
        # Jetson-specific optimizations for inference resolution
        if self.is_jetson:
            if self.device == 'cuda':
                # CUDA MODE: Use smaller input for Jetson to avoid OOM
                self.input_size = (256, 256)  # Reduced for Jetson GPU memory
                self.threshold = 0.5
                print(f"   Jetson Orin detected - CUDA MODE: {self.input_size}")
            else:
                # CPU MODE: Even smaller for speed
                self.input_size = (128, 128)
                self.threshold = 0.5
                print(f"   Jetson Orin detected - CPU MODE: {self.input_size}")
        else:
            # Full resolution for powerful machines
            self.input_size = (512, 512)
            self.threshold = 0.5
            print(f"   Using full resolution: {self.input_size}")
        
        try:
            self.use_tensorrt = False  # TensorRT not used for this model
            
            print(f"   Building UNetResNet50 model...")
            
            # Aggressive garbage collection before building model
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Build model (same architecture as training)
            try:
                self.model = UNetResNet50()
                print(f"   ✓ Model architecture created")
            except Exception as e:
                print(f"❌ Failed to create model: {e}")
                raise
            
            # Load trained weights with error handling
            if os.path.exists(model_path):
                print(f"   Loading weights: {model_path}")
                
                # Load weights to CPU first to avoid CUDA allocation errors
                try:
                    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"   ✓ Weights loaded to CPU")
                    
                    # Clean up state_dict from memory
                    del state_dict
                    gc.collect()
                except Exception as e:
                    print(f"❌ Failed to load weights: {e}")
                    raise
            else:
                print(f"⚠️ Warning: Model file not found at {model_path}")
            
            # Move to device AFTER loading weights with proper error handling
            try:
                if self.device == 'cuda':
                    print(f"   Moving model to CUDA...")
                    
                    # Aggressive memory cleanup before CUDA transfer
                    gc.collect()
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # Wait for all operations to complete
                    
                    # Try to move model to CUDA
                    self.model = self.model.to(self.device)
                    print(f"   ✓ Model successfully moved to CUDA")
                    
                    # Clear cache again after model transfer
                    torch.cuda.empty_cache()
                else:
                    self.model = self.model.to(self.device)
                    
            except RuntimeError as e:
                print(f"❌ CUDA out of memory: {e}")
                print("   Falling back to CPU mode")
                
                # Force cleanup
                if hasattr(self, 'model'):
                    del self.model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Rebuild model on CPU
                self.device = 'cpu'
                self.model = UNetResNet50()
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                self.model.load_state_dict(state_dict, strict=False)
                del state_dict
                self.model = self.model.to('cpu')
                
                # Update input size for CPU mode
                if self.is_jetson:
                    self.input_size = (128, 128)
                    print(f"   Using CPU-optimized input size: {self.input_size}")
            
            self.model.eval()
            
            # Disable gradient computation globally for this model
            for param in self.model.parameters():
                param.requires_grad = False
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # Preprocessing pipeline - OPTIMIZED
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Warm-up inference (prevents first-frame crash on Jetson)
        if self.is_jetson and self.device == 'cuda':
            try:
                print("🔥 Warming up CUDA model...")
                
                # Small dummy frame for warm-up (reduces memory usage)
                dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                
                # Clear cache before warm-up
                torch.cuda.empty_cache()
                
                # Run warm-up inference
                _ = self.detect(dummy_frame)
                
                # Aggressive cleanup after warm-up
                gc.collect()
                torch.cuda.empty_cache()
                
                print("✓ CUDA model warmed up successfully")
            except Exception as e:
                print(f"⚠️ Warm-up failed: {e}")
                print("   This is normal, model will work during actual inference")
        elif self.is_jetson:
            print("⚠️ Skipping warm-up (CPU mode)")
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s on {self.device}")
    
    def _detect_jetson(self):
        """Detect if running on Jetson device"""
        try:
            # Check for Jetson Orin/Nano/Xavier
            if os.path.exists('/etc/nv_tegra_release'):
                return True
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                return 'jetson' in model or 'tegra' in model
        except:
            return False

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
        """Run inference on a single frame with CUDA acceleration - FAST VERSION"""
        try:
            original_size = frame.shape[:2]
            
            # ---- FAST PREPROCESSING (no PIL, direct numpy/torch) ----
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, self.input_size)
            
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            tensor = transforms.functional.normalize(
                tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)
            
            # ---- INFERENCE ----
            with torch.no_grad():
                if self.device == 'cuda':
                    # Use mixed precision for faster inference
                    with torch.cuda.amp.autocast():
                        output = self.model(tensor)
                else:
                    output = self.model(tensor)
            
            # ---- POSTPROCESS ----
            prob_mask = torch.sigmoid(output)[0, 0].cpu().numpy()
            mask = (prob_mask > self.threshold).astype(np.uint8)
            
            mask_resized = cv2.resize(mask, (original_size[1], original_size[0]),
                                      interpolation=cv2.INTER_NEAREST)
            
            # Cleanup for Jetson
            if self.is_jetson:
                del tensor, output
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            return mask_resized * 255
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)


def get_global_detector(model_path):
    """Get or create the global detector instance (singleton pattern)"""
    global _global_detector
    with _detector_lock:
        if _global_detector is None:
            _global_detector = PotholeDetector(model_path)
    return _global_detector


def unload_global_detector():
    """Unload the global detector instance to free memory"""
    global _global_detector
    with _detector_lock:
        if _global_detector is not None:
            print("🗑️ Unloading pothole detection model from memory...")
            # Clear model from memory
            del _global_detector
            _global_detector = None
            # Force garbage collection
            gc.collect()
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ Pothole model unloaded")


def get_camera_pipeline(camera_id=0):
    """
    Get optimized camera pipeline based on platform
    Returns: [(pipeline, backend), ...]
    """
    system = platform.system()
    
    # Detect if running on Jetson device
    is_jetson = False
    is_jetson_orin = False
    try:
        if os.path.exists('/etc/nv_tegra_release'):
            is_jetson = True
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                if 'orin' in content.lower():
                    is_jetson_orin = True
    except:
        pass
    
    if is_jetson:
        if is_jetson_orin:
            print("🚀 Jetson Orin Nano detected - Using optimized pipeline")
        else:
            print("📹 Jetson Nano detected - Using optimized pipeline")
        
        # Priority 1: V4L2 with MJPEG (TESTED & WORKING on your Jetson!)
        # Since GStreamer is not available, prioritize V4L2 which works
        
        # Priority 2: CSI Camera (if available)
        gst_csi = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1,format=NV12 ! "
            "nvvidconv ! "
            "video/x-raw,width=640,height=480,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        # Priority 3: GStreamer USB MJPEG (if GStreamer available)
        gst_usb_mjpeg = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "image/jpeg,width=640,height=480,framerate=30/1 ! "
            "nvjpegdec ! "  # Hardware JPEG decode
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        # Priority 4: GStreamer Software JPEG
        gst_usb_mjpeg_sw = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "image/jpeg,width=640,height=480,framerate=30/1 ! "
            "jpegdec ! "  # Software JPEG decode
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        # Priority 5: GStreamer YUYV
        gst_usb_yuyv = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        
        # OPTIMIZED: Try V4L2 first (proven to work), then GStreamer, then fallbacks
        return [
            (camera_id, cv2.CAP_V4L2),      # Try V4L2 FIRST (works!)
            (camera_id, cv2.CAP_ANY),       # CAP_ANY as backup
            (gst_csi, cv2.CAP_GSTREAMER),   # CSI camera (if available)
            (gst_usb_mjpeg, cv2.CAP_GSTREAMER),    # GStreamer MJPEG HW
            (gst_usb_mjpeg_sw, cv2.CAP_GSTREAMER), # GStreamer MJPEG SW
            (gst_usb_yuyv, cv2.CAP_GSTREAMER),     # GStreamer YUYV
            (camera_id, None)               # Direct index (last resort)
        ]
    
    elif system == "Linux":
        print("🐧 Linux detected - Using V4L2 GStreamer")
        gst = (
            f"v4l2src device=/dev/video{camera_id} ! "
            "video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=2"
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
        self._stop_event = threading.Event()  # More reliable stop signal
        
        # Thread references for proper cleanup
        self._video_thread = None
        self._ai_thread = None
        
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
        
        # Performance settings (Adaptive for CUDA/CPU)
        # OPTIMIZED FOR JETSON: Balance quality and speed
        if hasattr(self.detector, 'is_jetson') and self.detector.is_jetson:
            if self.detector.device == 'cuda':
                self.jpeg_quality = 60  # Good balance
            else:
                self.jpeg_quality = 55  # CPU: Slightly lower
        else:
            self.jpeg_quality = 70  # Higher for powerful machines
    
    def _open_camera(self):
        """Open camera with platform-specific optimizations and better error handling"""
        if self.cap is not None and self.cap.isOpened():
            return True
        
        print(f"📹 Opening camera {self.camera_id}...")
        start_time = time.time()
        
        # Try multiple pipelines based on platform
        pipelines = get_camera_pipeline(self.camera_id)
        
        for idx, (pipeline, backend) in enumerate(pipelines):
            try:
                print(f"   Attempt {idx+1}/{len(pipelines)}: {pipeline if isinstance(pipeline, str) else f'Camera {pipeline}'}")
                
                # Important: Add small delay between attempts on Jetson to prevent segfault
                if idx > 0:
                    time.sleep(0.3)
                
                # Handle None backend (direct index)
                if backend is None:
                    self.cap = cv2.VideoCapture(pipeline)
                else:
                    self.cap = cv2.VideoCapture(pipeline, backend)
                
                if not self.cap.isOpened():
                    print(f"    ❌ Failed to open")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                    continue
                
                # Set properties BEFORE testing read (important for Jetson stability)
                if backend not in [cv2.CAP_GSTREAMER, None]:
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # No buffering for low latency
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG for fast capture
                    except:
                        pass  # Some properties may not be supported
                    # Add small delay for camera to stabilize
                    time.sleep(0.3)
                
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
        
        # Import camera manager for lock management
        from app.services.camera_manager import acquire_camera_lock, release_camera_lock
        
        # Acquire camera lock before opening
        if not acquire_camera_lock("pothole", timeout=3.0):
            print("❌ Could not acquire camera lock - another service may be using the camera")
            return False
        
        # Open camera if not already open
        if self.cap is None or not self.cap.isOpened():
            if not self._open_camera():
                release_camera_lock("pothole")  # Release lock if camera open fails
                return False
        else:
            print("Camera already initialized, starting threads...")
        
        self.is_running = True
        self._stop_event.clear()  # Clear stop event before starting threads
        
        # Thread 1: FAST video streaming (no AI)
        self._video_thread = threading.Thread(target=self._video_stream_loop, daemon=True)
        self._video_thread.start()
        
        # Thread 2: SLOW AI inference (independent)
        self._ai_thread = threading.Thread(target=self._ai_inference_loop, daemon=True)
        self._ai_thread.start()
        
        print("✓ Two parallel threads started:")
        print("  - Thread 1: Video streaming (30 FPS)")
        print("  - Thread 2: AI inference (REAL-TIME, adaptive)")
        
        return True
    
    def _video_stream_loop(self):
        """
        THREAD 1: FAST Video Streaming - SMOOTH LAG-FREE FOR JETSON
        Applies latest detection overlay and streams at 15 FPS (Jetson optimized)
        """
        frame_counter = 0
        last_gc = time.time()
        
        while self.is_running and not self._stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                break
            
            # Quick check for stop signal
            if self._stop_event.is_set():
                break
            
            # More frequent garbage collection to prevent memory buildup (every 10 seconds)
            current_time = time.time()
            if current_time - last_gc > 10.0:  # More frequent on Jetson
                gc.collect()
                if hasattr(self.detector, 'device') and self.detector.device == 'cuda':
                    torch.cuda.empty_cache()
                last_gc = current_time
                print(f"🗑️ Pothole: GC performed (frame {frame_counter})")
            
            # SMOOTH APPROACH: Always grab twice - first discards buffered, second is fresh
            # This prevents lag WITHOUT causing stutter
            self.cap.grab()  # Discard potentially stale frame
            
            # Now grab and retrieve the fresh frame
            if not self.cap.grab():
                if self._stop_event.wait(timeout=0.001):
                    break
                continue
            
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                if self._stop_event.wait(timeout=0.001):
                    break
                continue
            
            frame_counter += 1
            
            # Save raw frame for AI thread + get latest mask (single lock)
            with self.lock:
                self.latest_frame = frame.copy()
                current_mask = self.current_mask
            
            # Quick check for stop signal before heavy operations
            if self._stop_event.is_set():
                del frame
                break
            
            # Apply overlay immediately if we have detection
            if current_mask is not None:
                color_mask = np.zeros_like(frame)
                color_mask[current_mask > 0] = [0, 0, 255]  # Red
                overlay_frame = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
            else:
                overlay_frame = frame
            
            # Smart resize for Jetson - smaller but good quality
            if hasattr(self.detector, 'is_jetson') and self.detector.is_jetson:
                overlay_frame = cv2.resize(overlay_frame, (480, 360), interpolation=cv2.INTER_AREA)
            
            # Fast JPEG encoding with optimization flag
            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality,
                int(cv2.IMWRITE_JPEG_OPTIMIZE), 1  # Enable fast encoding
            ]
            ret, buffer = cv2.imencode('.jpg', overlay_frame, encode_param)
            
            # Clean up overlay_frame immediately
            del overlay_frame
            
            if ret:
                frame_bytes = buffer.tobytes()
                del buffer  # Free buffer immediately
                
                with self.lock:
                    # Delete old encoded frame before replacing
                    if self.encoded_frame is not None:
                        del self.encoded_frame
                    self.encoded_frame = frame_bytes
        
        print("🛑 Pothole video thread exiting")
    
    def _ai_inference_loop(self):
        """
        THREAD 2: INSTANT AI Inference
        Optimized for absolute minimum latency
        """
        while self.is_running and not self._stop_event.is_set():
            # Quick exit check at start of each iteration
            if self._stop_event.is_set():
                break
                
            # Get latest frame copy (non-blocking)
            frame_small = None
            original_h, original_w = 0, 0
            
            with self.lock:
                if self.latest_frame is None:
                    pass  # Will sleep below
                else:
                    # Adaptive resize based on device (CUDA vs CPU)
                    if hasattr(self.detector, 'device') and self.detector.device == 'cuda':
                        # CUDA MODE: Can handle larger frames for better quality!
                        frame_small = cv2.resize(self.latest_frame, (320, 240), interpolation=cv2.INTER_LINEAR)
                    else:
                        # CPU MODE: Keep small for speed
                        frame_small = cv2.resize(self.latest_frame, (160, 120), interpolation=cv2.INTER_LINEAR)
                    
                    # CRITICAL: Store original dimensions inside lock to prevent race condition
                    original_h, original_w = self.latest_frame.shape[:2]
            
            if frame_small is None:
                if self._stop_event.wait(timeout=0.005):
                    break
                continue
            
            # Quick exit check before heavy AI operation
            if self._stop_event.is_set():
                del frame_small
                break
            
            # Run AI inference immediately (GPU accelerated if CUDA available!)
            mask = self.detector.detect(frame_small)
            
            # Resize mask back to original size for overlay (using stored dimensions)
            mask_full = cv2.resize(mask, (original_w, original_h), 
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
            
            # FRAME SKIP: Only needed on CPU mode to prevent overload
            # GPU mode can handle full-speed processing without throttling
            if hasattr(self.detector, 'is_jetson') and self.detector.is_jetson and self.detector.device == 'cpu':
                if self._stop_event.wait(timeout=0.033):  # ~30 FPS AI processing on CPU mode
                    break
                # CUDA mode: No delay - GPU can process at full speed!
        
        print("🛑 Pothole AI thread exiting")
    
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
        """Stop both video and AI threads - ROBUST cleanup for Jetson"""
        print("🛑 Stopping pothole detection...")
        
        # Signal threads to stop using both mechanisms for reliability
        self.is_running = False
        self._stop_event.set()  # Signal threads via event (faster response)
        
        # Import camera manager for proper cleanup
        from app.services.camera_manager import force_release_camera, release_camera_lock
        
        # Wait for threads with proper join
        is_jetson = hasattr(self.detector, 'is_jetson') and self.detector.is_jetson
        thread_timeout = 2.0 if is_jetson else 1.0
        
        threads = [
            ('video', self._video_thread),
            ('ai', self._ai_thread)
        ]
        
        # First pass: try to join all threads gracefully
        for name, thread in threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=thread_timeout)
                if thread.is_alive():
                    print(f"⚠️ Pothole {name} thread did not stop after {thread_timeout}s")
        
        # Second pass: check again after a small delay
        time.sleep(0.1)
        for name, thread in threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)
                if thread.is_alive():
                    print(f"⚠️ Pothole {name} thread still running - will be abandoned (daemon thread)")
        
        # Clear thread references
        self._video_thread = None
        self._ai_thread = None
        
        # Release camera with proper Jetson cleanup
        if self.cap is not None:
            force_release_camera(self.cap, "pothole")
            self.cap = None
        
        # Release camera lock
        release_camera_lock("pothole")
        
        # Clear all buffers
        with self.lock:
            self.latest_frame = None
            self.encoded_frame = None
            self.current_mask = None
            self.pothole_detected = False
        
        # Clear detection history
        self.detection_history = []
        self.is_currently_detecting = False
        self.frames_since_detection = 0
        
        # Force garbage collection on Jetson
        gc.collect()
        if hasattr(self.detector, 'device') and self.detector.device == 'cuda':
            torch.cuda.empty_cache()
        
        # Additional delay on Jetson for camera driver to fully release
        if is_jetson:
            time.sleep(0.2)
        
        print("✓ Pothole detection stopped")
    
    def is_active(self):
        """Check if stream is active"""
        return self.is_running and self.cap is not None and self.cap.isOpened()
