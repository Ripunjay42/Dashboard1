import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import threading
import time
from queue import Queue

# Global model instance for reuse
_global_detector = None
_detector_lock = threading.Lock()


class PotholeDetector:
    def __init__(self, model_path, device=None):
        print(f"Loading model from {model_path}...")
        start_time = time.time()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = (256, 256)  # must match training
        self.threshold = 0.5

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

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds on {self.device}")

    def preprocess(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def postprocess(self, output, original_size):
        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (prob_mask > self.threshold).astype(np.uint8)
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask_resized * 255

    def detect(self, frame):
        original_size = frame.shape[:2]
        tensor = self.preprocess(frame)
        with torch.no_grad():
            output = self.model(tensor)
        return self.postprocess(output, original_size)


def get_global_detector(model_path):
    """Get or create the global detector instance (singleton pattern)"""
    global _global_detector
    with _detector_lock:
        if _global_detector is None:
            _global_detector = PotholeDetector(model_path)
    return _global_detector


class VideoStreamManager:
    def __init__(self, model_path, camera_id=0, pre_open_camera=False):
        # Use global detector instance to avoid reloading model
        self.detector = get_global_detector(model_path)
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.frame = None
        self.encoded_frame = None
        self.pothole_detected = False  # Python bool, not NumPy bool_
        self.lock = threading.Lock()
        
        # Performance optimization: Process every Nth frame
        self.process_every_n_frames = 2  # Run inference on every 2nd frame
        self.frame_count = 0
        
        # Queue for frame processing (producer-consumer pattern)
        self.frame_queue = Queue(maxsize=1)  # Single frame buffer for lowest latency
        
        # FPS control
        self.fps_limit = 0.033  # ~30 FPS (1/30 = 0.033 seconds)
        self.last_frame_time = 0
        
        # Keep track of last detection mask to prevent blinking
        self.last_detection_mask = None
        self.detection_persistence_frames = 25  # Keep detection longer (increased from 20)
        
        # Smoothing for detection state (ANTI-BLINKING)
        self.detection_history = []  # Track last N detection results
        self.detection_history_size = 10  # Average over 10 frames (increased from 8)
        self.min_detections_for_active = 7  # Need 7 out of 10 frames to show detection (70% threshold)
        
        # Pre-open camera if requested (for faster startup later)
        if pre_open_camera:
            self._open_camera()
    
    def _open_camera(self):
        """Internal method to open and initialize camera"""
        if self.cap is not None and self.cap.isOpened():
            return True
            
        print(f"Opening camera {self.camera_id}...")
        start_time = time.time()
        
        # Use default backend directly (works best on this system)
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("Failed to open camera!")
            return False
        
        # Optimize camera settings for fast startup and smooth streaming
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Quick warmup - just verify camera can read
        print("Initializing camera...")
        time.sleep(0.3)  # Brief pause for camera initialization
        
        # Try to read one frame to verify camera is working
        ret, _ = self.cap.read()
        if not ret:
            # Try a few more times with small delays
            for i in range(3):
                time.sleep(0.1)
                ret, _ = self.cap.read()
                if ret:
                    print(f"Camera ready after {i+2} attempts")
                    break
        
        if not ret:
            print("Camera opened but failed to read frames after warmup")
            self.cap.release()
            self.cap = None
            return False
        
        open_time = time.time() - start_time
        print(f"✓ Camera opened successfully in {open_time:.2f} seconds")
        return True
        
    def start(self):
        if self.is_running:
            return True
        
        # Open camera if not already open
        if self.cap is None or not self.cap.isOpened():
            if not self._open_camera():
                return False
        else:
            print("✓ Camera already initialized, starting stream immediately...")
            
        self.is_running = True
        
        # Start separate threads for capture and inference
        capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        
        capture_thread.start()
        inference_thread.start()
        
        return True
    
    def _capture_frames(self):
        """Continuously capture frames from camera (producer thread)"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                break
            
            # Control frame rate
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed < self.fps_limit:
                time.sleep(self.fps_limit - elapsed)
                
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"Camera failed to read {max_failures} consecutive frames, stopping...")
                    self.stop()
                    break
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            self.last_frame_time = time.time()
            self.frame_count += 1
            
            # Add frame to queue for processing (non-blocking)
            # If queue is full, skip the frame to avoid lag
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
            # Always update display frame with latest capture (smooth feed)
            # Encode immediately for streaming
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # Lower quality for speed
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if ret:
                with self.lock:
                    self.frame = frame
                    self.encoded_frame = buffer.tobytes()
    
    def _inference_worker(self):
        """Process frames for pothole detection (consumer thread)"""
        last_mask = None
        frames_since_detection = 0
        
        while self.is_running:
            try:
                # Get frame from queue (blocking with timeout)
                frame = self.frame_queue.get(timeout=0.1)
                
                # Only run inference on every Nth frame
                if self.frame_count % self.process_every_n_frames == 0:
                    # Run actual detection
                    mask = self.detector.detect(frame)
                    
                    # Check if pothole is detected (more than 0.1% of frame)
                    pothole_pixels = np.sum(mask > 0)
                    total_pixels = mask.shape[0] * mask.shape[1]
                    detection_threshold = 0.001  # 0.1% of frame
                    
                    is_pothole_detected = (pothole_pixels / total_pixels) > detection_threshold
                    
                    # Add to detection history for smoothing
                    self.detection_history.append(is_pothole_detected)
                    if len(self.detection_history) > self.detection_history_size:
                        self.detection_history.pop(0)
                    
                    # Use threshold voting for stable detection (reduce blinking)
                    # Need at least min_detections_for_active positive detections
                    detections_count = sum(self.detection_history)
                    stable_detection = detections_count >= self.min_detections_for_active
                    
                    if stable_detection:
                        last_mask = mask
                        frames_since_detection = 0
                        # Convert NumPy bool_ to Python bool for JSON serialization
                        with self.lock:
                            self.pothole_detected = True
                    else:
                        frames_since_detection += 1
                        # Keep detection active for persistence_frames to prevent blinking
                        if frames_since_detection > self.detection_persistence_frames:
                            last_mask = None
                            with self.lock:
                                self.pothole_detected = False
                    
                    # Always create overlay if we have a valid mask (including persisted)
                    if last_mask is not None:
                        color_mask = np.zeros_like(frame)
                        color_mask[last_mask > 0] = [0, 0, 255]
                        overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
                        
                        # Encode and update the display frame with overlay
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                        ret, buffer = cv2.imencode('.jpg', overlay, encode_param)
                        
                        if ret:
                            with self.lock:
                                self.encoded_frame = buffer.tobytes()
                else:
                    # For skipped frames, reuse last mask if it exists
                    if last_mask is not None:
                        color_mask = np.zeros_like(frame)
                        color_mask[last_mask > 0] = [0, 0, 255]
                        overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
                        
                        # Encode with last detection overlay
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                        ret, buffer = cv2.imencode('.jpg', overlay, encode_param)
                        
                        if ret:
                            with self.lock:
                                self.encoded_frame = buffer.tobytes()
                
                self.frame_queue.task_done()
                
            except Exception as e:
                # Queue is empty or error occurred
                time.sleep(0.01)
                continue
    
    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    
    def get_encoded_frame(self):
        """Get pre-encoded JPEG frame for faster streaming"""
        with self.lock:
            if self.encoded_frame is None:
                return None
            return self.encoded_frame
    
    def is_pothole_detected(self):
        """Check if pothole is currently detected"""
        with self.lock:
            return self.pothole_detected
    
    def stop(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.frame = None
            self.encoded_frame = None
            self.pothole_detected = False
    
    def is_active(self):
        return self.is_running and self.cap is not None and self.cap.isOpened()
