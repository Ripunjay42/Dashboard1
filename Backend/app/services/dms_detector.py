import cv2
import numpy as np
import threading
import time
import platform
import os
import gc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Jetson Orin Nano: Prevent segmentation faults
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

# Global model instance for reuse
_global_detector = None
_detector_lock = threading.Lock()


class DMSDetector:
    """Driver Monitoring System - Drowsiness and Yawn Detection using MediaPipe"""
    
    def __init__(self):
        print(f"🚀 Loading DMS Detection models (JETSON OPTIMIZED)...")
        start_time = time.time()
        
        # Detect if running on Jetson device
        self.is_jetson = self._detect_jetson()
        
        # Import MediaPipe
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_face_mesh = mp.solutions.face_mesh
            
            # Configure FaceMesh based on platform
            if self.is_jetson:
                # Jetson: More aggressive optimization
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,  # Huge speed improvement
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("   ✓ MediaPipe FaceMesh loaded (Jetson optimized)")
            else:
                # PC: Can handle more features
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6
                )
                print("   ✓ MediaPipe FaceMesh loaded (PC mode)")
                
        except ImportError as e:
            print(f"❌ MediaPipe not installed: {e}")
            print("   Install with: pip install mediapipe")
            raise
        
        # Landmark indices for detection
        self.LEFT_EYE = [386, 374, 263, 362]
        self.RIGHT_EYE = [159, 145, 33, 133]
        self.MOUTH = [61, 291, 13, 14, 17, 18]
        
        # Detection thresholds
        self.EAR_THRESHOLD = 0.20  # Eye Aspect Ratio threshold for drowsiness
        self.YAWN_THRESHOLD = 0.40  # Mouth aspect ratio threshold for yawning
        
        # Cooldown to prevent spam alerts
        self.last_drowsy_alert = 0
        self.last_yawn_alert = 0
        self.alert_cooldown = 3.0  # seconds between alerts
        
        load_time = time.time() - start_time
        print(f"✓ DMS models loaded in {load_time:.2f}s")
    
    def _detect_jetson(self):
        """Detect if running on Jetson device"""
        try:
            if os.path.exists('/etc/nv_tegra_release'):
                return True
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    return 'jetson' in model or 'tegra' in model
            return False
        except:
            return False
    
    def _dist(self, a, b):
        """Fast Euclidean distance"""
        return np.hypot(a[0] - b[0], a[1] - b[1])
    
    def _calc_EAR(self, landmarks, w, h):
        """Calculate Eye Aspect Ratio"""
        l = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.LEFT_EYE]
        r = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.RIGHT_EYE]
        
        # Avoid division by zero
        left_horizontal = self._dist(l[2], l[3])
        right_horizontal = self._dist(r[2], r[3])
        
        if left_horizontal == 0 or right_horizontal == 0:
            return 1.0  # Return normal EAR if can't calculate
        
        left_ear = self._dist(l[0], l[1]) / left_horizontal
        right_ear = self._dist(r[0], r[1]) / right_horizontal
        return (left_ear + right_ear) / 2
    
    def _calc_yawn(self, landmarks, w, h):
        """Calculate mouth aspect ratio for yawn detection"""
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.MOUTH]
        
        vertical = self._dist(pts[2], pts[3])  # upper to lower lip
        horizontal = self._dist(pts[0], pts[1])  # mouth width
        
        if horizontal == 0:
            return 0.0
        
        return vertical / horizontal
    
    def _draw_box(self, frame, landmarks, indices, w, h, color):
        """Draw bounding box around facial feature"""
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        cv2.rectangle(frame, (min(x) - 5, min(y) - 5), (max(x) + 5, max(y) + 5), color, 2)
    
    def process_frame(self, frame):
        """
        Process a single frame for drowsiness and yawn detection
        Returns: (processed_frame, is_drowsy, is_yawning, ear_value, yawn_value)
        """
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with FaceMesh
        results = self.face_mesh.process(rgb)
        
        is_drowsy = False
        is_yawning = False
        ear_value = 0.0
        yawn_value = 0.0
        current_time = time.time()
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR for drowsiness detection
            ear_value = self._calc_EAR(landmarks, w, h)
            if ear_value < self.EAR_THRESHOLD:
                is_drowsy = True
                # Draw boxes around eyes
                self._draw_box(frame, landmarks, self.LEFT_EYE, w, h, (0, 0, 255))
                self._draw_box(frame, landmarks, self.RIGHT_EYE, w, h, (0, 0, 255))
                cv2.putText(frame, "DROWSY!", (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 0, 255), 2)
            
            # Calculate yawn ratio
            yawn_value = self._calc_yawn(landmarks, w, h)
            if yawn_value > self.YAWN_THRESHOLD:
                is_yawning = True
                # Draw box around mouth
                self._draw_box(frame, landmarks, self.MOUTH, w, h, (255, 165, 0))
                cv2.putText(frame, "YAWNING!", (40, 130), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 165, 0), 2)
            
            # Draw face mesh outline (minimal for performance)
            # Just draw a simple face bounding box
            face_pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                       for i in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]]
            if face_pts:
                x_coords = [p[0] for p in face_pts]
                y_coords = [p[1] for p in face_pts]
                cv2.rectangle(frame, (min(x_coords), min(y_coords)), 
                            (max(x_coords), max(y_coords)), (0, 255, 0), 1)
        else:
            # No face detected
            cv2.putText(frame, "No Face Detected", (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (128, 128, 128), 2)
        
        # Add status overlay
        status_color = (0, 255, 0)  # Green = OK
        if is_drowsy:
            status_color = (0, 0, 255)  # Red = Drowsy
        elif is_yawning:
            status_color = (255, 165, 0)  # Orange = Yawning
        
        cv2.putText(frame, f"EAR: {ear_value:.2f}", (w - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"YAWN: {yawn_value:.2f}", (w - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return frame, is_drowsy, is_yawning, ear_value, yawn_value


def get_global_detector():
    """Get or create the global DMS detector instance (singleton pattern)"""
    global _global_detector
    with _detector_lock:
        if _global_detector is None:
            _global_detector = DMSDetector()
    return _global_detector


def get_camera_pipeline(camera_id=0):
    """Get optimized camera pipeline based on platform"""
    system = platform.system()
    
    # Detect if running on Jetson device
    is_jetson = os.path.exists('/etc/nv_tegra_release')
    
    if is_jetson:
        print("📹 Jetson detected - Using V4L2 backend for DMS")
        return [
            (camera_id, cv2.CAP_V4L2),
            (camera_id, cv2.CAP_ANY),
            (camera_id, None)
        ]
    elif system == "Linux":
        return [(camera_id, cv2.CAP_V4L2), (camera_id, cv2.CAP_ANY)]
    else:
        # Windows
        return [(camera_id, cv2.CAP_DSHOW), (camera_id, cv2.CAP_ANY)]


class DMSStreamManager:
    """
    DMS Stream Manager - Two Thread Architecture (JETSON OPTIMIZED)
    
    Thread 1 (Video): Camera → Apply Overlay → Encode → Stream (FAST, no AI)
    Thread 2 (AI): Copy frame → MediaPipe → Update status (SLOW, independent)
    
    Result: Smooth video + accurate detection, NO LAG!
    """
    
    def __init__(self, camera_id=0):
        self.detector = get_global_detector()
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self._stop_event = threading.Event()  # More reliable stop signal
        
        # Thread references for proper cleanup
        self._video_thread = None
        self._ai_thread = None
        
        # Detect Jetson
        self.is_jetson = os.path.exists('/etc/nv_tegra_release')
        
        # Shared buffers (thread-safe) - TWO THREAD ARCHITECTURE
        self.latest_frame = None  # Raw frame from camera (for AI thread)
        self.encoded_frame = None  # JPEG frame (for streaming)
        self.current_overlay = None  # Detection overlay from AI thread
        
        # Detection status (updated by AI thread)
        self.is_drowsy = False
        self.is_yawning = False
        self.ear_value = 0.0
        self.yawn_value = 0.0
        
        # AI frame skip counter for Jetson optimization
        self.ai_frame_skip = 3 if self.is_jetson else 2  # Process every Nth frame
        
        # Alert state with persistence (smoother alerts)
        self.drowsy_alert_active = False
        self.yawn_alert_active = False
        self.last_drowsy_time = 0
        self.last_yawn_time = 0
        self.alert_persistence = 2.0  # Keep alert active for 2 seconds
        
        # Performance settings (JETSON OPTIMIZED)
        if self.is_jetson:
            self.jpeg_quality = 50  # Lower quality for faster encoding
            self.frame_size = (400, 300)  # Smaller for Jetson
        else:
            self.jpeg_quality = 70
            self.frame_size = (640, 480)
    
    def _open_camera(self):
        """Open camera with platform-specific optimizations"""
        if self.cap is not None and self.cap.isOpened():
            return True
        
        print(f"📹 Opening DMS camera {self.camera_id}...")
        start_time = time.time()
        
        pipelines = get_camera_pipeline(self.camera_id)
        
        for idx, (pipeline, backend) in enumerate(pipelines):
            try:
                print(f"   Attempt {idx+1}: Camera {pipeline}")
                
                if idx > 0:
                    time.sleep(0.3)
                
                if backend is None:
                    self.cap = cv2.VideoCapture(pipeline)
                else:
                    self.cap = cv2.VideoCapture(pipeline, backend)
                
                if not self.cap.isOpened():
                    continue
                
                # Configure camera
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except:
                    pass
                
                time.sleep(0.3)
                
                # Test read
                for attempt in range(5):
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"   ✓ Camera opened in {time.time() - start_time:.2f}s")
                        return True
                    time.sleep(0.2)
                
                self.cap.release()
                
            except Exception as e:
                print(f"   Failed: {e}")
                if self.cap:
                    self.cap.release()
        
        print("❌ Failed to open DMS camera")
        self.cap = None
        return False
    
    def start(self):
        """Start DMS detection with TWO parallel threads"""
        if self.is_running:
            return True
        
        # Import camera manager for lock management
        from app.services.camera_manager import acquire_camera_lock, release_camera_lock
        
        # Acquire camera lock before opening
        if not acquire_camera_lock("dms", timeout=3.0):
            print("❌ Could not acquire camera lock - another service may be using the camera")
            return False
        
        if not self._open_camera():
            release_camera_lock("dms")  # Release lock if camera open fails
            return False
        
        self.is_running = True
        self._stop_event.clear()  # Clear stop event before starting threads
        
        # Thread 1: FAST video streaming (no AI, just overlay)
        self._video_thread = threading.Thread(target=self._video_stream_loop, daemon=True)
        self._video_thread.start()
        
        # Thread 2: SLOW AI inference (MediaPipe, independent)
        self._ai_thread = threading.Thread(target=self._ai_inference_loop, daemon=True)
        self._ai_thread.start()
        
        print("✓ DMS detection started (TWO THREAD ARCHITECTURE)")
        print("   - Thread 1: Video streaming (30 FPS, smooth)")
        print("   - Thread 2: AI inference (MediaPipe, independent)")
        return True
    
    def _video_stream_loop(self):
        """
        THREAD 1: FAST Video Streaming - SMOOTH LAG-FREE
        Only captures, applies overlay, and encodes. NO AI processing here.
        """
        frame_counter = 0
        last_gc = time.time()
        
        while self.is_running and not self._stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                break
            
            # Quick check for stop signal
            if self._stop_event.is_set():
                break
            
            # More frequent garbage collection to prevent memory buildup
            current_time = time.time()
            if current_time - last_gc > 10.0:  # More frequent on Jetson
                gc.collect()
                last_gc = current_time
                print(f"🗑️ DMS: GC performed (frame {frame_counter})")
            
            # SMOOTH APPROACH: Double grab - first discards buffered, second is fresh
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
            
            # Save raw frame for AI thread + get latest overlay
            with self.lock:
                self.latest_frame = frame.copy()
                current_overlay = self.current_overlay
                is_drowsy = self.drowsy_alert_active
                is_yawning = self.yawn_alert_active
                ear = self.ear_value
                yawn = self.yawn_value
            
            # Quick check for stop signal before heavy operations
            if self._stop_event.is_set():
                del frame
                break
            
            # Apply overlay from AI thread (if available)
            if current_overlay is not None:
                display_frame = current_overlay.copy()
            else:
                display_frame = frame.copy()
                # Add "Initializing..." text if no overlay yet
                cv2.putText(display_frame, "Initializing DMS...", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Resize for streaming
            frame_resized = cv2.resize(display_frame, self.frame_size, 
                                       interpolation=cv2.INTER_AREA)
            
            # Fast JPEG encoding
            encode_param = [
                cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            ret_enc, buffer = cv2.imencode('.jpg', frame_resized, encode_param)
            
            # Clean up frames immediately
            del frame, display_frame, frame_resized
            
            if ret_enc:
                frame_bytes = buffer.tobytes()
                del buffer  # Free buffer immediately
                
                with self.lock:
                    # Delete old encoded frame before replacing
                    if self.encoded_frame is not None:
                        del self.encoded_frame
                    self.encoded_frame = frame_bytes
        
        print("🛑 DMS video thread exiting")
    
    def _ai_inference_loop(self):
        """
        THREAD 2: AI Inference with MediaPipe - JETSON OPTIMIZED
        Processes frames and updates overlay/status without blocking video
        """
        frame_counter = 0
        
        while self.is_running and not self._stop_event.is_set():
            # Quick exit check at start of each iteration
            if self._stop_event.is_set():
                break
            
            # Frame skipping for Jetson optimization (process every Nth frame)
            frame_counter += 1
            if frame_counter % self.ai_frame_skip != 0:
                if self._stop_event.wait(timeout=0.01):
                    break
                continue
                
            # Get latest frame copy (non-blocking)
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    # Resize frame for faster MediaPipe processing on Jetson
                    if self.is_jetson:
                        frame = cv2.resize(self.latest_frame, (320, 240), interpolation=cv2.INTER_LINEAR)
                    else:
                        frame = self.latest_frame.copy()
            
            if frame is None:
                if self._stop_event.wait(timeout=0.01):
                    break
                continue
            
            # Quick exit check before heavy AI operation
            if self._stop_event.is_set():
                del frame
                break
            
            # Process frame with DMS detector (MediaPipe)
            processed_frame, is_drowsy, is_yawning, ear, yawn = self.detector.process_frame(frame)
            
            # Resize back to stream size for overlay
            if self.is_jetson and processed_frame is not None:
                processed_frame = cv2.resize(processed_frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
            
            # Update alert states with persistence
            current_time = time.time()
            
            if is_drowsy:
                self.last_drowsy_time = current_time
                self.drowsy_alert_active = True
            elif current_time - self.last_drowsy_time > self.alert_persistence:
                self.drowsy_alert_active = False
            
            if is_yawning:
                self.last_yawn_time = current_time
                self.yawn_alert_active = True
            elif current_time - self.last_yawn_time > self.alert_persistence:
                self.yawn_alert_active = False
            
            # Update shared state with lock
            with self.lock:
                self.current_overlay = processed_frame
                self.is_drowsy = self.drowsy_alert_active
                self.is_yawning = self.yawn_alert_active
                self.ear_value = ear
                self.yawn_value = yawn
            
            # Frame skipping is now handled at the start of the loop
        
        print("🛑 DMS AI thread exiting")
    
    def get_encoded_frame(self):
        """Get pre-encoded JPEG frame"""
        with self.lock:
            return self.encoded_frame
    
    def get_status(self):
        """Get current DMS status"""
        with self.lock:
            return {
                'is_drowsy': self.is_drowsy,
                'is_yawning': self.is_yawning,
                'ear_value': float(self.ear_value),
                'yawn_value': float(self.yawn_value)
            }
    
    def is_drowsy_detected(self):
        """Check if drowsiness is detected"""
        with self.lock:
            return self.is_drowsy
    
    def is_yawn_detected(self):
        """Check if yawning is detected"""
        with self.lock:
            return self.is_yawning
    
    def stop(self):
        """Stop DMS detection - ROBUST cleanup for Jetson"""
        print("🛑 Stopping DMS detection...")
        
        # Signal threads to stop using both mechanisms for reliability
        self.is_running = False
        self._stop_event.set()  # Signal threads via event (faster response)
        
        # Import camera manager for proper cleanup
        from app.services.camera_manager import force_release_camera, release_camera_lock
        
        # Wait for threads with proper join
        thread_timeout = 2.0 if self.is_jetson else 1.0
        
        threads = [
            ('video', self._video_thread),
            ('ai', self._ai_thread)
        ]
        
        # First pass: try to join all threads gracefully
        for name, thread in threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=thread_timeout)
                if thread.is_alive():
                    print(f"⚠️ DMS {name} thread did not stop after {thread_timeout}s")
        
        # Second pass: check again after a small delay
        time.sleep(0.1)
        for name, thread in threads:
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.5)
                if thread.is_alive():
                    print(f"⚠️ DMS {name} thread still running - will be abandoned (daemon thread)")
        
        # Clear thread references
        self._video_thread = None
        self._ai_thread = None
        
        # Release camera with proper Jetson cleanup
        if self.cap is not None:
            force_release_camera(self.cap, "dms")
            self.cap = None
        
        # Release camera lock
        release_camera_lock("dms")
        
        # Clear all buffers
        with self.lock:
            self.encoded_frame = None
            self.latest_frame = None
            self.current_overlay = None
            self.is_drowsy = False
            self.is_yawning = False
            self.ear_value = 0.0
            self.yawn_value = 0.0
            self.drowsy_alert_active = False
            self.yawn_alert_active = False
        
        # Force garbage collection
        gc.collect()
        
        # Additional delay on Jetson for camera driver to fully release
        if self.is_jetson:
            time.sleep(0.2)
        
        print("✓ DMS detection stopped")
    
    def is_active(self):
        """Check if DMS is active"""
        return self.is_running and self.cap is not None and self.cap.isOpened()
