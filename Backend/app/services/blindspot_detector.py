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

# Configuration
FRAME_W, FRAME_H = 640, 480
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
        print(f"⚡ Loading Blind Spot Detection models (OPTIMIZED MODE)...")
        start_time = time.time()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO
        self.yolo = YOLO("yolov8s.pt")
        
        # Load MiDaS depth estimation
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas_trans = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        self.midas.to(self.device).eval()
        
        load_time = time.time() - start_time
        print(f" Blind Spot models loaded in {load_time:.2f}s on {self.device}")
    
    def draw_side_mirror_grid(self, frame, color, is_left=True):
        """Generate SIDE-MIRROR style 3D cage"""
        h, w = frame.shape[:2]
        base_y = int(h * 0.90)
        
        # Perspective skew for side-mirror effect
        skew = -120 if is_left else +120
        
        # Front width vs back width
        front_w = int(w * 0.55)
        mid_w = int(w * 0.40)
        back_w = int(w * 0.22)
        
        # Depths
        d1 = 120
        d2 = 230
        d3 = 340
        
        # Vertical height of cage
        height = 110
        
        # Define TOP vertices
        TL1 = (int(w/2 - front_w/2), base_y - d1)
        TR1 = (int(w/2 + front_w/2), base_y - d1)
        
        TL2 = (int(w/2 - mid_w/2) + skew, base_y - d2)
        TR2 = (int(w/2 + mid_w/2) + skew, base_y - d2)
        
        TL3 = (int(w/2 - back_w/2) + skew*2, base_y - d3)
        TR3 = (int(w/2 + back_w/2) + skew*2, base_y - d3)
        
        # Define BOTTOM vertices
        BL1 = (TL1[0], TL1[1] + height)
        BR1 = (TR1[0], TR1[1] + height)
        
        BL2 = (TL2[0], TL2[1] + height)
        BR2 = (TR2[0], TR2[1] + height)
        
        BL3 = (TL3[0], TL3[1] + height)
        BR3 = (TR3[0], TR3[1] + height)
        
        # DRAW CAGE STRUCTURE
        lines = [
            # vertical pillars
            (TL1, BL1), (TR1, BR1),
            (TL2, BL2), (TR2, BR2),
            (TL3, BL3), (TR3, BR3),
            
            # top horizontal edges
            (TL1, TR1), (TL2, TR2), (TL3, TR3),
            (TL1, TL2), (TL2, TL3),
            (TR1, TR2), (TR2, TR3),
            
            # bottom ground edges
            (BL1, BR1), (BL2, BR2), (BL3, BR3),
            (BL1, BL2), (BL2, BL3),
            (BR1, BR2), (BR2, BR3),
            
            # diagonal supports
            (BL1, TL2), (BR1, TR2),
            (BL2, TL3), (BR2, TR3),
        ]
        
        for a, b in lines:
            cv2.line(frame, a, b, color, 2)
        
        # Polygon for inside check
        polygon = np.array([BL1, BR1, BR2, BL2])
        
        return polygon
    
    def inside(self, poly, cx, cy):
        """Check if point is inside blind-spot cage"""
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
    
    def process_frame(self, frame, is_left=True):
        """Process single frame for blind spot detection"""
        # Depth estimation
        inp = self.midas_trans(frame).to(self.device)
        with torch.no_grad():
            d = self.midas(inp).squeeze().cpu().numpy()
        
        depth = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # YOLO detection
        dets = self.yolo(frame)[0]
        
        # Default grid color (green)
        color = (0, 255, 0)
        danger = False
        
        # Draw initial grid
        poly = self.draw_side_mirror_grid(frame, color, is_left)
        
        # Evaluate detections
        for box in dets.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Depth-based distance
            distance = (255 - depth[cy, cx]) / DEPTH_SCALE
            
            # Condition 1: too close
            if distance < NEAR_THRESHOLD:
                danger = True
            
            # Condition 2: inside cage polygon
            if self.inside(poly, cx, cy):
                danger = True
        
        # If danger → redraw cage as RED
        if danger:
            self.draw_side_mirror_grid(frame, (0, 0, 255), is_left)
        
        return frame, danger


class DualCameraManager:
    """
    HIGH PERFORMANCE ARCHITECTURE: Dual Camera Blind Spot Detection
    
    Two parallel threads for each camera (4 threads total):
    - Thread 1: Left Camera Video Stream
    - Thread 2: Left Camera AI Detection
    - Thread 3: Right Camera Video Stream
    - Thread 4: Right Camera AI Detection
    """
    
    def __init__(self, left_cam_id=0, right_cam_id=1):
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        
        self.left_cap = None
        self.right_cap = None
        
        self.detector = None
        
        self.left_frame_queue = Queue(maxsize=2)
        self.right_frame_queue = Queue(maxsize=2)
        
        self.left_encoded = None
        self.right_encoded = None
        
        self.left_danger = False
        self.right_danger = False
        
        self.running = False
        self.active = False
        
        self.left_video_thread = None
        self.right_video_thread = None
        self.left_ai_thread = None
        self.right_ai_thread = None
        
        self.lock = threading.Lock()
    
    def _open_cameras(self):
        """Open both cameras with optimized settings"""
        if self.left_cap is not None and self.right_cap is not None:
            if self.left_cap.isOpened() and self.right_cap.isOpened():
                return True
        
        print(f"📹 Opening cameras {self.left_cam_id} and {self.right_cam_id}...")
        import time
        start_time = time.time()
        
        # Use DirectShow backend on Windows for faster camera opening
        import platform
        system = platform.system()
        
        if system == "Windows":
            # DirectShow is much faster than default backend on Windows
            self.left_cap = cv2.VideoCapture(self.left_cam_id, cv2.CAP_DSHOW)
            self.right_cap = cv2.VideoCapture(self.right_cam_id, cv2.CAP_DSHOW)
        else:
            self.left_cap = cv2.VideoCapture(self.left_cam_id)
            self.right_cap = cv2.VideoCapture(self.right_cam_id)
        
        # Set buffer size for low latency
        self.left_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.right_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set resolution
        self.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        
        # Set FPS
        self.left_cap.set(cv2.CAP_PROP_FPS, 30)
        self.right_cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Set MJPEG codec for faster capture (if supported)
        self.left_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.right_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        if not self.left_cap.isOpened() or not self.right_cap.isOpened():
            print("Failed to open cameras")
            return False
        
        open_time = time.time() - start_time
        print(f"Cameras opened in {open_time:.2f}s")
        return True
    
    def start(self):
        """Start dual camera blind spot detection"""
        if self.active:
            return True
        
        if not self._open_cameras():
            return False
        
        # Load detector
        if self.detector is None:
            self.detector = get_global_detector()
        
        self.running = True
        self.active = True
        
        # Start threads for left camera
        self.left_video_thread = threading.Thread(target=self._left_video_loop, daemon=True)
        self.left_ai_thread = threading.Thread(target=self._left_ai_loop, daemon=True)
        
        # Start threads for right camera
        self.right_video_thread = threading.Thread(target=self._right_video_loop, daemon=True)
        self.right_ai_thread = threading.Thread(target=self._right_ai_loop, daemon=True)
        
        self.left_video_thread.start()
        self.left_ai_thread.start()
        self.right_video_thread.start()
        self.right_ai_thread.start()
        
        print("Blind spot detection started")
        return True
    
    def _left_video_loop(self):
        """Left camera video streaming loop"""
        while self.running:
            ret, frame = self.left_cap.read()
            if not ret:
                continue
            
            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.lock:
                self.left_encoded = buffer.tobytes()
            
            # Send copy to AI thread
            if not self.left_frame_queue.full():
                self.left_frame_queue.put(frame.copy())
    
    def _right_video_loop(self):
        """Right camera video streaming loop"""
        while self.running:
            ret, frame = self.right_cap.read()
            if not ret:
                continue
            
            # Encode frame for streaming
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.lock:
                self.right_encoded = buffer.tobytes()
            
            # Send copy to AI thread
            if not self.right_frame_queue.full():
                self.right_frame_queue.put(frame.copy())
    
    def _left_ai_loop(self):
        """Left camera AI detection loop"""
        while self.running:
            if self.left_frame_queue.empty():
                time.sleep(0.01)
                continue
            
            frame = self.left_frame_queue.get()
            processed_frame, danger = self.detector.process_frame(frame, is_left=True)
            
            with self.lock:
                self.left_danger = danger
    
    def _right_ai_loop(self):
        """Right camera AI detection loop"""
        while self.running:
            if self.right_frame_queue.empty():
                time.sleep(0.01)
                continue
            
            frame = self.right_frame_queue.get()
            processed_frame, danger = self.detector.process_frame(frame, is_left=False)
            
            with self.lock:
                self.right_danger = danger
    
    def get_left_frame(self):
        """Get encoded left camera frame"""
        with self.lock:
            return self.left_encoded
    
    def get_right_frame(self):
        """Get encoded right camera frame"""
        with self.lock:
            return self.right_encoded
    
    def is_left_danger(self):
        """Check if left blind spot has danger"""
        with self.lock:
            return bool(self.left_danger)
    
    def is_right_danger(self):
        """Check if right blind spot has danger"""
        with self.lock:
            return bool(self.right_danger)
    
    def stop(self):
        """Stop blind spot detection"""
        self.running = False
        self.active = False
        
        # Wait for threads to finish
        if self.left_video_thread:
            self.left_video_thread.join(timeout=2)
        if self.right_video_thread:
            self.right_video_thread.join(timeout=2)
        if self.left_ai_thread:
            self.left_ai_thread.join(timeout=2)
        if self.right_ai_thread:
            self.right_ai_thread.join(timeout=2)
        
        # Release cameras
        if self.left_cap:
            self.left_cap.release()
            self.left_cap = None
        if self.right_cap:
            self.right_cap.release()
            self.right_cap = None
        
        print("⏹ Blind spot detection stopped")
    
    def is_active(self):
        """Check if detection is active"""
        return self.active
