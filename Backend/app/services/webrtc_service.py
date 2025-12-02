"""
WebRTC Video Streaming Service
==============================
Uses aiortc for WebRTC video streaming - much smoother than MJPEG!
Falls back gracefully when aiortc is not installed.
"""

import threading
import time

# Try importing aiortc components
try:
    import asyncio
    import uuid
    import cv2
    import numpy as np
    from fractions import Fraction
    from typing import Dict, Optional
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import VideoStreamTrack
    from aiortc.contrib.media import MediaRelay
    from av import VideoFrame
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False
    print("aiortc not installed. WebRTC disabled. Run: pip install aiortc av")


def is_webrtc_available():
    """Check if WebRTC is available"""
    return AIORTC_AVAILABLE


def get_webrtc_manager():
    """Get or create the global WebRTC manager"""
    if not AIORTC_AVAILABLE:
        return None
    global _webrtc_manager
    with _manager_lock:
        if _webrtc_manager is None:
            _webrtc_manager = WebRTCStreamManager()
    return _webrtc_manager


# Only define WebRTC classes if aiortc is available
if AIORTC_AVAILABLE:
    
    class FrameVideoTrack(VideoStreamTrack):
        """Custom VideoStreamTrack that serves frames from our detectors."""
        
        kind = "video"
        
        def __init__(self, track_id, fps=20):
            super().__init__()
            self.track_id = track_id
            self._fps = fps
            self._frame_count = 0
            self._current_frame = None
            self._frame_lock = threading.Lock()
            self._running = True
            self._placeholder = self._create_placeholder()
            print(f"WebRTC track created: {track_id} @ {fps} FPS")
        
        def _create_placeholder(self):
            """Create a placeholder frame for when no video is available"""
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for video...", (180, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            return frame
        
        def update_frame(self, frame):
            """Update the current frame (called by detector threads)"""
            if frame is None:
                return
            with self._frame_lock:
                self._current_frame = frame.copy()
        
        def stop(self):
            """Stop the track"""
            self._running = False
        
        async def recv(self):
            """Called by aiortc to get the next frame."""
            pts, time_base = await self.next_timestamp()
            
            with self._frame_lock:
                if self._current_frame is not None:
                    frame = self._current_frame.copy()
                else:
                    frame = self._placeholder.copy()
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
            video_frame.pts = pts
            video_frame.time_base = time_base
            
            self._frame_count += 1
            return video_frame


    class WebRTCStreamManager:
        """Manages WebRTC connections for all video streams."""
        
        def __init__(self):
            self._peer_connections = {}
            self._tracks = {}
            self._relay = MediaRelay()
            self._lock = threading.Lock()
            self._loop = None
            self._loop_thread = None
            self._start_event_loop()
            print("WebRTC Stream Manager initialized")
        
        def _start_event_loop(self):
            """Start a background event loop for async WebRTC operations"""
            def run_loop():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()
            
            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()
            
            while self._loop is None:
                time.sleep(0.01)
        
        def _run_async(self, coro):
            """Run an async coroutine from sync code"""
            if self._loop is None:
                raise RuntimeError("Event loop not started")
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=30)
        
        def create_track(self, track_id, fps=20):
            """Create a new video track for a detector"""
            with self._lock:
                if track_id in self._tracks:
                    return self._tracks[track_id]
                track = FrameVideoTrack(track_id, fps)
                self._tracks[track_id] = track
                return track
        
        def get_track(self, track_id):
            """Get an existing track by ID"""
            with self._lock:
                return self._tracks.get(track_id)
        
        def update_frame(self, track_id, frame):
            """Update a track's current frame (called by detectors)"""
            track = self.get_track(track_id)
            if track:
                track.update_frame(frame)
        
        def remove_track(self, track_id):
            """Remove and stop a track"""
            with self._lock:
                if track_id in self._tracks:
                    self._tracks[track_id].stop()
                    del self._tracks[track_id]
        
        async def _create_offer_async(self, track_id):
            """Create WebRTC offer (async)"""
            track = self.get_track(track_id)
            if not track:
                track = self.create_track(track_id)
            if not track:
                return None
            
            pc = RTCPeerConnection()
            connection_id = str(uuid.uuid4())[:8]
            
            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                if pc.connectionState in ["failed", "closed"]:
                    await self._cleanup_connection_async(connection_id)
            
            pc.addTrack(track)
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            while pc.iceGatheringState != "complete":
                await asyncio.sleep(0.1)
            
            with self._lock:
                self._peer_connections[connection_id] = pc
            
            return {
                "connection_id": connection_id,
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
        
        async def _handle_answer_async(self, connection_id, sdp):
            """Handle WebRTC answer from client (async)"""
            with self._lock:
                pc = self._peer_connections.get(connection_id)
            if not pc:
                return False
            answer = RTCSessionDescription(sdp=sdp, type="answer")
            await pc.setRemoteDescription(answer)
            return True
        
        async def _cleanup_connection_async(self, connection_id):
            """Clean up a peer connection (async)"""
            with self._lock:
                pc = self._peer_connections.pop(connection_id, None)
            if pc:
                await pc.close()
        
        def create_offer(self, track_id):
            """Create WebRTC offer (sync wrapper)"""
            return self._run_async(self._create_offer_async(track_id))
        
        def handle_answer(self, connection_id, sdp):
            """Handle WebRTC answer (sync wrapper)"""
            return self._run_async(self._handle_answer_async(connection_id, sdp))
        
        def close_connection(self, connection_id):
            """Close a connection (sync wrapper)"""
            self._run_async(self._cleanup_connection_async(connection_id))
        
        def get_stats(self):
            """Get manager statistics"""
            with self._lock:
                return {
                    "tracks": list(self._tracks.keys()),
                    "connections": len(self._peer_connections),
                    "aiortc_available": True
                }
        
        def cleanup_all(self):
            """Clean up all connections and tracks"""
            with self._lock:
                connection_ids = list(self._peer_connections.keys())
            for conn_id in connection_ids:
                try:
                    self.close_connection(conn_id)
                except:
                    pass
            with self._lock:
                for track in self._tracks.values():
                    track.stop()
                self._tracks.clear()

    # Global WebRTC manager instance
    _webrtc_manager = None
    _manager_lock = threading.Lock()
