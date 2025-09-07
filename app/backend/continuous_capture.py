#!/usr/bin/env python3
"""
Continuous screen capture implementation for FASTVLM Jarvis
Replaces event-driven capture with continuous 15-30 fps capture
"""

import asyncio
import numpy as np
import time
import logging
import threading
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import mss
from PIL import Image
import cv2
import hashlib
from datetime import datetime
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Frame:
    """Single captured frame with metadata"""
    timestamp: float
    data: np.ndarray
    frame_hash: str
    motion_score: float = 0.0
    roi_regions: List[Tuple[int, int, int, int]] = None

class ContinuousScreenCapture:
    """Continuous screen capture with ring buffer and motion detection"""
    
    def __init__(
        self,
        fps: int = 15,
        buffer_seconds: int = 20,
        target_size: Tuple[int, int] = (640, 360),  # Reduced for 5K displays
        save_screenshots: bool = False,
        gpu_throttle_threshold: float = 0.8
    ):
        """
        Initialize continuous screen capture
        
        Args:
            fps: Target frames per second (15-30)
            buffer_seconds: How many seconds to keep in ring buffer
            target_size: Downscale resolution for VLM processing
            save_screenshots: Whether to save important frames to disk
            gpu_throttle_threshold: GPU usage threshold to throttle FPS
        """
        self.target_fps = fps
        self.actual_fps = fps
        self.buffer_seconds = buffer_seconds
        self.target_size = target_size
        self.save_screenshots = save_screenshots
        self.gpu_throttle_threshold = gpu_throttle_threshold
        
        # Ring buffer for frame history
        buffer_capacity = fps * buffer_seconds
        self.ring_buffer = deque(maxlen=buffer_capacity)
        
        # Capture state
        self.is_running = False
        self.capture_thread = None
        self.last_frame = None
        self.last_motion_frame = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.capture_times = deque(maxlen=30)
        
        # Screenshot directory
        if self.save_screenshots:
            self.screenshot_dir = Path("screenshots")
            self.screenshot_dir.mkdir(exist_ok=True)
            
        # Initialize mss
        self.sct = mss.mss()
        
        # Threading lock for thread-safe operations
        self.lock = threading.Lock()
        
        logger.info(f"ContinuousScreenCapture initialized (fps={fps}, buffer={buffer_seconds}s)")
        
    def start(self):
        """Start continuous capture in background thread"""
        if self.is_running:
            logger.warning("Capture already running")
            return
            
        self.is_running = True
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info("Continuous capture started")
        
    def stop(self):
        """Stop continuous capture"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        logger.info(f"Continuous capture stopped. Total frames: {self.frame_count}")
        
    def _capture_loop(self):
        """Main capture loop running in background thread"""
        frame_interval = 1.0 / self.target_fps
        next_capture_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Capture if it's time
                if current_time >= next_capture_time:
                    capture_start = time.time()
                    
                    # Capture screen
                    frame_data = self._capture_screen()
                    
                    if frame_data is not None:
                        # Calculate motion score
                        motion_score = self._calculate_motion(frame_data)
                        
                        # Detect regions of interest
                        roi_regions = self._detect_roi_regions(frame_data, motion_score)
                        
                        # Create frame object
                        frame = Frame(
                            timestamp=current_time,
                            data=frame_data,
                            frame_hash=self._hash_frame(frame_data),
                            motion_score=motion_score,
                            roi_regions=roi_regions
                        )
                        
                        # Add to ring buffer
                        with self.lock:
                            self.ring_buffer.append(frame)
                            self.frame_count += 1
                        
                        # Track performance
                        capture_duration = time.time() - capture_start
                        self.capture_times.append(capture_duration)
                        
                        # Adjust FPS based on GPU load
                        self._adjust_fps_for_gpu()
                        
                    # Calculate next capture time
                    next_capture_time += frame_interval
                    
                    # Skip frames if we're behind
                    if next_capture_time < current_time:
                        skipped = int((current_time - next_capture_time) / frame_interval)
                        if skipped > 0:
                            logger.debug(f"Skipping {skipped} frames to catch up")
                            next_capture_time = current_time + frame_interval
                            
                else:
                    # Sleep briefly to avoid busy waiting
                    sleep_time = min(0.001, next_capture_time - current_time)
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
                
    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture the primary monitor with immediate downscaling"""
        try:
            monitor = self.sct.monitors[1]
            
            # For 5K displays, capture a smaller region or use monitor scaling
            # This reduces memory usage significantly
            if monitor['width'] > 3000:  # Likely a high-res display
                # Option 1: Capture center region only
                capture_width = min(1920, monitor['width'])
                capture_height = min(1080, monitor['height'])
                x_offset = (monitor['width'] - capture_width) // 2
                y_offset = (monitor['height'] - capture_height) // 2
                
                capture_region = {
                    'left': monitor['left'] + x_offset,
                    'top': monitor['top'] + y_offset,
                    'width': capture_width,
                    'height': capture_height
                }
                screenshot = self.sct.grab(capture_region)
            else:
                screenshot = self.sct.grab(monitor)
            
            # Convert to numpy array and resize in one step
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
            
            # Always resize to target size to save memory
            if self.target_size:
                img = img.resize(self.target_size, Image.Resampling.LANCZOS)
                
            frame = np.array(img)
            del img  # Explicitly delete PIL image
            
            # Store for motion detection
            self.last_frame = frame
            
            return frame
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
            
    def _calculate_motion(self, frame: np.ndarray) -> float:
        """Calculate motion score using optical flow"""
        if self.last_motion_frame is None:
            self.last_motion_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return 0.0
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.last_motion_frame, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            # Amplify the motion score since optical flow returns very small values
            motion_score = np.mean(magnitude) * 100.0  # Amplify instead of reducing
            
            # Update reference frame
            self.last_motion_frame = gray
            
            return min(motion_score, 1.0)
            
        except Exception as e:
            logger.debug(f"Motion calculation failed: {e}")
            return 0.0
            
    def _detect_roi_regions(self, frame: np.ndarray, motion_score: float) -> List[Tuple[int, int, int, int]]:
        """Detect regions of interest based on motion and saliency"""
        roi_regions = []
        
        if motion_score > 0.001:  # Detect ROI even with small motion
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Apply edge detection
                edges = cv2.Canny(gray, 50, 150)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter and sort contours by area
                significant_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        significant_contours.append((x, y, w, h, area))
                        
                # Keep top 5 largest regions
                significant_contours.sort(key=lambda x: x[4], reverse=True)
                roi_regions = [(x, y, w, h) for x, y, w, h, _ in significant_contours[:5]]
                
            except Exception as e:
                logger.debug(f"ROI detection failed: {e}")
                
        return roi_regions
        
    def _hash_frame(self, frame: np.ndarray) -> str:
        """Generate hash for frame deduplication"""
        # Downsample for faster hashing
        small = cv2.resize(frame, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()
        
    def _adjust_fps_for_gpu(self):
        """Adjust FPS based on GPU load"""
        try:
            # Check GPU usage (simplified - you might want to use pynvml for NVIDIA GPUs)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if cpu_percent > self.gpu_throttle_threshold * 100:
                # Throttle down
                self.actual_fps = max(5, self.target_fps // 2)
            else:
                # Return to target
                self.actual_fps = self.target_fps
                
        except Exception:
            pass
            
    def get_frame_window(self, start_time: float, end_time: float) -> List[Frame]:
        """
        Get frames within a time window
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of frames within the window
        """
        with self.lock:
            frames = []
            for frame in self.ring_buffer:
                if start_time <= frame.timestamp <= end_time:
                    frames.append(frame)
            return frames
            
    def get_latest_frames(self, count: int = 1) -> List[Frame]:
        """Get the most recent frames"""
        with self.lock:
            if count == 1 and self.ring_buffer:
                return [self.ring_buffer[-1]]
            return list(self.ring_buffer)[-count:]
            
    def extract_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract region of interest from frame
        
        Args:
            frame: Source frame
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Cropped region
        """
        x, y, w, h = bbox
        return frame[y:y+h, x:x+w]
        
    def get_motion_map(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Generate motion heatmap between two frames
        
        Returns:
            Motion heatmap as numpy array
        """
        try:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Convert to magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Normalize to 0-255
            motion_map = np.clip(magnitude * 10, 0, 255).astype(np.uint8)
            
            return motion_map
            
        except Exception as e:
            logger.error(f"Motion map generation failed: {e}")
            return np.zeros_like(frame1[:, :, 0])
            
    def save_frame(self, frame: Frame, prefix: str = "frame"):
        """Save a frame to disk"""
        if not self.save_screenshots:
            return None
            
        timestamp_str = datetime.fromtimestamp(frame.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp_str}_{frame.frame_hash[:8]}.png"
        filepath = self.screenshot_dir / filename
        
        img = Image.fromarray(frame.data)
        img.save(filepath)
        
        return str(filepath)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        if not self.capture_times:
            return {}
            
        current_time = time.time()
        runtime = current_time - self.start_time if self.start_time else 0
        
        return {
            'frame_count': self.frame_count,
            'buffer_size': len(self.ring_buffer),
            'runtime_seconds': runtime,
            'actual_fps': self.frame_count / runtime if runtime > 0 else 0,
            'target_fps': self.target_fps,
            'avg_capture_time': np.mean(self.capture_times) if self.capture_times else 0,
            'buffer_memory_mb': len(self.ring_buffer) * self.target_size[0] * self.target_size[1] * 3 / (1024 * 1024)
        }
        
    async def get_latest_frame_async(self) -> Optional[Frame]:
        """Async wrapper for getting latest frame"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_latest_frames(1)[0] if self.ring_buffer else None)