#!/usr/bin/env python3
"""
Cross-platform screen capture module using mss.
Works on Windows, macOS, and Linux without platform-specific dependencies.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import mss
import mss.tools
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CaptureMode(Enum):
    """Screen capture modes"""
    FULL_SCREEN = "full_screen"
    PRIMARY_MONITOR = "primary_monitor"
    ALL_MONITORS = "all_monitors"
    REGION = "region"
    WINDOW = "window"


@dataclass
class CaptureConfig:
    """Configuration for screen capture"""
    mode: CaptureMode = CaptureMode.PRIMARY_MONITOR
    fps: int = 10
    width: Optional[int] = 1920
    height: Optional[int] = 1080
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    monitor_index: int = 1  # 0 = all, 1+ = specific monitor
    resize: bool = True
    quality: int = 95  # For JPEG compression if needed


class CrossPlatformCapture:
    """
    Cross-platform screen capture using mss.
    
    Features:
    - Works on Windows, macOS, and Linux
    - No platform-specific dependencies
    - Efficient memory usage
    - Configurable capture modes
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """Initialize the capture system"""
        self.config = config or CaptureConfig()
        self.is_running = False
        self.frame_callback: Optional[Callable] = None
        self.frames_captured = 0
        self.start_time = 0
        self._sct: Optional[mss.mss] = None
        self.monitors_info: List[Dict] = []
        
        logger.info(f"CrossPlatformCapture initialized with mode: {self.config.mode.value}")
        
    def get_monitors_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all available monitors.
        
        Returns:
            List of monitor information dictionaries
        """
        with mss.mss() as sct:
            monitors = []
            for i, monitor in enumerate(sct.monitors):
                monitors.append({
                    "index": i,
                    "left": monitor.get("left", 0),
                    "top": monitor.get("top", 0),
                    "width": monitor.get("width", 0),
                    "height": monitor.get("height", 0),
                    "is_primary": i == 1  # mss convention: monitor 1 is primary
                })
                
            self.monitors_info = monitors
            logger.info(f"Found {len(monitors)-1} monitor(s)")  # -1 because index 0 is all monitors
            
        return monitors[1:]  # Return without the "all monitors" entry
        
    async def initialize(self) -> bool:
        """
        Initialize the capture system.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get monitor information
            self.get_monitors_info()
            
            # Validate configuration
            if self.config.mode == CaptureMode.REGION and not self.config.region:
                logger.error("Region mode requires region coordinates")
                return False
                
            logger.info("Screen capture initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize screen capture: {e}")
            return False
            
    def set_frame_callback(self, callback: Callable):
        """Set callback function for captured frames"""
        self.frame_callback = callback
        
    async def start_capture(self):
        """Start the capture loop"""
        if self.is_running:
            logger.warning("Capture already running")
            return
            
        self.is_running = True
        self.start_time = time.time()
        self.frames_captured = 0
        
        # Start capture loop
        asyncio.create_task(self._capture_loop())
        logger.info(f"Started screen capture at {self.config.fps} FPS")
        
    async def stop_capture(self):
        """Stop the capture loop"""
        self.is_running = False
        await asyncio.sleep(0.1)  # Allow loop to finish
        
        if self.frames_captured > 0:
            elapsed = time.time() - self.start_time
            actual_fps = self.frames_captured / elapsed
            logger.info(f"Stopped capture. Captured {self.frames_captured} frames at {actual_fps:.2f} FPS")
            
    async def _capture_loop(self):
        """Main capture loop"""
        frame_interval = 1.0 / self.config.fps
        
        with mss.mss() as sct:
            self._sct = sct
            
            while self.is_running:
                loop_start = time.time()
                
                try:
                    # Capture frame
                    frame = await self._capture_frame(sct)
                    
                    if frame is not None and self.frame_callback:
                        # Call callback with frame
                        if asyncio.iscoroutinefunction(self.frame_callback):
                            await self.frame_callback(frame, self.frames_captured)
                        else:
                            self.frame_callback(frame, self.frames_captured)
                            
                    self.frames_captured += 1
                    
                    # Log periodically
                    if self.frames_captured % (self.config.fps * 10) == 0:
                        elapsed = time.time() - self.start_time
                        actual_fps = self.frames_captured / elapsed
                        logger.debug(f"Captured {self.frames_captured} frames, actual FPS: {actual_fps:.2f}")
                        
                except Exception as e:
                    logger.error(f"Error in capture loop: {e}")
                    
                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        self._sct = None
        
    async def _capture_frame(self, sct: mss.mss) -> Optional[np.ndarray]:
        """
        Capture a single frame based on configured mode.
        
        Args:
            sct: mss screen capture object
            
        Returns:
            numpy array of captured frame or None
        """
        try:
            # Determine capture region based on mode
            if self.config.mode == CaptureMode.FULL_SCREEN:
                # All monitors combined
                monitor = sct.monitors[0]
                
            elif self.config.mode == CaptureMode.PRIMARY_MONITOR:
                # Primary monitor (usually index 1 in mss)
                monitor = sct.monitors[self.config.monitor_index]
                
            elif self.config.mode == CaptureMode.ALL_MONITORS:
                # Capture all monitors and combine
                return await self._capture_all_monitors(sct)
                
            elif self.config.mode == CaptureMode.REGION:
                # Custom region
                x, y, w, h = self.config.region
                monitor = {"left": x, "top": y, "width": w, "height": h}
                
            else:
                # Default to primary monitor
                monitor = sct.monitors[1]
                
            # Capture the screen
            screenshot = sct.grab(monitor)
            
            # Convert to numpy array (RGB format)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Resize if configured
            if self.config.resize and (self.config.width or self.config.height):
                target_width = self.config.width or img.width
                target_height = self.config.height or img.height
                
                # Maintain aspect ratio
                if img.width != target_width or img.height != target_height:
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    
            # Convert to numpy array
            frame = np.array(img)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
            
    async def _capture_all_monitors(self, sct: mss.mss) -> Optional[np.ndarray]:
        """
        Capture all monitors and combine them.
        
        Args:
            sct: mss screen capture object
            
        Returns:
            Combined frame from all monitors
        """
        try:
            frames = []
            
            # Capture each monitor (skip index 0 which is all monitors)
            for i in range(1, len(sct.monitors)):
                screenshot = sct.grab(sct.monitors[i])
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                frames.append(np.array(img))
                
            if not frames:
                return None
                
            # Combine frames horizontally
            combined = np.concatenate(frames, axis=1)
            
            # Resize if needed
            if self.config.resize and (self.config.width or self.config.height):
                img = Image.fromarray(combined)
                target_width = self.config.width or img.width
                target_height = self.config.height or img.height
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                combined = np.array(img)
                
            return combined
            
        except Exception as e:
            logger.error(f"Error capturing all monitors: {e}")
            return None
            
    async def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame without starting the loop.
        
        Returns:
            numpy array of captured frame
        """
        with mss.mss() as sct:
            return await self._capture_frame(sct)
            
    async def capture_region(
        self, 
        x: int, 
        y: int, 
        width: int, 
        height: int
    ) -> Optional[np.ndarray]:
        """
        Capture a specific region of the screen.
        
        Args:
            x, y: Top-left coordinates
            width, height: Size of region
            
        Returns:
            numpy array of captured region
        """
        with mss.mss() as sct:
            monitor = {"left": x, "top": y, "width": width, "height": height}
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return np.array(img)
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get capture statistics.
        
        Returns:
            Dictionary with capture statistics
        """
        if self.start_time == 0:
            return {
                "frames_captured": 0,
                "actual_fps": 0,
                "uptime": 0,
                "is_running": self.is_running
            }
            
        elapsed = time.time() - self.start_time
        actual_fps = self.frames_captured / elapsed if elapsed > 0 else 0
        
        return {
            "frames_captured": self.frames_captured,
            "actual_fps": actual_fps,
            "target_fps": self.config.fps,
            "uptime": elapsed,
            "is_running": self.is_running,
            "mode": self.config.mode.value
        }


# Test function
async def test_capture():
    """Test the cross-platform capture"""
    print("\n🖥️ Cross-Platform Screen Capture Test")
    print("=" * 40)
    
    # Create capture with config
    config = CaptureConfig(
        mode=CaptureMode.PRIMARY_MONITOR,
        fps=5,
        width=1280,
        height=720
    )
    
    capture = CrossPlatformCapture(config)
    
    # Get monitor info
    monitors = capture.get_monitors_info()
    print(f"\nFound {len(monitors)} monitor(s):")
    for monitor in monitors:
        print(f"  Monitor {monitor['index']}: {monitor['width']}x{monitor['height']}")
        
    # Initialize
    if not await capture.initialize():
        print("Failed to initialize capture")
        return
        
    # Capture single frame
    print("\nCapturing single frame...")
    frame = await capture.capture_single_frame()
    
    if frame is not None:
        print(f"✅ Captured frame: {frame.shape}")
        print(f"   Min value: {frame.min()}")
        print(f"   Max value: {frame.max()}")
        print(f"   Mean value: {frame.mean():.2f}")
    else:
        print("❌ Failed to capture frame")
        
    # Test continuous capture
    print("\nTesting continuous capture for 3 seconds...")
    
    frames_received = []
    
    def frame_callback(frame, frame_num):
        frames_received.append(frame_num)
        if frame_num % 5 == 0:
            print(f"  Received frame {frame_num}: {frame.shape}")
            
    capture.set_frame_callback(frame_callback)
    await capture.start_capture()
    
    # Run for 3 seconds
    await asyncio.sleep(3)
    
    await capture.stop_capture()
    
    # Get statistics
    stats = capture.get_statistics()
    print(f"\nCapture Statistics:")
    print(f"  Frames captured: {stats['frames_captured']}")
    print(f"  Actual FPS: {stats['actual_fps']:.2f}")
    print(f"  Target FPS: {stats['target_fps']}")
    
    print("\n✅ Cross-platform capture test complete!")


if __name__ == "__main__":
    asyncio.run(test_capture())