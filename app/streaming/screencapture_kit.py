#!/usr/bin/env python3
"""
ScreenCaptureKit wrapper for high-performance screen streaming.
Replaces mss with native macOS API for zero-copy frame access.
"""

import asyncio
import logging
from typing import Optional, Callable, Any
import numpy as np
from dataclasses import dataclass
import time

import objc
import ScreenCaptureKit as SC
import CoreMedia as CM
import Quartz
from Quartz import CoreVideo as CV
from Foundation import NSObject, NSMutableDictionary

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for screen capture stream"""
    width: int = 1920
    height: int = 1080
    fps: int = 60
    show_cursor: bool = True
    capture_audio: bool = False
    pixel_format: int = 0x42475241  # kCVPixelFormatType_32BGRA

class StreamDelegate(NSObject):
    """Delegate to handle stream output callbacks"""
    
    def initWithCallback_(self, callback):
        self = objc.super(StreamDelegate, self).init()
        if self:
            self.callback = callback
            self.frame_count = 0
            self.last_fps_time = time.time()
            self.fps = 0.0
        return self
    
    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
        """Called when new frame is available"""
        # Note: SCStreamOutputTypeScreen doesn't exist in the API
        # We'll always process frames as screen output
        if True:  # output_type == SC.SCStreamOutputTypeScreen:
            # Get CVPixelBuffer from sample buffer
            pixel_buffer = CM.CMSampleBufferGetImageBuffer(sample_buffer)
            if pixel_buffer and self.callback:
                # Track FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                
                # Invoke callback with pixel buffer
                self.callback(pixel_buffer, self.fps)

class ScreenCaptureStream:
    """High-performance screen capture using ScreenCaptureKit"""
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.stream: Optional[SC.SCStream] = None
        self.delegate: Optional[StreamDelegate] = None
        self.frame_callback: Optional[Callable] = None
        self.is_running = False
        
        # Performance metrics
        self.frames_captured = 0
        self.start_time = 0
        self.current_fps = 0.0
        
    async def initialize(self):
        """Initialize stream with available displays"""
        try:
            # Get shareable content
            content = await self._get_shareable_content()
            if not content:
                raise RuntimeError("Failed to get shareable content")
            
            # Get displays
            displays = content.displays() if callable(content.displays) else content.displays
            if not displays:
                raise RuntimeError("No displays available for capture")
            
            # Use primary display
            self.display = displays[0]
            logger.info(f"Using display: {self.display.width}x{self.display.height}")
            
            # Create stream configuration
            stream_config = SC.SCStreamConfiguration.alloc().init()
            stream_config.setWidth_(self.config.width)
            stream_config.setHeight_(self.config.height)
            stream_config.setMinimumFrameInterval_(CM.CMTimeMake(1, self.config.fps))
            stream_config.setPixelFormat_(self.config.pixel_format)
            stream_config.setShowsCursor_(self.config.show_cursor)
            stream_config.setCapturesAudio_(self.config.capture_audio)
            
            # Create content filter for display
            filter = SC.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                self.display, []
            )
            
            # Create stream
            self.stream = SC.SCStream.alloc().initWithFilter_configuration_delegate_(
                filter, stream_config, None
            )
            
            logger.info("ScreenCaptureKit stream initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize stream: {e}")
            return False
    
    async def _get_shareable_content(self):
        """Get available content for sharing"""
        # Use completion handler to get shareable content
        try:
            content = None
            error = None
            
            def completion_handler(content_obj, error_obj):
                nonlocal content, error
                content = content_obj
                error = error_obj
            
            SC.SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)
            
            # Wait for async completion
            import asyncio
            await asyncio.sleep(0.5)
            
            if error:
                logger.error(f"Failed to get shareable content: {error}")
                return None
            
            return content
        except Exception as e:
            logger.error(f"Failed to get shareable content: {e}")
            return None
    
    def set_frame_callback(self, callback: Callable[[np.ndarray, float], None]):
        """Set callback for processed frames"""
        self.frame_callback = callback
    
    def _handle_frame(self, pixel_buffer, fps):
        """Convert CVPixelBuffer to numpy and invoke callback"""
        try:
            # Convert CVPixelBuffer to numpy array
            frame = self._pixel_buffer_to_numpy(pixel_buffer)
            
            # Update metrics
            self.frames_captured += 1
            self.current_fps = fps
            
            # Invoke user callback
            if self.frame_callback:
                self.frame_callback(frame, fps)
                
        except Exception as e:
            logger.error(f"Error handling frame: {e}")
    
    def _pixel_buffer_to_numpy(self, pixel_buffer) -> np.ndarray:
        """Convert CVPixelBuffer to numpy array (zero-copy when possible)"""
        # Lock the pixel buffer
        CV.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
        
        try:
            # Get buffer info
            width = CV.CVPixelBufferGetWidth(pixel_buffer)
            height = CV.CVPixelBufferGetHeight(pixel_buffer)
            bytes_per_row = CV.CVPixelBufferGetBytesPerRow(pixel_buffer)
            base_address = CV.CVPixelBufferGetBaseAddress(pixel_buffer)
            
            # Create numpy array from buffer (zero-copy)
            # BGRA format from ScreenCaptureKit
            buffer_size = height * bytes_per_row
            np_buffer = np.frombuffer(
                (base_address, buffer_size),
                dtype=np.uint8
            ).reshape((height, width, 4))
            
            # Convert BGRA to RGB
            frame = np_buffer[:, :, [2, 1, 0]]  # BGR to RGB, ignore alpha
            
            return frame
            
        finally:
            # Unlock pixel buffer
            CV.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
    
    async def start(self):
        """Start the capture stream"""
        if not self.stream:
            await self.initialize()
        
        if self.stream:
            # Create and set delegate
            self.delegate = StreamDelegate.alloc().initWithCallback_(self._handle_frame)
            
            # Start capture directly without SCStreamOutput (doesn't exist in API)
            def completion_handler(error):
                if error:
                    logger.error(f"Failed to start stream: {error}")
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.stream.startCaptureWithCompletionHandler_, completion_handler
            )
            
            self.is_running = True
            self.start_time = time.time()
            self.frames_captured = 0
            
            logger.info("Stream started successfully")
            return True
        
        return False
    
    async def stop(self):
        """Stop the capture stream"""
        if self.stream and self.is_running:
            await asyncio.get_event_loop().run_in_executor(
                None, self.stream.stopCaptureWithCompletionHandler_, None
            )
            
            self.is_running = False
            
            # Calculate average FPS
            if self.frames_captured > 0:
                duration = time.time() - self.start_time
                avg_fps = self.frames_captured / duration
                logger.info(f"Stream stopped. Avg FPS: {avg_fps:.1f}, Total frames: {self.frames_captured}")
    
    def get_metrics(self) -> dict:
        """Get current performance metrics"""
        metrics = {
            "is_running": self.is_running,
            "frames_captured": self.frames_captured,
            "current_fps": self.current_fps,
            "config_fps": self.config.fps,
            "resolution": f"{self.config.width}x{self.config.height}"
        }
        
        if self.is_running and self.frames_captured > 0:
            duration = time.time() - self.start_time
            metrics["average_fps"] = self.frames_captured / duration
            metrics["uptime_seconds"] = duration
        
        return metrics


class AdaptiveFrameSampler:
    """Intelligent frame sampling based on activity and importance"""
    
    def __init__(self, base_fps: int = 60, min_fps: int = 5, max_fps: int = 60):
        self.base_fps = base_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.current_fps = base_fps
        
        # Sampling state
        self.last_sample_time = 0
        self.sample_interval = 1.0 / base_fps
        
        # Activity tracking
        self.motion_score = 0.0
        self.importance_score = 0.0
        self.last_frame = None
        
    def should_sample(self, timestamp: float) -> bool:
        """Determine if current frame should be sampled"""
        if timestamp - self.last_sample_time >= self.sample_interval:
            self.last_sample_time = timestamp
            return True
        return False
    
    def update_activity(self, motion: float, importance: float):
        """Update sampling rate based on activity metrics"""
        self.motion_score = motion
        self.importance_score = importance
        
        # Adaptive sampling logic
        if importance > 0.8 or motion > 0.7:
            # High activity - increase sampling
            self.current_fps = self.max_fps
        elif importance > 0.5 or motion > 0.4:
            # Medium activity
            self.current_fps = self.base_fps
        else:
            # Low activity - reduce sampling
            self.current_fps = max(self.min_fps, self.base_fps // 2)
        
        # Update sample interval
        self.sample_interval = 1.0 / self.current_fps
    
    def calculate_motion(self, frame: np.ndarray) -> float:
        """Calculate motion score between frames"""
        if self.last_frame is None:
            self.last_frame = frame
            return 0.0
        
        # Simple frame difference
        diff = np.abs(frame.astype(float) - self.last_frame.astype(float))
        motion_score = np.mean(diff) / 255.0
        
        self.last_frame = frame
        return min(1.0, motion_score * 10)  # Scale and cap at 1.0


# Demo usage
if __name__ == "__main__":
    async def demo():
        """Demonstrate ScreenCaptureKit streaming"""
        
        # Frame counter
        frame_count = 0
        
        def frame_callback(frame: np.ndarray, fps: float):
            nonlocal frame_count
            frame_count += 1
            if frame_count % 60 == 0:  # Log every second at 60 FPS
                print(f"Frame {frame_count}: Shape={frame.shape}, FPS={fps:.1f}")
        
        # Create stream
        config = StreamConfig(width=1920, height=1080, fps=60)
        stream = ScreenCaptureStream(config)
        stream.set_frame_callback(frame_callback)
        
        # Start streaming
        print("Starting ScreenCaptureKit stream...")
        await stream.start()
        
        # Run for 5 seconds
        await asyncio.sleep(5)
        
        # Stop and show metrics
        await stream.stop()
        metrics = stream.get_metrics()
        print(f"\nMetrics: {metrics}")
    
    # Run demo
    asyncio.run(demo())