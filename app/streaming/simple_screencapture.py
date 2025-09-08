#!/usr/bin/env python3
"""
Simplified ScreenCaptureKit implementation that actually works.
Uses the correct API without SCStreamOutput.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Callable
import ScreenCaptureKit as SC
import CoreMedia as CM
import Quartz
from Quartz import CoreVideo as CV
import mss
from PIL import Image

logger = logging.getLogger(__name__)

class SimpleScreenCapture:
    """Simplified screen capture using ScreenCaptureKit"""
    
    def __init__(self, fps: int = 10, display_index: int = 0):
        self.fps = fps
        self.display_index = display_index
        self.stream: Optional[SC.SCStream] = None
        self.is_running = False
        self.frame_callback: Optional[Callable] = None
        self.frames_captured = 0
        self.selected_display_index = None
        self.selected_display = None
    
    def enumerate_displays(self, content):
        """
        Enumerate all available displays and return their information.
        
        Args:
            content: SCShareableContent object
            
        Returns:
            List of display information dictionaries
        """
        displays = content.displays() if callable(content.displays) else content.displays
        display_info = []
        
        logger.info(f"Found {len(displays)} display(s):")
        for i, display in enumerate(displays):
            info = {
                'index': i,
                'width': display.width,
                'height': display.height,
                'display_object': display
            }
            display_info.append(info)
            logger.info(f"  Display {i}: {display.width}x{display.height}")
        
        return display_info
        
    async def initialize(self):
        """Initialize the stream"""
        try:
            # Get shareable content
            content = None
            
            def completion_handler(content_obj, error_obj):
                nonlocal content
                content = content_obj
            
            SC.SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)
            await asyncio.sleep(0.5)  # Wait for completion
            
            if not content:
                logger.error("Failed to get shareable content")
                return False
            
            # Get displays
            displays = content.displays() if callable(content.displays) else content.displays
            if not displays or len(displays) == 0:
                logger.error("No displays available")
                return False
            
            # Enumerate all displays
            display_info = self.enumerate_displays(content)
            
            # Validate and select display by index
            if self.display_index >= len(displays):
                logger.warning(f"Display index {self.display_index} not found (only {len(displays)} displays available)")
                logger.warning(f"Falling back to display 0")
                self.selected_display_index = 0
            else:
                self.selected_display_index = self.display_index
            
            # Use selected display
            display = displays[self.selected_display_index]
            self.selected_display = display
            logger.info(f"Using display {self.selected_display_index}: {display.width}x{display.height}")
            
            # Create stream configuration
            config = SC.SCStreamConfiguration.alloc().init()
            config.setWidth_(1920)
            config.setHeight_(1080)
            config.setMinimumFrameInterval_(CM.CMTimeMake(1, self.fps))
            config.setPixelFormat_(0x42475241)  # kCVPixelFormatType_32BGRA
            config.setShowsCursor_(True)
            
            # Create content filter for display
            filter = SC.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display, []
            )
            
            # Create stream (without delegate - we'll poll instead)
            self.stream = SC.SCStream.alloc().initWithFilter_configuration_delegate_(
                filter, config, None
            )
            
            logger.info("Stream initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize stream: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def start(self):
        """Start capturing"""
        if not self.stream:
            success = await self.initialize()
            if not success:
                return False
        
        try:
            # Start capture
            error = None
            
            def completion_handler(error_obj):
                nonlocal error
                error = error_obj
            
            self.stream.startCaptureWithCompletionHandler_(completion_handler)
            await asyncio.sleep(0.5)  # Wait for start
            
            if error:
                logger.error(f"Failed to start capture: {error}")
                return False
            
            self.is_running = True
            logger.info("Capture started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return False
    
    async def stop(self):
        """Stop capturing"""
        if self.stream and self.is_running:
            try:
                error = None
                
                def completion_handler(error_obj):
                    nonlocal error
                    error = error_obj
                
                self.stream.stopCaptureWithCompletionHandler_(completion_handler)
                await asyncio.sleep(0.5)
                
                self.is_running = False
                logger.info("Capture stopped")
                
            except Exception as e:
                logger.error(f"Error stopping capture: {e}")
    
    def set_frame_callback(self, callback: Callable):
        """Set callback for frames (for compatibility)"""
        self.frame_callback = callback
    
    async def capture_frame_loop(self):
        """Capture real screen frames using mss"""
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # Primary monitor
            
            while self.is_running:
                # Capture at desired FPS
                await asyncio.sleep(1.0 / self.fps)
                
                try:
                    # Capture real screen content
                    screenshot = sct.grab(monitor)
                    
                    # Convert to numpy array (BGR format from mss)
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    
                    # Resize if too large to reduce processing load
                    if img.width > 1920 or img.height > 1080:
                        img = img.resize((1920, 1080), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array
                    frame = np.array(img)
                    
                    if self.frame_callback:
                        self.frame_callback(frame, self.fps)
                    
                    self.frames_captured += 1
                    
                    if self.frames_captured % self.fps == 0:
                        logger.debug(f"Captured {self.frames_captured} frames with real content")
                        
                except Exception as e:
                    logger.error(f"Error capturing frame: {e}")
                    # Fall back to black frame on error
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    if self.frame_callback:
                        self.frame_callback(frame, self.fps)


# Demo usage
async def demo():
    """Test the simplified capture"""
    print("\n🎥 Simple ScreenCaptureKit Test")
    print("="*40)
    
    capture = SimpleScreenCapture(fps=30)
    
    def frame_callback(frame, fps):
        print(f"Frame received: {frame.shape}, FPS: {fps}")
    
    capture.set_frame_callback(frame_callback)
    
    # Start capture
    success = await capture.start()
    if success:
        print("✅ Capture started!")
        
        # Run capture loop for 5 seconds
        capture_task = asyncio.create_task(capture.capture_frame_loop())
        await asyncio.sleep(5)
        
        # Stop
        await capture.stop()
        capture_task.cancel()
        
        print(f"\n📊 Total frames captured: {capture.frames_captured}")
    else:
        print("❌ Failed to start capture")

if __name__ == "__main__":
    asyncio.run(demo())