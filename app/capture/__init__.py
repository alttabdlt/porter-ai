"""
Cross-platform screen capture module for Porter.AI
Provides unified interface for screen capture across Windows, macOS, and Linux.
"""

from .cross_platform_capture import (
    CrossPlatformCapture,
    CaptureConfig,
    CaptureMode
)

__all__ = [
    'CrossPlatformCapture',
    'CaptureConfig',
    'CaptureMode',
    'get_capture_system'
]


def get_capture_system(config=None):
    """
    Get the appropriate capture system for the current platform.
    
    Args:
        config: CaptureConfig object or None for defaults
        
    Returns:
        Capture system instance
    """
    import platform
    import os
    
    system = platform.system()
    
    # Check if we should use native macOS capture
    if system == "Darwin" and os.getenv("USE_NATIVE_CAPTURE", "false").lower() == "true":
        try:
            from app.streaming.simple_screencapture import SimpleScreenCapture
            # Convert config if needed
            if config:
                return SimpleScreenCapture(
                    fps=config.fps,
                    display_index=config.monitor_index - 1  # Convert from 1-based to 0-based
                )
            return SimpleScreenCapture()
        except ImportError:
            pass
    
    # Default to cross-platform capture
    return CrossPlatformCapture(config or CaptureConfig())