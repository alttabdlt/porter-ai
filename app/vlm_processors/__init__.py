"""
Vision-Language Model Processors for Porter.AI
Cross-platform VLM implementations with automatic backend selection.
"""

from .base_processor import (
    BaseVLMProcessor,
    VLMConfig,
    VLMResult,
    ProcessingMode
)

# Import available processors
__all__ = [
    'BaseVLMProcessor',
    'VLMConfig', 
    'VLMResult',
    'ProcessingMode',
    'get_best_processor'
]

# Try to import processors
try:
    from .omnivlm_processor import OmniVLMProcessor
    __all__.append('OmniVLMProcessor')
except ImportError:
    OmniVLMProcessor = None

# Platform-specific imports
import platform
if platform.system() == "Darwin":
    try:
        from app.backend.vlm_processor import FastVLMProcessor
        __all__.append('FastVLMProcessor')
    except ImportError:
        FastVLMProcessor = None
else:
    FastVLMProcessor = None


def get_best_processor(prefer_fast=False):
    """
    Get the best available VLM processor for the current platform.
    
    Args:
        prefer_fast: Prefer faster models over more accurate ones
        
    Returns:
        VLM processor class
    """
    import os
    
    # Check environment preferences
    if os.getenv("USE_SIMPLE_PROCESSOR", "false").lower() == "true":
        from app.backend.simple_processor import SimplifiedVLMProcessor
        return SimplifiedVLMProcessor
    
    # macOS with FastVLM preference
    if platform.system() == "Darwin" and FastVLMProcessor:
        if os.getenv("USE_FASTVLM", "false").lower() == "true":
            return FastVLMProcessor
    
    # OmniVLM (cross-platform)
    if OmniVLMProcessor:
        return OmniVLMProcessor
    
    # Fallback to simple processor
    try:
        from app.backend.simple_processor import SimplifiedVLMProcessor
        return SimplifiedVLMProcessor
    except ImportError:
        from app.backend.vlm_processor import SimplifiedVLMProcessor
        return SimplifiedVLMProcessor