#!/usr/bin/env python3
"""
Scene-Aware VLM Processor that integrates with the memory system.
Handles scene changes to prevent context bleeding.
"""

import asyncio
import logging
import time
from typing import Optional, List, Dict, Any
from collections import deque
import numpy as np
from PIL import Image

from .omnivlm_processor import OmniVLMProcessor
from .base_processor import VLMConfig, VLMResult

logger = logging.getLogger(__name__)


class SceneAwareVLMProcessor(OmniVLMProcessor):
    """
    VLM Processor with scene change awareness.
    
    Key features:
    - Clears context on scene changes
    - Maintains context within scenes
    - Integrates with DualMemorySystem
    - Prevents context bleeding between different applications/contexts
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """Initialize scene-aware VLM processor"""
        super().__init__(config)
        
        # Scene-aware context management
        self.context_history = deque(maxlen=config.context_window if config else 3)
        self.last_description = None
        self.current_scene_id = None
        self.memory_system = None
        
        # Scene tracking
        self.last_app = None
        self.scene_start_time = None
        self.frames_in_scene = 0
        
        logger.info("Initialized SceneAwareVLMProcessor with context window: " 
                   f"{self.config.context_window}")
    
    def set_memory_system(self, memory_system):
        """
        Connect to a memory system for scene change detection.
        
        Args:
            memory_system: DualMemorySystem instance
        """
        self.memory_system = memory_system
        logger.info("Connected to memory system")
    
    def clear_context(self):
        """Clear all context on scene change"""
        self.context_history.clear()
        self.last_description = None
        self.frames_in_scene = 0
        logger.debug("Context cleared due to scene change")
    
    def add_to_context(self, description: str):
        """
        Add description to context history.
        
        Args:
            description: Description to add
        """
        self.context_history.append(description)
        self.last_description = description
    
    def on_scene_change(self, change_info: Dict[str, Any]):
        """
        Handle scene change signal.
        
        Args:
            change_info: Scene change information from detector
        """
        if change_info.get('scene_changed', False):
            logger.info(f"Scene change detected: {change_info.get('change_type')} "
                       f"(confidence={change_info.get('confidence', 0):.2f})")
            
            # Clear context to prevent bleeding
            self.clear_context()
            
            # Update scene tracking
            self.current_scene_id = f"scene_{time.time()}"
            self.scene_start_time = time.time()
            
            # Store previous app for reference
            if 'prev_app' in change_info:
                logger.info(f"App transition: {change_info['prev_app']} → "
                           f"{change_info.get('details', {}).get('new_app', 'Unknown')}")
    
    async def process_image(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VLMResult:
        """
        Process an image with scene awareness.
        
        Args:
            image: numpy array of the image (RGB)
            prompt: Optional prompt to guide the description
            metadata: Optional metadata including scene change signals
            
        Returns:
            VLMResult with description and metadata
        """
        # Check for scene change signal
        if metadata and metadata.get('scene_changed', False):
            self.on_scene_change(metadata)
        
        # Update current app if provided
        if metadata and 'active_app' in metadata:
            current_app = metadata['active_app']
            if self.last_app and current_app != self.last_app:
                # App change detected
                logger.debug(f"App change: {self.last_app} → {current_app}")
                self.clear_context()
            self.last_app = current_app
        
        # Build context-aware prompt if context is enabled
        if self.config.use_context and not metadata.get('scene_changed', False):
            prompt = self._build_contextual_prompt(prompt)
        
        # Process with base class
        result = await super().process_image(image, prompt)
        
        # Add to context if successful and no scene change
        if result.confidence > 0 and not metadata.get('scene_changed', False):
            self.add_to_context(result.description)
            self.frames_in_scene += 1
        
        # Add scene-aware metadata
        result.metadata.update({
            'scene_aware': True,
            'scene_id': self.current_scene_id,
            'frames_in_scene': self.frames_in_scene,
            'context_size': len(self.context_history)
        })
        
        return result
    
    def _build_contextual_prompt(self, prompt: Optional[str] = None) -> str:
        """
        Build a prompt that includes relevant context.
        
        Args:
            prompt: Base prompt
            
        Returns:
            Context-enhanced prompt
        """
        if prompt is None:
            prompt = "Describe what you see in this image."
        
        # Add context if available
        if self.context_history and len(self.context_history) > 0:
            # Get recent context (last 2-3 descriptions)
            recent_context = list(self.context_history)[-2:]
            
            if recent_context:
                context_str = " ".join(recent_context)
                # Limit context length to avoid overwhelming the model
                if len(context_str) > 200:
                    context_str = context_str[-200:]
                
                prompt = f"Previous context: {context_str}\n\nCurrent: {prompt}"
                logger.debug(f"Added context to prompt: {len(context_str)} chars")
        
        return prompt
    
    async def process_with_memory(
        self,
        frame: np.ndarray,
        metadata: Dict[str, Any],
        prompt: Optional[str] = None
    ) -> VLMResult:
        """
        Process frame with memory system integration.
        
        Args:
            frame: Current frame
            metadata: Frame metadata including scene info
            prompt: Optional prompt
            
        Returns:
            VLMResult with scene-aware processing
        """
        # If connected to memory system, use its scene detection
        if self.memory_system:
            # Memory system has already detected scene changes
            scene_info = metadata
            
            # Check for scene change from memory system
            if scene_info.get('scene_changed', False):
                self.on_scene_change(scene_info)
        
        # Process with scene awareness
        result = await self.process_image(frame, prompt, metadata)
        
        # If memory system is connected, we can query relevant memories
        if self.memory_system and self.config.use_context:
            try:
                # Get relevant memories for context
                memories = await self.memory_system.get_relevant_memories(
                    query=prompt or "current activity",
                    metadata=metadata,
                    n_results=3
                )
                
                if memories:
                    # Add memory context to result metadata
                    result.metadata['relevant_memories'] = len(memories)
                    logger.debug(f"Found {len(memories)} relevant memories")
                    
            except Exception as e:
                logger.error(f"Failed to query memories: {e}")
        
        return result
    
    async def process_batch_with_scenes(
        self,
        images: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
        prompts: Optional[List[str]] = None
    ) -> List[VLMResult]:
        """
        Process multiple images with scene change detection.
        
        Args:
            images: List of images
            metadata_list: List of metadata for each image
            prompts: Optional list of prompts
            
        Returns:
            List of VLMResults
        """
        if len(images) != len(metadata_list):
            raise ValueError("Number of images must match metadata")
        
        results = []
        
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            prompt = prompts[i] if prompts else None
            
            # Process each frame with scene awareness
            result = await self.process_image(image, prompt, metadata)
            results.append(result)
            
            # Small delay between frames to simulate real-time processing
            if i < len(images) - 1:
                await asyncio.sleep(0.01)
        
        return results
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the current context.
        
        Returns:
            Summary string
        """
        if not self.context_history:
            return "No context available"
        
        summary = f"Scene: {self.current_scene_id or 'unknown'}\n"
        summary += f"App: {self.last_app or 'unknown'}\n"
        summary += f"Frames in scene: {self.frames_in_scene}\n"
        summary += f"Context entries: {len(self.context_history)}\n"
        
        if self.context_history:
            summary += f"Latest: {self.last_description[:100]}..."
        
        return summary
    
    def reset(self):
        """Reset the processor state"""
        self.clear_context()
        self.current_scene_id = None
        self.last_app = None
        self.scene_start_time = None
        self.frames_in_scene = 0
        logger.info("SceneAwareVLMProcessor reset")


class ContextManager:
    """
    Manages context for VLM processing across scenes.
    """
    
    def __init__(self, max_context_length: int = 500):
        """
        Initialize context manager.
        
        Args:
            max_context_length: Maximum character length for context
        """
        self.max_context_length = max_context_length
        self.contexts: Dict[str, deque] = {}  # Scene ID -> context
        
    def add_context(self, scene_id: str, description: str):
        """Add context for a scene"""
        if scene_id not in self.contexts:
            self.contexts[scene_id] = deque(maxlen=5)
        
        self.contexts[scene_id].append(description)
    
    def get_context(self, scene_id: str) -> Optional[str]:
        """Get context for a scene"""
        if scene_id not in self.contexts:
            return None
        
        context_list = list(self.contexts[scene_id])
        if not context_list:
            return None
        
        # Join and truncate if needed
        context = " ".join(context_list)
        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]
        
        return context
    
    def clear_scene(self, scene_id: str):
        """Clear context for a specific scene"""
        if scene_id in self.contexts:
            del self.contexts[scene_id]
    
    def clear_all(self):
        """Clear all contexts"""
        self.contexts.clear()


async def create_scene_aware_processor(
    config: Optional[VLMConfig] = None,
    memory_system: Optional[Any] = None
) -> SceneAwareVLMProcessor:
    """
    Create and initialize a scene-aware VLM processor.
    
    Args:
        config: VLM configuration
        memory_system: Optional memory system to connect
        
    Returns:
        Initialized SceneAwareVLMProcessor
    """
    processor = SceneAwareVLMProcessor(config)
    
    # Initialize the model
    await processor.initialize()
    
    # Connect memory system if provided
    if memory_system:
        processor.set_memory_system(memory_system)
    
    return processor


# Example usage
async def test_scene_aware_vlm():
    """Test the scene-aware VLM processor"""
    logger.info("Testing Scene-Aware VLM Processor...")
    
    # Create processor
    config = VLMConfig(use_context=True, context_window=3)
    processor = await create_scene_aware_processor(config)
    
    # Simulate frames with scene changes
    frames = [
        (np.zeros((720, 1280, 3), dtype=np.uint8), 
         {'active_app': 'VSCode', 'scene_changed': False}),
        (np.zeros((720, 1280, 3), dtype=np.uint8), 
         {'active_app': 'VSCode', 'scene_changed': False}),
        (np.ones((720, 1280, 3), dtype=np.uint8) * 255, 
         {'active_app': 'Chrome', 'scene_changed': True}),
        (np.ones((720, 1280, 3), dtype=np.uint8) * 255, 
         {'active_app': 'Chrome', 'scene_changed': False}),
    ]
    
    for i, (frame, metadata) in enumerate(frames):
        result = await processor.process_image(
            frame,
            prompt="What is happening?",
            metadata=metadata
        )
        logger.info(f"Frame {i}: {result.description[:50]}...")
        logger.info(f"  Context size: {result.metadata.get('context_size', 0)}")
    
    # Get context summary
    summary = processor.get_context_summary()
    logger.info(f"Context Summary:\n{summary}")
    
    logger.info("✅ Scene-Aware VLM test complete!")


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_scene_aware_vlm())