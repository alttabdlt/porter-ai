#!/usr/bin/env python3
"""
Base abstract class for all Vision-Language Model processors.
Provides a unified interface for different VLM implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for VLM"""
    FAST = "fast"  # Prioritize speed
    BALANCED = "balanced"  # Balance speed and quality
    QUALITY = "quality"  # Prioritize quality


@dataclass
class VLMConfig:
    """Configuration for VLM processors"""
    model_name: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda, mps
    max_tokens: int = 150
    temperature: float = 0.5
    mode: ProcessingMode = ProcessingMode.BALANCED
    cache_dir: Optional[str] = None
    use_fp16: bool = True
    batch_size: int = 1
    use_context: bool = False  # Whether to use context from previous frames
    context_window: int = 3  # Number of previous descriptions to keep as context


@dataclass
class VLMResult:
    """Result from VLM processing"""
    description: str
    confidence: float = 1.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None


class BaseVLMProcessor(ABC):
    """Abstract base class for Vision-Language Model processors"""
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        Initialize the VLM processor.
        
        Args:
            config: Configuration for the processor
        """
        self.config = config or VLMConfig()
        self.initialized = False
        self.model = None
        self.processor = None
        self.device = self._determine_device()
        
    def _determine_device(self) -> str:
        """Determine the best available device for inference"""
        if self.config.device != "auto":
            return self.config.device
            
        # Try to detect best available device
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("MPS available, using Apple Silicon GPU")
                return "mps"
        except ImportError:
            pass
            
        logger.info("Using CPU for inference")
        return "cpu"
        
    @abstractmethod
    async def initialize(self):
        """
        Initialize the model and processor.
        Must be called before processing.
        """
        pass
        
    @abstractmethod
    async def process_image(
        self, 
        image: np.ndarray, 
        prompt: Optional[str] = None
    ) -> VLMResult:
        """
        Process an image and generate a description.
        
        Args:
            image: numpy array of the image (RGB format)
            prompt: Optional prompt to guide the description
            
        Returns:
            VLMResult with description and metadata
        """
        pass
        
    @abstractmethod
    async def process_batch(
        self, 
        images: List[np.ndarray], 
        prompts: Optional[List[str]] = None
    ) -> List[VLMResult]:
        """
        Process multiple images in batch.
        
        Args:
            images: List of numpy arrays (RGB format)
            prompts: Optional list of prompts (one per image)
            
        Returns:
            List of VLMResults
        """
        pass
        
    async def analyze_risks(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image for potential risks or important events.
        
        Args:
            image: numpy array of the image
            
        Returns:
            Dictionary with risk analysis
        """
        prompt = """Analyze this screen for risks:
        1. Destructive commands or actions
        2. Sensitive data exposure
        3. Security warnings
        4. Potential errors
        
        Rate risk level (low/medium/high) and explain briefly."""
        
        result = await self.process_image(image, prompt)
        
        # Parse result to extract risk level
        risk_level = "low"
        if "high" in result.description.lower():
            risk_level = "high"
        elif "medium" in result.description.lower():
            risk_level = "medium"
            
        return {
            "risk_level": risk_level,
            "description": result.description,
            "confidence": result.confidence
        }
        
    async def suggest_action(
        self, 
        image: np.ndarray, 
        context: Optional[str] = None
    ) -> str:
        """
        Suggest next action based on screen content.
        
        Args:
            image: Current screen capture
            context: Optional context about current task
            
        Returns:
            Action suggestion
        """
        prompt = f"""Suggest the most helpful next action in 10 words or less.
        {f'Context: {context}' if context else ''}
        Be specific and actionable."""
        
        result = await self.process_image(image, prompt)
        # Truncate to ensure brevity
        words = result.description.split()[:10]
        return ' '.join(words)
        
    async def extract_text(self, image: np.ndarray) -> List[str]:
        """
        Extract visible text from the image.
        
        Args:
            image: Image to extract text from
            
        Returns:
            List of extracted text strings
        """
        prompt = "List all visible text in this image. Format as a list."
        result = await self.process_image(image, prompt)
        
        # Parse lines from result
        lines = result.description.split('\n')
        return [line.strip() for line in lines if line.strip()]
        
    async def identify_ui_elements(self, image: np.ndarray) -> Dict[str, List[str]]:
        """
        Identify UI elements in the image.
        
        Args:
            image: Screen capture
            
        Returns:
            Dictionary categorizing UI elements
        """
        prompt = """Identify UI elements in this screen:
        - Buttons
        - Text fields
        - Menus
        - Links
        - Errors/Warnings
        
        Format as categories."""
        
        result = await self.process_image(image, prompt)
        
        # Simple parsing - can be improved
        elements = {
            "buttons": [],
            "text_fields": [],
            "menus": [],
            "links": [],
            "errors": []
        }
        
        # Basic extraction logic
        for line in result.description.lower().split('\n'):
            if 'button' in line:
                elements["buttons"].append(line)
            elif 'field' in line or 'input' in line:
                elements["text_fields"].append(line)
            elif 'menu' in line:
                elements["menus"].append(line)
            elif 'link' in line or 'url' in line:
                elements["links"].append(line)
            elif 'error' in line or 'warning' in line:
                elements["errors"].append(line)
                
        return elements
        
    def cleanup(self):
        """Clean up resources"""
        self.model = None
        self.processor = None
        self.initialized = False
        logger.info(f"{self.__class__.__name__} cleaned up")
        
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()