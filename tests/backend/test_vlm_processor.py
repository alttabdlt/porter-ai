#!/usr/bin/env python3
"""
TDD Test suite for FastVLM processor.
Tests real ML inference capabilities.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.backend.vlm_processor import FastVLMProcessor


class TestFastVLMProcessor:
    """Test suite for FastVLM processor with real ML inference"""
    
    @pytest.fixture
    def processor(self):
        """Create FastVLM processor instance"""
        # Use the local model path
        model_path = "/Users/axel/Desktop/Coding-Projects/porter.ai/ml-fastvlm/models/fastvlm-0.5b-mlx"
        return FastVLMProcessor(model_name=model_path)
    
    @pytest.fixture
    def test_frame(self):
        """Create a test screen frame"""
        # Create a realistic test frame (simulating a code editor)
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 30  # Dark background
        
        # Add some UI elements
        frame[50:100, :] = 50  # Top bar
        frame[:, 200:250] = 40  # Sidebar
        frame[200:800, 300:1600] = 25  # Main content area
        
        # Add some "text" (bright lines)
        for i in range(10):
            y = 250 + i * 30
            frame[y:y+2, 320:1000] = 180  # Simulate text lines
        
        return frame
    
    @pytest.mark.asyncio
    async def test_model_initialization(self, processor):
        """Test that FastVLM model loads correctly"""
        # Initialize the model
        await processor.initialize()
        
        # Check model is loaded
        assert processor.initialized is True
        assert processor.model is not None
        assert processor.processor is not None
        
    @pytest.mark.asyncio
    async def test_basic_description_generation(self, processor, test_frame):
        """Test that model can generate basic descriptions"""
        # Initialize model
        await processor.initialize()
        
        # Generate description
        description = await processor.describe_screen(test_frame)
        
        # Verify we get a real description (not mock)
        assert description is not None
        assert len(description) > 10
        assert description != "No description available"
        assert "Processing screen content..." not in description
        
    @pytest.mark.asyncio
    async def test_description_quality(self, processor):
        """Test that descriptions are meaningful and contextual"""
        await processor.initialize()
        
        # Create a frame with obvious content (terminal)
        terminal_frame = np.zeros((768, 1024, 3), dtype=np.uint8)
        terminal_frame[:] = [10, 10, 10]  # Black background
        terminal_frame[50:100, 100:900] = [0, 255, 0]  # Green text area
        
        description = await processor.describe_screen(
            terminal_frame,
            prompt="What application is visible on screen?"
        )
        
        # Should recognize dark theme or terminal-like interface
        assert len(description) > 20
        # Should not be random pattern matching
        assert not any(keyword in description.lower() for keyword in [
            "morning", "afternoon", "evening", "late night"
        ])
        
    @pytest.mark.asyncio
    async def test_context_tracking(self, processor, test_frame):
        """Test that processor maintains context between frames"""
        await processor.initialize()
        
        # First frame
        desc1 = await processor.describe_screen(test_frame)
        assert processor.last_description == desc1
        
        # Second frame with context
        desc2 = await processor.describe_screen(test_frame)
        # Should have access to previous context
        assert processor.last_description == desc2
        
    @pytest.mark.asyncio
    async def test_custom_prompts(self, processor, test_frame):
        """Test different prompt templates"""
        await processor.initialize()
        
        prompts = [
            "What is the user doing?",
            "Identify any errors on screen",
            "What application is open?",
        ]
        
        descriptions = []
        for prompt in prompts:
            desc = await processor.describe_screen(test_frame, prompt)
            descriptions.append(desc)
            assert desc is not None
            assert len(desc) > 10
        
        # Descriptions should be different for different prompts
        assert len(set(descriptions)) > 1
        
    @pytest.mark.asyncio
    async def test_performance_requirements(self, processor, test_frame):
        """Test that inference meets performance requirements"""
        await processor.initialize()
        
        # Warm up
        await processor.describe_screen(test_frame)
        
        # Measure inference time
        start = time.time()
        description = await processor.describe_screen(test_frame)
        inference_time = (time.time() - start) * 1000  # ms
        
        # Should be fast enough for real-time (< 100ms ideal, < 200ms acceptable)
        assert inference_time < 200, f"Inference too slow: {inference_time:.1f}ms"
        assert description is not None
        
    @pytest.mark.asyncio
    async def test_error_recovery(self, processor):
        """Test graceful handling of errors"""
        await processor.initialize()
        
        # Test with invalid input
        invalid_frame = np.array([])  # Empty array
        description = await processor.describe_screen(invalid_frame)
        
        # Should return fallback, not crash
        assert description is not None
        
        # Test with None
        description = await processor.describe_screen(None)
        assert description is not None
        
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, processor, test_frame):
        """Test that processor doesn't leak memory"""
        await processor.initialize()
        
        # Run multiple inferences
        for _ in range(10):
            await processor.describe_screen(test_frame)
        
        # Should not accumulate temp files
        temp_files = list(Path(".").glob("temp_screen*.jpg"))
        assert len(temp_files) == 0, "Temporary files not cleaned up"
        
    @pytest.mark.asyncio
    async def test_fallback_chain(self, processor):
        """Test fallback from FastVLM to alternatives"""
        # Mock the model loading to fail
        with patch('app.backend.vlm_processor.load', side_effect=Exception("Model load failed")):
            processor = FastVLMProcessor()
            await processor.initialize()
            
            # Should fall back gracefully
            frame = np.ones((100, 100, 3), dtype=np.uint8)
            description = await processor.describe_screen(frame)
            
            assert description is not None
            # Should use mock fallback
            assert processor.model is None
            
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, processor, test_frame):
        """Test handling multiple concurrent description requests"""
        await processor.initialize()
        
        # Create multiple concurrent tasks
        tasks = [
            processor.describe_screen(test_frame, f"Prompt {i}")
            for i in range(5)
        ]
        
        # All should complete without errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 10
            
    def test_model_file_exists(self):
        """Test that the FastVLM model files exist"""
        model_path = Path("/Users/axel/Desktop/Coding-Projects/porter.ai/ml-fastvlm/models/fastvlm-0.5b-mlx")
        
        assert model_path.exists(), f"Model directory not found: {model_path}"
        
        # Check essential files
        assert (model_path / "config.json").exists()
        assert (model_path / "model.safetensors").exists() or \
               (model_path / "model.safetensors.index.json").exists()
        assert (model_path / "tokenizer.json").exists()


class TestSimplifiedVLMProcessor:
    """Test the fallback SimplifiedVLMProcessor"""
    
    @pytest.mark.asyncio
    async def test_fallback_works(self):
        """Test that simplified processor works as fallback"""
        from app.backend.simple_processor import SimplifiedVLMProcessor
        
        processor = SimplifiedVLMProcessor()
        frame = np.ones((768, 1024, 3), dtype=np.uint8) * 100
        
        description = await processor.describe_screen(frame)
        
        assert description is not None
        assert len(description) > 10
        
        # Should be using pattern matching (contains time references)
        # This is what we want to upgrade FROM
        

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])