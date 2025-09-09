#!/usr/bin/env python3
"""
Test suite for OmniVLM processor using TDD approach.
Tests are written first, then implementation follows.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import time

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.vlm_processors.base_processor import VLMConfig, ProcessingMode, VLMResult
from app.vlm_processors.omnivlm_processor import OmniVLMProcessor


class TestOmniVLMProcessor:
    """Test suite for OmniVLM processor"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return VLMConfig(
            model_name="NexaAI/OmniVLM-968M",
            device="cpu",  # Use CPU for testing
            max_tokens=50,
            temperature=0.5,
            mode=ProcessingMode.FAST,
            use_fp16=False  # Disable for CPU testing
        )
    
    @pytest.fixture
    def processor(self, config):
        """Create processor instance"""
        return OmniVLMProcessor(config)
    
    @pytest.fixture
    def test_image(self):
        """Create test image array"""
        # Create a simple test pattern
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some patterns for testing
        image[100:200, 100:300] = [255, 0, 0]  # Red rectangle
        image[250:350, 350:500] = [0, 255, 0]  # Green rectangle
        return image
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing without loading real model"""
        mock = Mock()
        mock.generate = Mock(return_value=[1, 2, 3, 4, 5])  # Mock token IDs
        mock.eval = Mock(return_value=None)
        mock.parameters = Mock(return_value=[Mock(device='cpu')])
        return mock
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_processor_initialization(self, config):
        """Test processor initializes with correct configuration"""
        processor = OmniVLMProcessor(config)
        
        assert processor.config == config
        assert processor.config.model_name == "NexaAI/OmniVLM-968M"
        assert processor.device == "cpu"
        assert not processor.initialized
        assert processor.model is None
        assert processor.processor is None
    
    def test_device_detection_auto(self):
        """Test automatic device detection"""
        config = VLMConfig(device="auto")
        processor = OmniVLMProcessor(config)
        
        # Should default to CPU in test environment
        assert processor.device in ["cpu", "cuda", "mps"]
    
    @pytest.mark.asyncio
    async def test_initialization_lazy_loading(self, processor):
        """Test that model is not loaded until initialize is called"""
        assert processor.model is None
        assert processor.processor is None
        assert not processor.initialized
    
    @pytest.mark.asyncio
    async def test_initialization_with_mock_model(self, processor, mock_model):
        """Test initialization with mocked model"""
        with patch('app.vlm_processors.omnivlm_processor.AutoModelForVision2Seq') as mock_auto:
            with patch('app.vlm_processors.omnivlm_processor.AutoProcessor') as mock_proc:
                with patch('app.vlm_processors.omnivlm_processor.AutoTokenizer') as mock_tok:
                    mock_auto.from_pretrained.return_value = mock_model
                    mock_proc.from_pretrained.return_value = Mock()
                    mock_tok.from_pretrained.return_value = Mock(
                        pad_token_id=0,
                        eos_token_id=1
                    )
                    
                    await processor.initialize()
                    
                    assert processor.initialized
                    assert processor.model is not None
    
    # ==================== IMAGE PROCESSING TESTS ====================
    
    @pytest.mark.asyncio
    async def test_process_image_without_initialization(self, processor, test_image):
        """Test processing image initializes model if needed"""
        with patch.object(processor, '_load_model') as mock_load:
            with patch.object(processor, '_generate') as mock_gen:
                mock_gen.return_value = "Test description"
                
                result = await processor.process_image(test_image)
                
                assert mock_load.called
                assert isinstance(result, VLMResult)
    
    @pytest.mark.asyncio
    async def test_process_image_with_numpy_array(self, processor, test_image):
        """Test processing numpy array image"""
        with patch.object(processor, '_generate') as mock_gen:
            mock_gen.return_value = "A screen with red and green rectangles"
            processor.initialized = True
            processor.model = Mock()
            processor.processor = Mock()
            
            result = await processor.process_image(test_image)
            
            assert isinstance(result, VLMResult)
            assert result.description == "A screen with red and green rectangles"
            assert result.confidence > 0
            assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_process_image_with_pil_image(self, processor):
        """Test processing PIL Image"""
        pil_image = Image.new('RGB', (640, 480), color='red')
        
        with patch.object(processor, '_generate') as mock_gen:
            mock_gen.return_value = "A red image"
            processor.initialized = True
            processor.model = Mock()
            processor.processor = Mock()
            
            result = await processor.process_image(pil_image)
            
            assert result.description == "A red image"
    
    @pytest.mark.asyncio
    async def test_process_image_with_custom_prompt(self, processor, test_image):
        """Test processing with custom prompt"""
        prompt = "Count the colored rectangles"
        
        with patch.object(processor, '_generate') as mock_gen:
            mock_gen.return_value = "There are 2 colored rectangles"
            processor.initialized = True
            processor.model = Mock()
            processor.processor = Mock()
            
            result = await processor.process_image(test_image, prompt)
            
            mock_gen.assert_called_once()
            call_args = mock_gen.call_args[0]
            assert prompt in str(call_args)
    
    @pytest.mark.asyncio
    async def test_process_image_fallback_on_error(self, processor, test_image):
        """Test fallback to mock result when model fails"""
        processor.model = None  # No model loaded
        
        result = await processor.process_image(test_image)
        
        assert isinstance(result, VLMResult)
        assert result.confidence == 0.5  # Mock confidence
        assert "mock" in result.metadata.get("mode", "")
    
    # ==================== BATCH PROCESSING TESTS ====================
    
    @pytest.mark.asyncio
    async def test_process_batch_single_image(self, processor, test_image):
        """Test batch processing with single image"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult("Test", 0.9, 0.1)
            
            results = await processor.process_batch([test_image])
            
            assert len(results) == 1
            assert results[0].description == "Test"
    
    @pytest.mark.asyncio
    async def test_process_batch_multiple_images(self, processor):
        """Test batch processing with multiple images"""
        images = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 255,
            np.ones((480, 640, 3), dtype=np.uint8) * 128
        ]
        
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.side_effect = [
                VLMResult("Black image", 0.9, 0.1),
                VLMResult("White image", 0.9, 0.1),
                VLMResult("Gray image", 0.9, 0.1)
            ]
            
            results = await processor.process_batch(images)
            
            assert len(results) == 3
            assert results[0].description == "Black image"
            assert results[1].description == "White image"
            assert results[2].description == "Gray image"
    
    @pytest.mark.asyncio
    async def test_process_batch_with_prompts(self, processor):
        """Test batch processing with custom prompts"""
        images = [np.zeros((480, 640, 3), dtype=np.uint8)] * 2
        prompts = ["Describe colors", "Count objects"]
        
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.side_effect = [
                VLMResult("All black", 0.9, 0.1),
                VLMResult("No objects", 0.9, 0.1)
            ]
            
            results = await processor.process_batch(images, prompts)
            
            assert len(results) == 2
            # Verify prompts were passed correctly
            calls = mock_process.call_args_list
            assert prompts[0] in str(calls[0])
            assert prompts[1] in str(calls[1])
    
    @pytest.mark.asyncio
    async def test_process_batch_mismatched_prompts_raises_error(self, processor):
        """Test that mismatched prompts and images raises error"""
        images = [np.zeros((480, 640, 3), dtype=np.uint8)] * 2
        prompts = ["Only one prompt"]
        
        with pytest.raises(ValueError, match="Number of prompts must match"):
            await processor.process_batch(images, prompts)
    
    # ==================== RISK ANALYSIS TESTS ====================
    
    @pytest.mark.asyncio
    async def test_analyze_risks_high_risk(self, processor, test_image):
        """Test risk analysis detecting high risk"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult(
                "High risk: rm -rf command detected", 0.9, 0.1
            )
            
            result = await processor.analyze_risks(test_image)
            
            assert result["risk_level"] == "high"
            assert "rm -rf" in result["description"]
            assert result["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_analyze_risks_medium_risk(self, processor, test_image):
        """Test risk analysis detecting medium risk"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult(
                "Medium risk: unsaved changes", 0.9, 0.1
            )
            
            result = await processor.analyze_risks(test_image)
            
            assert result["risk_level"] == "medium"
    
    @pytest.mark.asyncio
    async def test_analyze_risks_low_risk(self, processor, test_image):
        """Test risk analysis detecting low risk"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult(
                "Normal operation, no risks", 0.9, 0.1
            )
            
            result = await processor.analyze_risks(test_image)
            
            assert result["risk_level"] == "low"
    
    # ==================== ACTION SUGGESTION TESTS ====================
    
    @pytest.mark.asyncio
    async def test_suggest_action_without_context(self, processor, test_image):
        """Test action suggestion without context"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult(
                "Click the save button to save your work", 0.9, 0.1
            )
            
            suggestion = await processor.suggest_action(test_image)
            
            # Should be truncated to 10 words
            words = suggestion.split()
            assert len(words) <= 10
    
    @pytest.mark.asyncio
    async def test_suggest_action_with_context(self, processor, test_image):
        """Test action suggestion with context"""
        context = "User is editing a document"
        
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult(
                "Save the document before closing", 0.9, 0.1
            )
            
            suggestion = await processor.suggest_action(test_image, context)
            
            # Verify context was included in prompt
            call_args = mock_process.call_args[0]
            assert context in str(call_args)
    
    # ==================== TEXT EXTRACTION TESTS ====================
    
    @pytest.mark.asyncio
    async def test_extract_text(self, processor, test_image):
        """Test text extraction from image"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult(
                "File\nEdit\nView\nHelp", 0.9, 0.1
            )
            
            texts = await processor.extract_text(test_image)
            
            assert len(texts) == 4
            assert "File" in texts
            assert "Edit" in texts
            assert "View" in texts
            assert "Help" in texts
    
    @pytest.mark.asyncio
    async def test_extract_text_empty(self, processor, test_image):
        """Test text extraction with no text"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult("", 0.9, 0.1)
            
            texts = await processor.extract_text(test_image)
            
            assert len(texts) == 0
    
    # ==================== UI ELEMENT IDENTIFICATION TESTS ====================
    
    @pytest.mark.asyncio
    async def test_identify_ui_elements(self, processor, test_image):
        """Test UI element identification"""
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult(
                "Button: Save\nButton: Cancel\nField: Username input\nMenu: File menu\nError: Invalid password",
                0.9, 0.1
            )
            
            elements = await processor.identify_ui_elements(test_image)
            
            assert "buttons" in elements
            assert "text_fields" in elements
            assert "menus" in elements
            assert "errors" in elements
            assert len(elements["buttons"]) > 0
            assert len(elements["errors"]) > 0
    
    # ==================== REGION ANALYSIS TESTS ====================
    
    @pytest.mark.asyncio
    async def test_analyze_screen_regions(self, processor, test_image):
        """Test analyzing specific screen regions"""
        regions = [(0, 0, 100, 100), (100, 100, 200, 200)]
        
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.side_effect = [
                VLMResult("Top-left corner", 0.9, 0.1),
                VLMResult("Center region", 0.9, 0.1)
            ]
            
            results = await processor.analyze_screen_regions(test_image, regions)
            
            assert len(results) == 2
            assert results[0]["region"] == (0, 0, 100, 100)
            assert results[0]["description"] == "Top-left corner"
            assert results[1]["region"] == (100, 100, 200, 200)
            assert results[1]["description"] == "Center region"
    
    # ==================== PERFORMANCE TESTS ====================
    
    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, processor, test_image):
        """Test that processing time is tracked correctly"""
        with patch.object(processor, '_generate') as mock_gen:
            mock_gen.return_value = "Test"
            processor.initialized = True
            processor.model = Mock()
            processor.processor = Mock()
            
            result = await processor.process_image(test_image)
            
            assert result.processing_time > 0
            assert result.processing_time < 10  # Should be fast for mock
    
    @pytest.mark.asyncio
    async def test_metadata_includes_device_info(self, processor, test_image):
        """Test that metadata includes device information"""
        with patch.object(processor, '_generate') as mock_gen:
            mock_gen.return_value = "Test"
            processor.initialized = True
            processor.model = Mock()
            processor.processor = Mock()
            
            result = await processor.process_image(test_image)
            
            assert "device" in result.metadata
            assert "model" in result.metadata
            assert result.metadata["model"] == "NexaAI/OmniVLM-968M"
    
    # ==================== CLEANUP TESTS ====================
    
    def test_cleanup(self, processor):
        """Test cleanup releases resources"""
        processor.model = Mock()
        processor.processor = Mock()
        processor.initialized = True
        
        processor.cleanup()
        
        assert processor.model is None
        assert processor.processor is None
        assert not processor.initialized
    
    def test_destructor_calls_cleanup(self):
        """Test that destructor calls cleanup"""
        processor = OmniVLMProcessor()
        processor.cleanup = Mock()
        
        del processor
        # Note: In practice, __del__ behavior is not guaranteed
        # This test documents expected behavior


# Performance benchmark tests
class TestOmniVLMPerformance:
    """Performance benchmarks for OmniVLM"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_single_image_inference_speed(self, benchmark):
        """Benchmark single image inference speed"""
        processor = OmniVLMProcessor(VLMConfig(device="cpu", use_fp16=False))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch.object(processor, '_generate') as mock_gen:
            mock_gen.return_value = "Test description"
            processor.initialized = True
            processor.model = Mock()
            processor.processor = Mock()
            
            async def process():
                return await processor.process_image(image)
            
            result = benchmark(asyncio.run, process())
            assert result is not None
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_batch_processing_speed(self, benchmark):
        """Benchmark batch processing speed"""
        processor = OmniVLMProcessor(VLMConfig(device="cpu", batch_size=4))
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
        
        with patch.object(processor, 'process_image') as mock_process:
            mock_process.return_value = VLMResult("Test", 0.9, 0.01)
            
            async def process():
                return await processor.process_batch(images)
            
            results = benchmark(asyncio.run, process())
            assert len(results) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])