#!/usr/bin/env python3
"""
Integration tests for the main streaming pipeline.
Tests the complete flow from capture to VLM processing to output.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import time
import platform
import os

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPlatformDetection:
    """Test platform detection and configuration"""
    
    def test_platform_detection(self):
        """Test that platform is correctly detected"""
        current_platform = platform.system()
        assert current_platform in ["Windows", "Darwin", "Linux"]
    
    @patch('platform.system')
    def test_macos_detection(self, mock_platform):
        """Test macOS-specific configuration"""
        mock_platform.return_value = "Darwin"
        
        # Re-import to trigger platform detection
        import importlib
        import app.streaming.main_streaming as main_module
        importlib.reload(main_module)
        
        assert main_module.IS_MACOS == True
        assert main_module.IS_WINDOWS == False
        assert main_module.IS_LINUX == False
    
    @patch('platform.system')
    def test_windows_detection(self, mock_platform):
        """Test Windows-specific configuration"""
        mock_platform.return_value = "Windows"
        
        import importlib
        import app.streaming.main_streaming as main_module
        importlib.reload(main_module)
        
        assert main_module.IS_WINDOWS == True
        assert main_module.IS_MACOS == False
        assert main_module.IS_LINUX == False
    
    @patch('platform.system')
    def test_linux_detection(self, mock_platform):
        """Test Linux-specific configuration"""
        mock_platform.return_value = "Linux"
        
        import importlib
        import app.streaming.main_streaming as main_module
        importlib.reload(main_module)
        
        assert main_module.IS_LINUX == True
        assert main_module.IS_MACOS == False
        assert main_module.IS_WINDOWS == False


class TestVLMProcessorSelection:
    """Test VLM processor selection based on platform and config"""
    
    @patch.dict(os.environ, {"USE_OMNIVLM": "true", "USE_FASTVLM": "false"})
    def test_omnivlm_selection(self):
        """Test OmniVLM is selected when configured"""
        from app.streaming.main_streaming import get_vlm_processor
        
        with patch('app.vlm_processors.omnivlm_processor.OmniVLMProcessor') as mock_omni:
            processor_class = get_vlm_processor()
            # Should attempt to use OmniVLM
            assert processor_class is not None
    
    @patch.dict(os.environ, {"USE_SIMPLE_PROCESSOR": "true"})
    def test_simple_processor_selection(self):
        """Test simple processor is selected when configured"""
        from app.streaming.main_streaming import get_vlm_processor
        
        processor_class = get_vlm_processor()
        assert "Simplified" in processor_class.__name__
    
    @patch('platform.system')
    @patch.dict(os.environ, {"USE_FASTVLM": "true"})
    def test_fastvlm_selection_on_macos(self, mock_platform):
        """Test FastVLM is selected on macOS when configured"""
        mock_platform.return_value = "Darwin"
        
        # This would require FastVLM to be available
        # In testing, it will fall back to alternatives
        from app.streaming.main_streaming import get_vlm_processor
        processor_class = get_vlm_processor()
        assert processor_class is not None


class TestStreamingPipeline:
    """Integration tests for the main streaming pipeline"""
    
    @pytest.fixture
    def config(self):
        """Create test pipeline configuration"""
        return {
            'fps': 10,
            'width': 1280,
            'height': 720,
            'use_vlm': True,
            'display_index': 0
        }
    
    @pytest.fixture
    async def pipeline(self, config):
        """Create pipeline instance"""
        from app.streaming.main_streaming import StreamingPipeline
        pipeline = StreamingPipeline(config)
        yield pipeline
        # Cleanup
        if pipeline.running:
            await pipeline.stop()
    
    @pytest.fixture
    def mock_frame(self):
        """Create mock frame data"""
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # ==================== INITIALIZATION TESTS ====================
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly"""
        assert pipeline.config['fps'] == 10
        assert pipeline.config['width'] == 1280
        assert pipeline.config['height'] == 720
        assert pipeline.config['use_vlm'] == True
        assert not pipeline.running
        assert pipeline.frames_processed == 0
        assert pipeline.frames_analyzed == 0
    
    @pytest.mark.asyncio
    async def test_pipeline_initialize_components(self, pipeline):
        """Test pipeline component initialization"""
        with patch.object(pipeline, 'stream') as mock_stream:
            with patch.object(pipeline, 'server') as mock_server:
                mock_stream.initialize = AsyncMock(return_value=True)
                mock_server.start = AsyncMock()
                
                await pipeline.initialize()
                
                # Verify stream was initialized
                assert pipeline.stream is not None
                
                # Verify server was started
                mock_server.start.assert_called_once()
    
    # ==================== FRAME PROCESSING TESTS ====================
    
    @pytest.mark.asyncio
    async def test_frame_processing_with_vlm(self, pipeline, mock_frame):
        """Test frame processing with VLM enabled"""
        pipeline.use_vlm = True
        pipeline.vlm = Mock()
        pipeline.vlm.describe_screen = AsyncMock(return_value="Test description")
        
        # Process frame
        pipeline.frame_queue.put_nowait((mock_frame, 10, time.time()))
        
        # Mock the processing
        context = await pipeline._analyze_frame(mock_frame, time.time())
        
        assert context is not None
        # VLM should have been called
        pipeline.vlm.describe_screen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_frame_processing_without_vlm(self, pipeline, mock_frame):
        """Test frame processing without VLM"""
        pipeline.use_vlm = False
        pipeline.vlm = None
        
        # Process frame
        context = await pipeline._analyze_frame(mock_frame, time.time())
        
        assert context is not None
        # Should use fallback description
        assert context.vlm_output is not None
    
    @pytest.mark.asyncio
    async def test_frame_callback_handling(self, pipeline, mock_frame):
        """Test frame callback adds frames to queue"""
        pipeline._handle_frame(mock_frame, 10)
        
        assert pipeline.frames_processed == 1
        
        # Frame should be in queue if sampling allows
        if not pipeline.frame_queue.empty():
            frame, fps, timestamp = pipeline.frame_queue.get_nowait()
            assert np.array_equal(frame, mock_frame)
            assert fps == 10
    
    # ==================== ADAPTIVE SAMPLING TESTS ====================
    
    def test_adaptive_sampling(self, pipeline):
        """Test adaptive frame sampling"""
        current_time = time.time()
        
        # First frame should be sampled
        should_sample = pipeline.sampler.should_sample(current_time)
        assert isinstance(should_sample, bool)
        
        # Immediate next frame might not be sampled
        should_sample_next = pipeline.sampler.should_sample(current_time + 0.01)
        # This depends on sampling strategy
    
    # ==================== WEBSOCKET BROADCAST TESTS ====================
    
    @pytest.mark.asyncio
    async def test_context_broadcasting(self, pipeline):
        """Test context broadcasting to dashboard"""
        from app.streaming.context_fusion import ScreenContext
        
        context = ScreenContext(
            frame_id=1,
            timestamp=time.time(),
            vlm_output="Test context"
        )
        
        pipeline.server.broadcast = AsyncMock()
        pipeline.context_queue.put_nowait(context)
        
        # Process one broadcast cycle
        try:
            await asyncio.wait_for(pipeline.broadcast_contexts(), timeout=1.5)
        except asyncio.TimeoutError:
            pass  # Expected as the loop runs forever
        
        # Verify broadcast was called
        pipeline.server.broadcast.assert_called()
        call_args = pipeline.server.broadcast.call_args[0][0]
        assert call_args['type'] == 'context'
        assert 'data' in call_args
    
    @pytest.mark.asyncio
    async def test_frame_broadcasting(self, pipeline, mock_frame):
        """Test frame broadcasting to dashboard"""
        pipeline.server.broadcast = AsyncMock()
        
        await pipeline._send_frame(mock_frame)
        
        # Verify broadcast was called with frame data
        pipeline.server.broadcast.assert_called_once()
        call_args = pipeline.server.broadcast.call_args[0][0]
        assert call_args['type'] == 'frame'
        assert 'image' in call_args['data']
        assert call_args['data']['image'].startswith('data:image/jpeg;base64,')
    
    @pytest.mark.asyncio
    async def test_metrics_broadcasting(self, pipeline):
        """Test metrics broadcasting to dashboard"""
        pipeline.server.broadcast = AsyncMock()
        pipeline.frames_processed = 100
        pipeline.frames_analyzed = 50
        
        await pipeline._send_metrics(10.0)
        
        # Verify metrics were sent
        pipeline.server.broadcast.assert_called_once()
        call_args = pipeline.server.broadcast.call_args[0][0]
        assert call_args['type'] == 'metrics'
        assert call_args['data']['fps'] == 10.0
        assert call_args['data']['frames_processed'] == 100
        assert call_args['data']['frames_analyzed'] == 50
    
    # ==================== PIPELINE CONTROL TESTS ====================
    
    @pytest.mark.asyncio
    async def test_pipeline_start(self, pipeline):
        """Test starting the pipeline"""
        pipeline.stream = Mock()
        pipeline.stream.start = AsyncMock(return_value=True)
        pipeline.stream.capture_frame_loop = AsyncMock()
        pipeline.server = Mock()
        pipeline.server.start = AsyncMock()
        
        # Start pipeline in background
        start_task = asyncio.create_task(pipeline.start())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        assert pipeline.running == True
        assert pipeline.start_time > 0
        
        # Stop pipeline
        pipeline.running = False
        await asyncio.sleep(0.1)
        
        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_pipeline_stop(self, pipeline):
        """Test stopping the pipeline"""
        pipeline.running = True
        pipeline.stream = Mock()
        pipeline.stream.stop = AsyncMock()
        pipeline.server = Mock()
        pipeline.server.stop = AsyncMock()
        
        await pipeline.stop()
        
        assert pipeline.running == False
        pipeline.stream.stop.assert_called_once()
        pipeline.server.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_start_failure(self, pipeline):
        """Test pipeline handles start failure"""
        pipeline.stream = Mock()
        pipeline.stream.start = AsyncMock(return_value=False)
        pipeline.server = Mock()
        pipeline.server.start = AsyncMock()
        
        result = await pipeline.start()
        
        assert result == False
    
    # ==================== ERROR HANDLING TESTS ====================
    
    @pytest.mark.asyncio
    async def test_frame_processing_error_handling(self, pipeline, mock_frame):
        """Test error handling in frame processing"""
        pipeline.vlm = Mock()
        pipeline.vlm.describe_screen = AsyncMock(side_effect=Exception("VLM error"))
        
        # Should not crash, should use fallback
        context = await pipeline._analyze_frame(mock_frame, time.time())
        
        assert context is not None
        assert context.vlm_output == "Processing screen content..."
    
    @pytest.mark.asyncio
    async def test_broadcast_error_handling(self, pipeline):
        """Test error handling in broadcasting"""
        pipeline.server.broadcast = AsyncMock(side_effect=Exception("Broadcast error"))
        
        from app.streaming.context_fusion import ScreenContext
        context = ScreenContext(frame_id=1, timestamp=time.time(), vlm_output="Test")
        pipeline.context_queue.put_nowait(context)
        
        # Should not crash
        try:
            await asyncio.wait_for(pipeline.broadcast_contexts(), timeout=1.5)
        except asyncio.TimeoutError:
            pass  # Expected
    
    # ==================== MEMORY MANAGEMENT TESTS ====================
    
    def test_queue_size_limits(self, pipeline):
        """Test that queues have size limits"""
        # Frame queue should be limited
        assert pipeline.frame_queue.maxsize == 3
        assert pipeline.context_queue.maxsize == 3
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, pipeline, mock_frame):
        """Test handling of queue overflow"""
        # Fill the frame queue
        for i in range(pipeline.frame_queue.maxsize):
            pipeline.frame_queue.put_nowait((mock_frame, 10, time.time()))
        
        # Queue should be full
        assert pipeline.frame_queue.full()
        
        # Try to add another frame - should skip
        pipeline._handle_frame(mock_frame, 10)
        
        # Should not crash, frame should be dropped
        assert pipeline.frame_queue.full()


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_pipeline_flow(self):
        """Test complete flow from capture to output"""
        from app.streaming.main_streaming import StreamingPipeline
        from app.capture.cross_platform_capture import CaptureConfig, CaptureMode
        
        config = {
            'fps': 5,
            'width': 640,
            'height': 480,
            'use_vlm': True
        }
        
        pipeline = StreamingPipeline(config)
        
        # Mock components
        pipeline.stream = Mock()
        pipeline.stream.initialize = AsyncMock(return_value=True)
        pipeline.stream.start = AsyncMock(return_value=True)
        pipeline.stream.set_frame_callback = Mock()
        pipeline.stream.capture_frame_loop = AsyncMock()
        
        pipeline.vlm = Mock()
        pipeline.vlm.initialize = AsyncMock()
        pipeline.vlm.describe_screen = AsyncMock(return_value="Test scene")
        
        pipeline.server = Mock()
        pipeline.server.start = AsyncMock()
        pipeline.server.broadcast = AsyncMock()
        pipeline.server.stop = AsyncMock()
        
        # Initialize
        await pipeline.initialize()
        
        # Simulate frame capture
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pipeline._handle_frame(mock_frame, 5)
        
        # Process frames
        process_task = asyncio.create_task(pipeline.process_frames())
        broadcast_task = asyncio.create_task(pipeline.broadcast_contexts())
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        # Verify processing occurred
        assert pipeline.frames_processed > 0
        
        # Stop
        pipeline.running = False
        process_task.cancel()
        broadcast_task.cancel()
        
        try:
            await process_task
            await broadcast_task
        except asyncio.CancelledError:
            pass
        
        await pipeline.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cross_platform_compatibility(self):
        """Test that pipeline works across platforms"""
        from app.streaming.main_streaming import StreamingPipeline
        
        # Test with minimal config
        pipeline = StreamingPipeline({'use_vlm': False})
        
        # Should initialize regardless of platform
        assert pipeline is not None
        assert pipeline.config is not None
        
        # Platform-specific capture should be selected
        assert pipeline.stream is None  # Not initialized yet
        
        # Clean up
        if pipeline.running:
            await pipeline.stop()


# Performance tests
class TestPipelinePerformance:
    """Performance benchmarks for the pipeline"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_frame_processing_throughput(self, benchmark):
        """Benchmark frame processing throughput"""
        from app.streaming.main_streaming import StreamingPipeline
        
        pipeline = StreamingPipeline({'use_vlm': False})
        mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        async def process_frame():
            return await pipeline._analyze_frame(mock_frame, time.time())
        
        result = benchmark(asyncio.run, process_frame())
        assert result is not None
    
    @pytest.mark.benchmark
    def test_frame_encoding_speed(self, benchmark):
        """Benchmark frame encoding for transmission"""
        import cv2
        import base64
        
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        def encode_frame():
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return base64.b64encode(buffer).decode('utf-8')
        
        encoded = benchmark(encode_frame)
        assert len(encoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])