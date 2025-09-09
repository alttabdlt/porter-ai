#!/usr/bin/env python3
"""
Test suite for cross-platform screen capture using TDD approach.
Tests screen capture functionality across different platforms.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from PIL import Image
import time

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.capture.cross_platform_capture import (
    CrossPlatformCapture,
    CaptureConfig,
    CaptureMode
)


class TestCaptureConfig:
    """Test capture configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CaptureConfig()
        
        assert config.mode == CaptureMode.PRIMARY_MONITOR
        assert config.fps == 10
        assert config.width == 1920
        assert config.height == 1080
        assert config.region is None
        assert config.monitor_index == 1
        assert config.resize == True
        assert config.quality == 95
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = CaptureConfig(
            mode=CaptureMode.REGION,
            fps=30,
            width=1280,
            height=720,
            region=(100, 100, 500, 400),
            monitor_index=2,
            resize=False,
            quality=80
        )
        
        assert config.mode == CaptureMode.REGION
        assert config.fps == 30
        assert config.width == 1280
        assert config.height == 720
        assert config.region == (100, 100, 500, 400)
        assert config.monitor_index == 2
        assert config.resize == False
        assert config.quality == 80


class TestCrossPlatformCapture:
    """Test suite for cross-platform capture"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return CaptureConfig(
            mode=CaptureMode.PRIMARY_MONITOR,
            fps=10,
            width=1280,
            height=720
        )
    
    @pytest.fixture
    def capture(self, config):
        """Create capture instance"""
        return CrossPlatformCapture(config)
    
    @pytest.fixture
    def mock_monitors(self):
        """Create mock monitor data"""
        return [
            {"left": 0, "top": 0, "width": 2560, "height": 1440},  # All monitors
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary
            {"left": 1920, "top": 0, "width": 1920, "height": 1080}  # Secondary
        ]
    
    @pytest.fixture
    def mock_screenshot(self):
        """Create mock screenshot object"""
        mock = Mock()
        mock.size = (1920, 1080)
        mock.bgra = b'\x00' * (1920 * 1080 * 4)  # Mock BGRA data
        return mock
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_capture_initialization(self, config):
        """Test capture initializes with correct configuration"""
        capture = CrossPlatformCapture(config)
        
        assert capture.config == config
        assert not capture.is_running
        assert capture.frame_callback is None
        assert capture.frames_captured == 0
        assert capture.start_time == 0
        assert capture._sct is None
    
    def test_capture_default_initialization(self):
        """Test capture with default configuration"""
        capture = CrossPlatformCapture()
        
        assert capture.config.mode == CaptureMode.PRIMARY_MONITOR
        assert capture.config.fps == 10
    
    # ==================== MONITOR DETECTION TESTS ====================
    
    def test_get_monitors_info(self, capture, mock_monitors):
        """Test getting monitor information"""
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = mock_monitors
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            monitors = capture.get_monitors_info()
            
            assert len(monitors) == 2  # Excludes "all monitors"
            assert monitors[0]["width"] == 1920
            assert monitors[0]["height"] == 1080
            assert monitors[0]["is_primary"] == True
            assert monitors[1]["is_primary"] == False
    
    def test_get_monitors_info_empty(self, capture):
        """Test getting monitor info with no monitors"""
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [{"width": 0, "height": 0}]  # Only "all monitors"
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            monitors = capture.get_monitors_info()
            
            assert len(monitors) == 0
    
    # ==================== INITIALIZATION TESTS ====================
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, capture):
        """Test successful initialization"""
        with patch.object(capture, 'get_monitors_info') as mock_get:
            mock_get.return_value = [{"index": 1, "width": 1920, "height": 1080}]
            
            result = await capture.initialize()
            
            assert result == True
            assert mock_get.called
    
    @pytest.mark.asyncio
    async def test_initialize_region_mode_without_region(self):
        """Test initialization fails for region mode without region"""
        config = CaptureConfig(mode=CaptureMode.REGION, region=None)
        capture = CrossPlatformCapture(config)
        
        result = await capture.initialize()
        
        assert result == False
    
    @pytest.mark.asyncio
    async def test_initialize_with_exception(self, capture):
        """Test initialization handles exceptions"""
        with patch.object(capture, 'get_monitors_info') as mock_get:
            mock_get.side_effect = Exception("Test error")
            
            result = await capture.initialize()
            
            assert result == False
    
    # ==================== CALLBACK TESTS ====================
    
    def test_set_frame_callback(self, capture):
        """Test setting frame callback"""
        callback = Mock()
        capture.set_frame_callback(callback)
        
        assert capture.frame_callback == callback
    
    # ==================== CAPTURE CONTROL TESTS ====================
    
    @pytest.mark.asyncio
    async def test_start_capture(self, capture):
        """Test starting capture"""
        with patch('asyncio.create_task') as mock_task:
            await capture.start_capture()
            
            assert capture.is_running == True
            assert capture.start_time > 0
            assert capture.frames_captured == 0
            assert mock_task.called
    
    @pytest.mark.asyncio
    async def test_start_capture_already_running(self, capture):
        """Test starting capture when already running"""
        capture.is_running = True
        
        with patch('asyncio.create_task') as mock_task:
            await capture.start_capture()
            
            assert not mock_task.called
    
    @pytest.mark.asyncio
    async def test_stop_capture(self, capture):
        """Test stopping capture"""
        capture.is_running = True
        capture.start_time = time.time()
        capture.frames_captured = 10
        
        await capture.stop_capture()
        
        assert capture.is_running == False
    
    # ==================== FRAME CAPTURE TESTS ====================
    
    @pytest.mark.asyncio
    async def test_capture_single_frame(self, capture, mock_screenshot):
        """Test capturing a single frame"""
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [
                {"left": 0, "top": 0, "width": 2560, "height": 1440},
                {"left": 0, "top": 0, "width": 1920, "height": 1080}
            ]
            mock_sct.grab.return_value = mock_screenshot
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch('PIL.Image.frombytes') as mock_frombytes:
                mock_img = Mock()
                mock_img.width = 1920
                mock_img.height = 1080
                mock_img.resize.return_value = mock_img
                mock_frombytes.return_value = mock_img
                
                with patch('numpy.array') as mock_array:
                    mock_array.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
                    
                    frame = await capture.capture_single_frame()
                    
                    assert frame is not None
                    assert frame.shape == (720, 1280, 3)
    
    @pytest.mark.asyncio
    async def test_capture_frame_full_screen_mode(self, capture, mock_screenshot):
        """Test capturing frame in full screen mode"""
        capture.config.mode = CaptureMode.FULL_SCREEN
        
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [
                {"left": 0, "top": 0, "width": 2560, "height": 1440},  # All monitors
                {"left": 0, "top": 0, "width": 1920, "height": 1080}
            ]
            mock_sct.grab.return_value = mock_screenshot
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch('PIL.Image.frombytes') as mock_frombytes:
                mock_img = Mock()
                mock_img.width = 2560
                mock_img.height = 1440
                mock_img.resize.return_value = mock_img
                mock_frombytes.return_value = mock_img
                
                with patch('numpy.array') as mock_array:
                    mock_array.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
                    
                    frame = await capture._capture_frame(mock_sct)
                    
                    # Should capture from monitor index 0 (all monitors)
                    mock_sct.grab.assert_called_with(mock_sct.monitors[0])
    
    @pytest.mark.asyncio
    async def test_capture_frame_region_mode(self, capture, mock_screenshot):
        """Test capturing frame in region mode"""
        capture.config.mode = CaptureMode.REGION
        capture.config.region = (100, 100, 500, 400)
        
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.grab.return_value = mock_screenshot
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch('PIL.Image.frombytes') as mock_frombytes:
                mock_img = Mock()
                mock_img.width = 500
                mock_img.height = 400
                mock_img.resize.return_value = mock_img
                mock_frombytes.return_value = mock_img
                
                with patch('numpy.array') as mock_array:
                    mock_array.return_value = np.zeros((400, 500, 3), dtype=np.uint8)
                    
                    frame = await capture._capture_frame(mock_sct)
                    
                    # Should capture the specified region
                    expected_monitor = {"left": 100, "top": 100, "width": 500, "height": 400}
                    mock_sct.grab.assert_called_with(expected_monitor)
    
    @pytest.mark.asyncio
    async def test_capture_frame_with_resize(self, capture, mock_screenshot):
        """Test frame resizing during capture"""
        capture.config.resize = True
        capture.config.width = 640
        capture.config.height = 360
        
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [None, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
            mock_sct.grab.return_value = mock_screenshot
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch('PIL.Image.frombytes') as mock_frombytes:
                mock_img = Mock()
                mock_img.width = 1920
                mock_img.height = 1080
                mock_resized = Mock()
                mock_img.resize.return_value = mock_resized
                mock_frombytes.return_value = mock_img
                
                with patch('numpy.array') as mock_array:
                    mock_array.return_value = np.zeros((360, 640, 3), dtype=np.uint8)
                    
                    frame = await capture._capture_frame(mock_sct)
                    
                    # Should resize the image
                    mock_img.resize.assert_called_once()
                    assert frame.shape == (360, 640, 3)
    
    @pytest.mark.asyncio
    async def test_capture_frame_error_handling(self, capture):
        """Test frame capture error handling"""
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.grab.side_effect = Exception("Capture error")
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            frame = await capture._capture_frame(mock_sct)
            
            assert frame is None
    
    # ==================== CAPTURE LOOP TESTS ====================
    
    @pytest.mark.asyncio
    async def test_capture_loop_with_callback(self, capture, mock_screenshot):
        """Test capture loop calls callback"""
        frames_received = []
        
        def callback(frame, frame_num):
            frames_received.append(frame_num)
            if len(frames_received) >= 3:
                capture.is_running = False
        
        capture.set_frame_callback(callback)
        capture.is_running = True
        capture.config.fps = 100  # Fast for testing
        
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [None, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
            mock_sct.grab.return_value = mock_screenshot
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch.object(capture, '_capture_frame') as mock_capture:
                mock_capture.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
                
                await capture._capture_loop()
                
                assert len(frames_received) >= 3
                assert capture.frames_captured >= 3
    
    @pytest.mark.asyncio
    async def test_capture_loop_async_callback(self, capture):
        """Test capture loop with async callback"""
        async_callback = AsyncMock()
        capture.set_frame_callback(async_callback)
        capture.is_running = True
        capture.config.fps = 100
        
        # Run briefly then stop
        async def stop_after_delay():
            await asyncio.sleep(0.1)
            capture.is_running = False
        
        with patch.object(capture, '_capture_frame') as mock_capture:
            mock_capture.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            await asyncio.gather(
                capture._capture_loop(),
                stop_after_delay()
            )
            
            assert async_callback.called
    
    # ==================== REGION CAPTURE TESTS ====================
    
    @pytest.mark.asyncio
    async def test_capture_region(self, capture, mock_screenshot):
        """Test capturing a specific region"""
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.grab.return_value = mock_screenshot
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch('PIL.Image.frombytes') as mock_frombytes:
                mock_img = Mock()
                mock_frombytes.return_value = mock_img
                
                with patch('numpy.array') as mock_array:
                    mock_array.return_value = np.zeros((200, 300, 3), dtype=np.uint8)
                    
                    region = await capture.capture_region(100, 100, 300, 200)
                    
                    assert region is not None
                    expected_monitor = {"left": 100, "top": 100, "width": 300, "height": 200}
                    mock_sct.grab.assert_called_with(expected_monitor)
    
    # ==================== ALL MONITORS CAPTURE TESTS ====================
    
    @pytest.mark.asyncio
    async def test_capture_all_monitors(self, capture):
        """Test capturing all monitors"""
        capture.config.mode = CaptureMode.ALL_MONITORS
        
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [
                None,
                {"left": 0, "top": 0, "width": 1920, "height": 1080},
                {"left": 1920, "top": 0, "width": 1920, "height": 1080}
            ]
            
            mock_screenshot1 = Mock()
            mock_screenshot1.size = (1920, 1080)
            mock_screenshot1.bgra = b'\x00' * (1920 * 1080 * 4)
            
            mock_screenshot2 = Mock()
            mock_screenshot2.size = (1920, 1080)
            mock_screenshot2.bgra = b'\xFF' * (1920 * 1080 * 4)
            
            mock_sct.grab.side_effect = [mock_screenshot1, mock_screenshot2]
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch('PIL.Image.frombytes') as mock_frombytes:
                mock_img1 = Mock()
                mock_img2 = Mock()
                mock_frombytes.side_effect = [mock_img1, mock_img2]
                
                with patch('numpy.array') as mock_array:
                    mock_array.side_effect = [
                        np.zeros((1080, 1920, 3), dtype=np.uint8),
                        np.ones((1080, 1920, 3), dtype=np.uint8) * 255
                    ]
                    
                    with patch('numpy.concatenate') as mock_concat:
                        mock_concat.return_value = np.zeros((1080, 3840, 3), dtype=np.uint8)
                        
                        frame = await capture._capture_all_monitors(mock_sct)
                        
                        assert frame is not None
                        assert mock_concat.called
                        # Should concatenate horizontally (axis=1)
                        mock_concat.assert_called_once()
                        assert mock_concat.call_args[1]['axis'] == 1
    
    # ==================== STATISTICS TESTS ====================
    
    def test_get_statistics_not_running(self, capture):
        """Test getting statistics when not running"""
        stats = capture.get_statistics()
        
        assert stats["frames_captured"] == 0
        assert stats["actual_fps"] == 0
        assert stats["uptime"] == 0
        assert stats["is_running"] == False
    
    def test_get_statistics_running(self, capture):
        """Test getting statistics while running"""
        capture.is_running = True
        capture.start_time = time.time() - 10  # Started 10 seconds ago
        capture.frames_captured = 100
        
        stats = capture.get_statistics()
        
        assert stats["frames_captured"] == 100
        assert stats["actual_fps"] == pytest.approx(10.0, rel=0.1)
        assert stats["target_fps"] == 10
        assert stats["uptime"] >= 10
        assert stats["is_running"] == True
        assert stats["mode"] == "primary_monitor"


# Performance benchmark tests
class TestCapturePerformance:
    """Performance benchmarks for screen capture"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_single_frame_capture_speed(self, benchmark):
        """Benchmark single frame capture speed"""
        config = CaptureConfig(resize=False)  # No resize for raw speed
        capture = CrossPlatformCapture(config)
        
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [None, {"left": 0, "top": 0, "width": 1920, "height": 1080}]
            mock_screenshot = Mock()
            mock_screenshot.size = (1920, 1080)
            mock_screenshot.bgra = b'\x00' * (1920 * 1080 * 4)
            mock_sct.grab.return_value = mock_screenshot
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            with patch('PIL.Image.frombytes') as mock_frombytes:
                mock_img = Mock()
                mock_img.width = 1920
                mock_img.height = 1080
                mock_frombytes.return_value = mock_img
                
                with patch('numpy.array') as mock_array:
                    mock_array.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    
                    async def capture_frame():
                        return await capture.capture_single_frame()
                    
                    frame = benchmark(asyncio.run, capture_frame())
                    assert frame is not None
    
    @pytest.mark.benchmark
    def test_monitor_detection_speed(self, benchmark):
        """Benchmark monitor detection speed"""
        capture = CrossPlatformCapture()
        
        with patch('mss.mss') as mock_mss:
            mock_sct = Mock()
            mock_sct.monitors = [
                {"left": 0, "top": 0, "width": 2560, "height": 1440},
                {"left": 0, "top": 0, "width": 1920, "height": 1080},
                {"left": 1920, "top": 0, "width": 1920, "height": 1080}
            ]
            mock_mss.return_value.__enter__.return_value = mock_sct
            
            monitors = benchmark(capture.get_monitors_info)
            assert len(monitors) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])