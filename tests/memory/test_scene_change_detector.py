#!/usr/bin/env python3
"""
Test suite for Scene Change Detection using TDD approach.
Tests are written first, then implementation follows.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
import time
from PIL import Image

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.utils.mock_data import MockScreenGenerator, ScreenType


class TestSceneChangeDetector:
    """Test suite for scene change detection"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        from app.memory.scene_change_detector import SceneChangeDetector
        return SceneChangeDetector()
    
    @pytest.fixture
    def mock_generator(self):
        """Create mock screen generator"""
        return MockScreenGenerator()
    
    @pytest.fixture
    def code_editor_frame(self, mock_generator):
        """Generate code editor frame"""
        return mock_generator.generate_screen(ScreenType.CODE_EDITOR)
    
    @pytest.fixture
    def browser_frame(self, mock_generator):
        """Generate browser frame"""
        return mock_generator.generate_screen(ScreenType.BROWSER)
    
    @pytest.fixture
    def terminal_frame(self, mock_generator):
        """Generate terminal frame"""
        return mock_generator.generate_screen(ScreenType.TERMINAL)
    
    # ==================== INITIALIZATION TESTS ====================
    
    def test_detector_initialization(self):
        """Test detector initializes with correct defaults"""
        from app.memory.scene_change_detector import SceneChangeDetector
        
        detector = SceneChangeDetector()
        
        assert detector.prev_frame is None
        assert detector.prev_app is None
        assert detector.prev_histogram is None
        assert detector.ssim_threshold == 0.4
        assert detector.histogram_threshold == 0.7
        assert len(detector.change_history) == 0
    
    def test_detector_with_custom_thresholds(self):
        """Test detector with custom thresholds"""
        from app.memory.scene_change_detector import SceneChangeDetector
        
        detector = SceneChangeDetector(
            ssim_threshold=0.5,
            histogram_threshold=0.8
        )
        
        assert detector.ssim_threshold == 0.5
        assert detector.histogram_threshold == 0.8
    
    # ==================== SSIM DETECTION TESTS ====================
    
    def test_detect_visual_change_major(self, detector, code_editor_frame, browser_frame):
        """Test detection of major visual change (app switch)"""
        metadata1 = {'active_app': 'VSCode', 'timestamp': time.time()}
        metadata2 = {'active_app': 'Chrome', 'timestamp': time.time()}
        
        # First frame establishes baseline
        result1 = detector.detect_change(code_editor_frame, metadata1)
        assert not result1['scene_changed']  # First frame, no change
        
        # Completely different frame should trigger change
        result2 = detector.detect_change(browser_frame, metadata2)
        assert result2['scene_changed']
        assert result2['change_type'] == 'visual'
        assert result2['confidence'] > 0.7
    
    def test_detect_visual_change_minor(self, detector, code_editor_frame):
        """Test detection of minor visual change (scrolling)"""
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        # First frame
        result1 = detector.detect_change(code_editor_frame, metadata)
        
        # Slightly modified frame (simulate scrolling)
        modified_frame = code_editor_frame.copy()
        modified_frame[100:200, :] = modified_frame[150:250, :]  # Shift content
        
        result2 = detector.detect_change(modified_frame, metadata)
        assert not result2['scene_changed']  # Minor change, should not trigger
        assert result2['confidence'] < 0.3
    
    def test_detect_same_frame(self, detector, code_editor_frame):
        """Test that identical frames don't trigger change"""
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        # First frame
        detector.detect_change(code_editor_frame, metadata)
        
        # Same frame again
        result = detector.detect_change(code_editor_frame.copy(), metadata)
        assert not result['scene_changed']
        assert result['confidence'] < 0.1
    
    # ==================== APPLICATION CHANGE TESTS ====================
    
    def test_detect_application_change(self, detector, code_editor_frame):
        """Test detection of application change"""
        metadata1 = {'active_app': 'VSCode', 'timestamp': time.time()}
        metadata2 = {'active_app': 'Chrome', 'timestamp': time.time()}
        
        # First frame with VSCode
        result1 = detector.detect_change(code_editor_frame, metadata1)
        assert not result1['scene_changed']
        
        # Same visual but different app
        result2 = detector.detect_change(code_editor_frame, metadata2)
        assert result2['scene_changed']
        assert result2['change_type'] == 'application'
        assert 'prev_app' in result2
        assert result2['prev_app'] == 'VSCode'
    
    def test_detect_tab_change_in_browser(self, detector, browser_frame):
        """Test detection of tab change within browser"""
        metadata1 = {'active_app': 'Chrome', 'tab': 'GitHub', 'timestamp': time.time()}
        metadata2 = {'active_app': 'Chrome', 'tab': 'Gmail', 'timestamp': time.time()}
        
        # First tab
        result1 = detector.detect_change(browser_frame, metadata1)
        
        # Different tab
        result2 = detector.detect_change(browser_frame, metadata2)
        # Tab changes should be detected if metadata includes tab info
        if 'tab' in metadata2:
            assert result2.get('tab_changed', False) or result2['scene_changed']
    
    # ==================== COLOR HISTOGRAM TESTS ====================
    
    def test_detect_theme_change(self, detector):
        """Test detection of theme change (light to dark)"""
        # Light theme frame
        light_frame = np.full((720, 1280, 3), 240, dtype=np.uint8)  # Light gray
        metadata1 = {'active_app': 'VSCode', 'theme': 'light', 'timestamp': time.time()}
        
        # Dark theme frame
        dark_frame = np.full((720, 1280, 3), 30, dtype=np.uint8)  # Dark gray
        metadata2 = {'active_app': 'VSCode', 'theme': 'dark', 'timestamp': time.time()}
        
        # First frame
        result1 = detector.detect_change(light_frame, metadata1)
        
        # Theme change
        result2 = detector.detect_change(dark_frame, metadata2)
        assert result2['scene_changed']
        assert result2['change_type'] in ['theme', 'visual']
    
    def test_color_histogram_calculation(self, detector, code_editor_frame):
        """Test color histogram is calculated correctly"""
        histogram = detector._calculate_histogram(code_editor_frame)
        
        assert histogram is not None
        assert len(histogram) == 3  # RGB channels
        assert all(len(h) > 0 for h in histogram)
        assert all(np.sum(h) > 0 for h in histogram)  # Non-empty histograms
    
    # ==================== CHANGE HISTORY TESTS ====================
    
    def test_change_history_tracking(self, detector, code_editor_frame, browser_frame):
        """Test that change history is tracked"""
        metadata1 = {'active_app': 'VSCode', 'timestamp': time.time()}
        metadata2 = {'active_app': 'Chrome', 'timestamp': time.time() + 1}
        
        # Multiple changes
        detector.detect_change(code_editor_frame, metadata1)
        detector.detect_change(browser_frame, metadata2)
        
        assert len(detector.change_history) == 2
        assert detector.change_history[-1]['scene_changed'] == True
        assert 'timestamp' in detector.change_history[-1]
    
    def test_change_history_limit(self, detector):
        """Test that change history has a size limit"""
        # Generate many frames
        for i in range(150):
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            metadata = {'active_app': f'App{i}', 'timestamp': time.time() + i}
            detector.detect_change(frame, metadata)
        
        # History should be limited (default 100)
        assert len(detector.change_history) <= 100
    
    # ==================== CONFIDENCE SCORE TESTS ====================
    
    def test_confidence_calculation(self, detector, code_editor_frame, browser_frame):
        """Test confidence score calculation"""
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        # First frame
        detector.detect_change(code_editor_frame, metadata)
        
        # Major change should have high confidence
        result = detector.detect_change(browser_frame, metadata)
        assert 0.0 <= result['confidence'] <= 1.0
        assert result['confidence'] > 0.5  # High confidence for major change
    
    def test_confidence_aggregation(self, detector):
        """Test confidence aggregation from multiple signals"""
        frame1 = np.full((720, 1280, 3), 100, dtype=np.uint8)
        frame2 = np.full((720, 1280, 3), 200, dtype=np.uint8)
        
        metadata1 = {'active_app': 'App1', 'timestamp': time.time()}
        metadata2 = {'active_app': 'App2', 'timestamp': time.time()}
        
        detector.detect_change(frame1, metadata1)
        result = detector.detect_change(frame2, metadata2)
        
        # Both visual and app change should increase confidence
        assert result['confidence'] > 0.8
    
    # ==================== EDGE CASES ====================
    
    def test_first_frame_no_change(self, detector, code_editor_frame):
        """Test that first frame doesn't trigger change"""
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        result = detector.detect_change(code_editor_frame, metadata)
        assert not result['scene_changed']
        assert result['confidence'] == 0.0
    
    def test_empty_metadata(self, detector, code_editor_frame):
        """Test detection with empty metadata"""
        result = detector.detect_change(code_editor_frame, {})
        # Should still work but only use visual detection
        assert 'scene_changed' in result
        assert 'change_type' in result
    
    def test_none_frame_handling(self, detector):
        """Test handling of None frame"""
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        result = detector.detect_change(None, metadata)
        assert not result['scene_changed']
        assert result['confidence'] == 0.0
    
    # ==================== RESET FUNCTIONALITY ====================
    
    def test_reset_detector(self, detector, code_editor_frame):
        """Test resetting detector state"""
        metadata = {'active_app': 'VSCode', 'timestamp': time.time()}
        
        # Add some state
        detector.detect_change(code_editor_frame, metadata)
        assert detector.prev_frame is not None
        assert detector.prev_app is not None
        
        # Reset
        detector.reset()
        assert detector.prev_frame is None
        assert detector.prev_app is None
        assert detector.prev_histogram is None
        assert len(detector.change_history) == 0
    
    # ==================== PERFORMANCE TESTS ====================
    
    @pytest.mark.benchmark
    def test_detection_speed(self, detector, benchmark):
        """Benchmark scene change detection speed"""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        metadata = {'active_app': 'Test', 'timestamp': time.time()}
        
        result = benchmark(detector.detect_change, frame, metadata)
        assert 'scene_changed' in result


class TestSceneChangeIntegration:
    """Integration tests for scene change detection"""
    
    @pytest.mark.asyncio
    async def test_rapid_scene_changes(self):
        """Test handling of rapid scene changes"""
        from app.memory.scene_change_detector import SceneChangeDetector
        
        detector = SceneChangeDetector()
        generator = MockScreenGenerator()
        
        # Simulate rapid app switching
        apps = ['VSCode', 'Chrome', 'Terminal', 'VSCode', 'Chrome']
        changes_detected = []
        
        for i, app in enumerate(apps):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {'active_app': app, 'timestamp': time.time() + i * 0.1}
            result = detector.detect_change(frame, metadata)
            changes_detected.append(result['scene_changed'])
        
        # Should detect changes when app switches
        assert sum(changes_detected) >= 3  # At least 3 app switches
    
    @pytest.mark.asyncio
    async def test_gradual_content_change(self):
        """Test that gradual content changes don't trigger scene change"""
        from app.memory.scene_change_detector import SceneChangeDetector
        
        detector = SceneChangeDetector()
        base_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        changes_detected = []
        for i in range(10):
            # Gradually modify frame
            frame = base_frame.copy()
            frame[i*70:(i+1)*70, :] = 255  # Add white stripe
            
            metadata = {'active_app': 'Editor', 'timestamp': time.time() + i}
            result = detector.detect_change(frame, metadata)
            changes_detected.append(result['scene_changed'])
            
            base_frame = frame  # Update base for next iteration
        
        # Gradual changes should not trigger scene change
        assert sum(changes_detected) <= 2  # Allow max 2 false positives
    
    @pytest.mark.asyncio
    async def test_scene_change_with_memory_integration(self):
        """Test scene change detector with memory system"""
        from app.memory.scene_change_detector import SceneChangeDetector
        
        detector = SceneChangeDetector()
        
        # Mock memory system
        memory_cleared = []
        
        def on_scene_change(change_info):
            if change_info['scene_changed']:
                memory_cleared.append(change_info)
        
        # Simulate workflow
        generator = MockScreenGenerator()
        
        # Work in code editor
        for i in range(3):
            frame = generator.generate_screen(ScreenType.CODE_EDITOR)
            metadata = {'active_app': 'VSCode', 'timestamp': time.time() + i}
            result = detector.detect_change(frame, metadata)
            on_scene_change(result)
        
        # Switch to browser
        frame = generator.generate_screen(ScreenType.BROWSER)
        metadata = {'active_app': 'Chrome', 'timestamp': time.time() + 5}
        result = detector.detect_change(frame, metadata)
        on_scene_change(result)
        
        # Memory should be cleared once (on app switch)
        assert len(memory_cleared) == 1
        assert memory_cleared[0]['change_type'] in ['application', 'visual']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])