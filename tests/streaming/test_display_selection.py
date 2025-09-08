#!/usr/bin/env python3
"""
Test display selection functionality for multi-monitor support.
Following TDD principles - tests written before implementation.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app'))

from streaming.simple_screencapture import SimpleScreenCapture


class TestDisplayEnumeration(unittest.TestCase):
    """Test display enumeration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.capture = SimpleScreenCapture(fps=10)
        
    @patch('streaming.simple_screencapture.SC')
    def test_enumerate_displays(self, mock_sc):
        """Test that all available displays are enumerated"""
        # Arrange: Mock multiple displays
        mock_display1 = Mock()
        mock_display1.width = 1920
        mock_display1.height = 1080
        
        mock_display2 = Mock()
        mock_display2.width = 2560
        mock_display2.height = 1440
        
        mock_display3 = Mock()
        mock_display3.width = 1920
        mock_display3.height = 1080
        
        mock_content = Mock()
        mock_content.displays.return_value = [mock_display1, mock_display2, mock_display3]
        
        # Act: Get display information
        # This method should be implemented
        displays = self.capture.enumerate_displays(mock_content)
        
        # Assert: Check all displays are returned with correct info
        self.assertEqual(len(displays), 3)
        self.assertEqual(displays[0]['width'], 1920)
        self.assertEqual(displays[0]['height'], 1080)
        self.assertEqual(displays[1]['width'], 2560)
        self.assertEqual(displays[1]['height'], 1440)
        self.assertEqual(displays[2]['width'], 1920)
        self.assertEqual(displays[2]['height'], 1080)


class TestDisplaySelection(unittest.TestCase):
    """Test display selection by index"""
    
    @patch('streaming.simple_screencapture.SC')
    @patch('streaming.simple_screencapture.asyncio.sleep')
    async def test_select_display_by_index(self, mock_sleep, mock_sc):
        """Test selecting a specific display by index"""
        # Arrange: Mock multiple displays
        mock_display0 = Mock()
        mock_display0.width = 1920
        mock_display0.height = 1080
        
        mock_display1 = Mock()
        mock_display1.width = 2560
        mock_display1.height = 1440
        
        mock_content = Mock()
        mock_content.displays.return_value = [mock_display0, mock_display1]
        
        # Mock the getShareableContent call
        def mock_get_content(handler):
            handler(mock_content, None)
        
        mock_sc.SCShareableContent.getShareableContentWithCompletionHandler_ = mock_get_content
        
        # Act: Initialize with display_index=1
        capture = SimpleScreenCapture(fps=10, display_index=1)
        result = await capture.initialize()
        
        # Assert: Should use display 1 (2560x1440)
        self.assertTrue(result)
        self.assertEqual(capture.selected_display_index, 1)
        self.assertEqual(capture.selected_display.width, 2560)
        self.assertEqual(capture.selected_display.height, 1440)
    
    @patch('streaming.simple_screencapture.SC')
    @patch('streaming.simple_screencapture.asyncio.sleep')
    async def test_invalid_display_index_fallback(self, mock_sleep, mock_sc):
        """Test fallback to display 0 when invalid index is provided"""
        # Arrange: Mock only 2 displays
        mock_display0 = Mock()
        mock_display0.width = 1920
        mock_display0.height = 1080
        
        mock_display1 = Mock()
        mock_display1.width = 2560
        mock_display1.height = 1440
        
        mock_content = Mock()
        mock_content.displays.return_value = [mock_display0, mock_display1]
        
        # Mock the getShareableContent call
        def mock_get_content(handler):
            handler(mock_content, None)
        
        mock_sc.SCShareableContent.getShareableContentWithCompletionHandler_ = mock_get_content
        
        # Act: Try to initialize with invalid display_index=5
        capture = SimpleScreenCapture(fps=10, display_index=5)
        result = await capture.initialize()
        
        # Assert: Should fallback to display 0
        self.assertTrue(result)
        self.assertEqual(capture.selected_display_index, 0)
        self.assertEqual(capture.selected_display.width, 1920)
        self.assertEqual(capture.selected_display.height, 1080)


class TestCommandLineArguments(unittest.TestCase):
    """Test command-line argument parsing for display selection"""
    
    def test_parse_display_argument(self):
        """Test parsing --display command-line argument"""
        from streaming.main_streaming import parse_arguments
        
        # Test default value
        args = parse_arguments([])
        self.assertEqual(args.display, 0)
        
        # Test custom display index
        args = parse_arguments(['--display', '2'])
        self.assertEqual(args.display, 2)
        
        # Test negative value (should be handled)
        with self.assertRaises(SystemExit):
            parse_arguments(['--display', '-1'])


class TestIntegration(unittest.TestCase):
    """Integration tests for multi-monitor capture"""
    
    @patch('streaming.simple_screencapture.SC')
    @patch('streaming.simple_screencapture.asyncio.sleep')
    async def test_pipeline_with_display_selection(self, mock_sleep, mock_sc):
        """Test full pipeline with display selection"""
        from streaming.main_streaming import StreamingPipeline
        
        # Arrange: Mock 3 displays
        displays = []
        for i in range(3):
            mock_display = Mock()
            mock_display.width = 1920
            mock_display.height = 1080
            displays.append(mock_display)
        
        mock_content = Mock()
        mock_content.displays.return_value = displays
        
        def mock_get_content(handler):
            handler(mock_content, None)
        
        mock_sc.SCShareableContent.getShareableContentWithCompletionHandler_ = mock_get_content
        
        # Act: Create pipeline with display_index=2
        pipeline = StreamingPipeline({
            'display_index': 2,
            'width': 1920,
            'height': 1080,
            'fps': 10
        })
        
        # Initialize just the stream component
        await pipeline.stream.initialize()
        
        # Assert: Should be using display 2
        self.assertEqual(pipeline.stream.selected_display_index, 2)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDisplayEnumeration))
    suite.addTests(loader.loadTestsFromTestCase(TestDisplaySelection))
    suite.addTests(loader.loadTestsFromTestCase(TestCommandLineArguments))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_async_test(test_func):
    """Helper to run async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_func())
    finally:
        loop.close()


if __name__ == '__main__':
    # Run tests
    success = run_tests()
    
    # Run async tests separately
    test_selection = TestDisplaySelection()
    test_integration = TestIntegration()
    
    print("\n--- Running Async Tests ---")
    
    # Run async test methods
    async_tests = [
        test_selection.test_select_display_by_index,
        test_selection.test_invalid_display_index_fallback,
        test_integration.test_pipeline_with_display_selection
    ]
    
    for test in async_tests:
        try:
            print(f"Running {test.__name__}...")
            run_async_test(test)
            print(f"✓ {test.__name__} passed")
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            success = False
    
    sys.exit(0 if success else 1)