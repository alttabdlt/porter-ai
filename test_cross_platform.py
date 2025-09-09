#!/usr/bin/env python3
"""
Test script for cross-platform Porter.AI components.
Tests capture, VLM processing, and platform detection.
"""

import asyncio
import sys
import platform
import os
from pathlib import Path
import numpy as np
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_platform_detection():
    """Test platform detection"""
    print("\n" + "="*50)
    print("🖥️  PLATFORM DETECTION TEST")
    print("="*50)
    
    print(f"Operating System: {platform.system()}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Implementation: {platform.python_implementation()}")
    
    # Check for GPU availability
    try:
        import torch
        print(f"\nPyTorch Available: ✅")
        print(f"CUDA Available: {'✅' if torch.cuda.is_available() else '❌'}")
        if hasattr(torch.backends, 'mps'):
            print(f"MPS Available: {'✅' if torch.backends.mps.is_available() else '❌'}")
    except ImportError:
        print(f"\nPyTorch Available: ❌")
    
    print("\n✅ Platform detection complete")


async def test_capture():
    """Test cross-platform screen capture"""
    print("\n" + "="*50)
    print("📸 SCREEN CAPTURE TEST")
    print("="*50)
    
    try:
        from capture.cross_platform_capture import CrossPlatformCapture, CaptureConfig, CaptureMode
        
        config = CaptureConfig(
            mode=CaptureMode.PRIMARY_MONITOR,
            fps=5,
            width=1280,
            height=720
        )
        
        capture = CrossPlatformCapture(config)
        
        # Get monitor info
        monitors = capture.get_monitors_info()
        print(f"\nFound {len(monitors)} monitor(s):")
        for monitor in monitors:
            print(f"  Monitor {monitor['index']}: {monitor['width']}x{monitor['height']}")
        
        # Initialize capture
        if await capture.initialize():
            print("\n✅ Capture initialized successfully")
            
            # Capture a single frame
            print("\nCapturing single frame...")
            frame = await capture.capture_single_frame()
            
            if frame is not None:
                print(f"✅ Captured frame: {frame.shape}")
                print(f"   Data type: {frame.dtype}")
                print(f"   Value range: [{frame.min()}, {frame.max()}]")
                return frame
            else:
                print("❌ Failed to capture frame")
                return None
        else:
            print("❌ Failed to initialize capture")
            return None
            
    except ImportError as e:
        print(f"❌ Capture module not available: {e}")
        print("   Using mock frame for testing")
        # Return mock frame
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


async def test_vlm_processor(frame=None):
    """Test VLM processor"""
    print("\n" + "="*50)
    print("🧠 VLM PROCESSOR TEST")
    print("="*50)
    
    # Create mock frame if needed
    if frame is None:
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        print("Using mock frame for testing")
    
    # Try OmniVLM first
    try:
        from vlm_processors.omnivlm_processor import OmniVLMProcessor
        from vlm_processors.base_processor import VLMConfig, ProcessingMode
        
        print("\nTesting OmniVLM Processor...")
        
        config = VLMConfig(
            model_name="NexaAI/OmniVLM-968M",
            device="auto",
            max_tokens=50,
            temperature=0.5,
            mode=ProcessingMode.FAST
        )
        
        processor = OmniVLMProcessor(config)
        await processor.initialize()
        
        # Test basic description
        print("\nGenerating description...")
        result = await processor.process_image(frame, "Describe this screen briefly.")
        
        print(f"✅ OmniVLM Result:")
        print(f"   Description: {result.description[:100]}...")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        print(f"   Device: {result.metadata.get('device', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ OmniVLM not available: {e}")
    
    # Try simple processor as fallback
    try:
        from backend.simple_processor import SimplifiedVLMProcessor
        
        print("\nTesting Simplified Processor (fallback)...")
        processor = SimplifiedVLMProcessor()
        
        result = await processor.describe_screen(frame)
        print(f"✅ Simple Processor Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple processor not available: {e}")
        return False


async def test_integration():
    """Test full integration"""
    print("\n" + "="*50)
    print("🔗 INTEGRATION TEST")
    print("="*50)
    
    # Test capture + VLM
    print("\nTesting capture + VLM pipeline...")
    
    frame = await test_capture()
    if frame is not None:
        await test_vlm_processor(frame)
    
    print("\n✅ Integration test complete")


async def main():
    """Main test function"""
    print("\n" + "="*60)
    print("   🚀 PORTER.AI CROSS-PLATFORM TEST SUITE")
    print("="*60)
    
    # Run tests
    await test_platform_detection()
    
    # For Codespaces/headless, only test components
    if os.getenv("CODESPACES") or not os.getenv("DISPLAY"):
        print("\n⚠️  Running in headless environment (Codespaces/no display)")
        print("   Skipping screen capture test")
        print("   Testing with mock data...")
        
        # Test with mock frame
        mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        await test_vlm_processor(mock_frame)
    else:
        # Full integration test
        await test_integration()
    
    print("\n" + "="*60)
    print("   ✅ ALL TESTS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements-base.txt")
    print("2. Copy .env.example to .env and configure")
    print("3. Run main app: python app/streaming/main_streaming.py")
    print("4. Open dashboard: http://localhost:8000/streaming")


if __name__ == "__main__":
    asyncio.run(main())