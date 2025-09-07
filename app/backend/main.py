#!/usr/bin/env python3
"""
Memory-safe version of FASTVLM Jarvis with aggressive memory management
Fixes the 30MB per event memory leak
"""

import asyncio
import logging
import sys
from pathlib import Path
import webbrowser
import argparse
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time
import hashlib
import psutil
import gc
import weakref

from continuous_capture import ContinuousScreenCapture
from simple_processor import SimplifiedVLMProcessor
from server import DashboardServer
from memory_safety_config import MemorySafetyConfig, MemoryMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemorySafeJarvisSystem:
    """Memory-safe Jarvis with aggressive cleanup"""
    
    def __init__(
        self, 
        fps: int = 5,  # Reduced from 10
        buffer_seconds: int = 5,  # Reduced from 20
        analysis_interval: float = 3.0,  # Increased from 2.0
        use_vlm: bool = False  # Use real VLM model
    ):
        # CRITICAL FIX 1: Use lower FPS and smaller buffer
        self.capture = ContinuousScreenCapture(
            fps=fps,
            buffer_seconds=buffer_seconds,
            save_screenshots=False
        )
        
        # Choose VLM processor based on flag
        if use_vlm:
            logger.info("⚠️ Using FastVLM processor (400MB+ memory usage)...")
            from vlm_processor import FastVLMProcessor
            self.vlm = FastVLMProcessor()
        else:
            logger.info("Using memory-safe simplified processor...")
            self.vlm = SimplifiedVLMProcessor()
        
        # Dashboard server
        self.server = DashboardServer()
        
        # Analysis settings
        self.analysis_interval = analysis_interval
        self.last_analysis_time = 0
        self.is_running = False
        
        # Session context (use weak references where possible)
        self.session_context = {
            'current_app': 'Unknown',
            'task_hypothesis': 'Monitoring screen activity',
            'last_intervention': 0
        }
        
        # Screenshot management with size limits
        self.last_screenshot_time = 0
        self.last_screenshot_hash = None
        self.recent_descriptions = []  # Limited to 5
        self.max_recent_descriptions = 5
        
        # CRITICAL FIX 3: Aggressive memory monitoring
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.memory_limit_mb = 500  # Hard limit
        self.memory_check_interval = 5  # Check every 5 seconds
        self.last_memory_check = 0
        self.event_count = 0
        self.screenshot_count = 0
        self.gc_interval = 20  # GC every 20 events (more aggressive)
        
        # CRITICAL FIX 4: Screenshot size limits
        self.max_screenshot_size_kb = 200  # Max 200KB per screenshot
        self.screenshot_quality = 60  # Lower quality
        self.resize_factor = 0.3  # Resize to 30% of original
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing Memory-Safe Jarvis System...")
        
        # Start dashboard
        await self.server.start()
        logger.info(f"✅ Dashboard available at: http://localhost:8000")
        
        # Start continuous capture with lower settings
        self.capture.start()
        logger.info(f"✅ Memory-safe capture started ({self.capture.target_fps} FPS, {self.capture.buffer_seconds}s buffer)")
        
        self.is_running = True
        logger.info("✅ Memory-Safe Jarvis ready!")
        
    async def analyze_frame(self, frame):
        """Analyze frame with memory safety"""
        analysis = {
            'timestamp': frame.timestamp,
            'motion_score': frame.motion_score,
            'roi_count': 0  # Skip ROI to save memory
        }
        
        # CRITICAL FIX 5: Process frame in-place without copies
        try:
            # Resize frame before processing to save memory
            if frame.data.shape[0] > 720:
                from cv2 import resize
                small_frame = resize(frame.data, (640, 360))
                description = await self.vlm.describe_screen(small_frame)
                risk_result = await self.vlm.analyze_for_risks(small_frame)
                # Explicitly delete the resized frame
                del small_frame
            else:
                description = await self.vlm.describe_screen(frame.data)
                risk_result = await self.vlm.analyze_for_risks(frame.data)
                
            analysis['description'] = description
            analysis['risk_score'] = risk_result['risk_score']
            analysis['risk_reason'] = risk_result.get('reason', '')
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis['description'] = "Analysis failed"
            analysis['risk_score'] = 0
            analysis['risk_reason'] = ""
        
        # Don't delete frame.data here - needed for screenshots
        # Will be deleted in send_to_dashboard after screenshot generation
        
        return analysis
        
    def calculate_importance(self, analysis):
        """Simplified importance calculation"""
        # Simpler calculation to reduce memory
        risk_weight = analysis['risk_score'] * 0.5
        motion_weight = min(analysis['motion_score'] * 10, 0.5)
        return min(risk_weight + motion_weight, 1.0)
        
    async def send_to_dashboard(self, analysis, frame=None):
        """Send to dashboard with aggressive screenshot compression"""
        importance = self.calculate_importance(analysis)
        
        # Determine tier
        if importance >= 0.7:
            tier = 'critical'
            ttl = None
        elif importance >= 0.4:
            tier = 'important'
            ttl = 120  # Reduced from 300
        else:
            tier = 'routine'
            ttl = 30  # Reduced from 60
            
        # Generate screenshots for most events (lowered threshold for better UX)
        screenshot_data = None
        logger.info(f"📸 Screenshot check: importance={importance:.2f}, has_frame={frame is not None}, has_data={frame is not None and frame.data is not None}")
        if importance >= 0.1 and frame is not None and frame.data is not None:
            logger.info(f"📸 Generating screenshot (frame shape: {frame.data.shape})")
            try:
                # Resize aggressively
                img = Image.fromarray(frame.data.astype('uint8'), 'RGB')
                new_size = (int(img.width * self.resize_factor), 
                           int(img.height * self.resize_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Compress aggressively
                buffer = BytesIO()
                img.save(buffer, format='JPEG', optimize=True, quality=self.screenshot_quality)
                
                # Check size
                buffer_size = buffer.tell()
                logger.info(f"📸 Buffer size: {buffer_size/1024:.1f}KB, limit: {self.max_screenshot_size_kb}KB")
                if buffer_size < self.max_screenshot_size_kb * 1024:
                    buffer.seek(0)
                    screenshot_data = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
                    self.screenshot_count += 1
                    logger.info(f"✅ Screenshot generated successfully")
                else:
                    logger.warning(f"⚠️ Screenshot too large: {buffer_size/1024:.1f}KB > {self.max_screenshot_size_kb}KB")
                    
                # Clean up
                buffer.close()
                del img
                
            except Exception as e:
                logger.error(f"❌ Screenshot generation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # CRITICAL FIX 6: Delete frame data after screenshot generation
        if frame is not None and hasattr(frame, 'data'):
            frame.data = None
            gc.collect(0)  # Collect generation 0 (fastest)
                
        # Update recent descriptions with limit
        self.recent_descriptions.append(analysis['description'])
        if len(self.recent_descriptions) > self.max_recent_descriptions:
            self.recent_descriptions.pop(0)
            
        # Send to dashboard
        logger.info(f"📤 Sending event: {analysis['description'][:50]}... (importance: {importance:.2f}, tier: {tier})")
        await self.server.send_screen_event(
            timestamp=float(analysis['timestamp']),
            description=analysis['description'],
            importance=float(importance),
            spoken=None,  # Skip TTS to save memory
            screenshot=screenshot_data,
            tier=tier,
            ttl=ttl,
            has_screenshot=screenshot_data is not None
        )
        
        self.event_count += 1
        
        # CRITICAL FIX 8: Aggressive garbage collection
        if self.event_count % self.gc_interval == 0:
            gc.collect()
            logger.debug(f"🗑️ Full GC at event {self.event_count}")
            
    async def check_memory_health(self):
        """Aggressive memory health monitoring"""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return True
            
        self.last_memory_check = current_time
        
        # Get memory info
        mem_info = self.process.memory_info()
        current_memory = mem_info.rss / 1024 / 1024
        memory_growth = current_memory - self.start_memory
        
        logger.info(f"💾 Memory: {current_memory:.1f}MB (growth: {memory_growth:.1f}MB, "
                   f"events: {self.event_count}, screenshots: {self.screenshot_count})")
        
        # CRITICAL FIX 9: Hard memory limit enforcement
        if current_memory > self.memory_limit_mb:
            logger.error(f"❌ MEMORY LIMIT EXCEEDED: {current_memory:.1f}MB > {self.memory_limit_mb}MB")
            
            # Emergency cleanup
            gc.collect()
            
            # Clear buffer
            with self.capture.lock:
                buffer_size = len(self.capture.ring_buffer)
                self.capture.ring_buffer.clear()
                logger.warning(f"🗑️ Cleared {buffer_size} frames from buffer")
                
            # Reduce capture rate
            self.capture.target_fps = max(2, self.capture.target_fps - 1)
            logger.warning(f"📉 Reduced FPS to {self.capture.target_fps}")
            
            # Increase analysis interval
            self.analysis_interval = min(10, self.analysis_interval + 1)
            logger.warning(f"📉 Increased interval to {self.analysis_interval}s")
            
            return False
            
        # Warning at 80% limit
        if memory_growth > self.memory_limit_mb * 0.8:
            logger.warning(f"⚠️ Memory warning: {memory_growth:.1f}MB growth")
            gc.collect()
            
        return True
        
    async def monitor_loop(self):
        """Main loop with memory safety"""
        logger.info("🔍 Starting memory-safe monitoring...")
        
        while self.is_running:
            try:
                # CRITICAL FIX 10: Check memory health first
                if not await self.check_memory_health():
                    logger.warning("⚠️ Memory critical - skipping analysis")
                    await asyncio.sleep(5)
                    continue
                    
                current_time = asyncio.get_event_loop().time()
                
                # Check if it's time to analyze
                if current_time - self.last_analysis_time >= self.analysis_interval:
                    # Get ONE frame only
                    frames = self.capture.get_latest_frames(1)
                    
                    if frames:
                        frame = frames[0]
                        
                        # Only analyze if there's motion
                        if frame.motion_score > 0.001:
                            analysis = await self.analyze_frame(frame)
                            await self.send_to_dashboard(analysis, frame)
                            
                        # CRITICAL: Delete frame after use
                        del frame
                        
                    self.last_analysis_time = current_time
                    
                # Longer sleep to reduce CPU
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(2)
                
    async def run(self):
        """Run memory-safe system"""
        await self.initialize()
        
        # Show memory safety settings
        logger.info("🛡️ Memory Safety Settings:")
        logger.info(f"   • FPS: {self.capture.target_fps}")
        logger.info(f"   • Buffer: {self.capture.buffer_seconds}s")
        logger.info(f"   • Analysis interval: {self.analysis_interval}s")
        logger.info(f"   • Memory limit: {self.memory_limit_mb}MB")
        logger.info(f"   • GC interval: {self.gc_interval} events")
        logger.info(f"   • Screenshot quality: {self.screenshot_quality}%")
        logger.info(f"   • Resize factor: {self.resize_factor}")
        
        try:
            await self.monitor_loop()
        except KeyboardInterrupt:
            logger.info("⛔ Interrupted by user")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("🛑 Shutting down Memory-Safe Jarvis...")
        
        self.is_running = False
        self.capture.stop()
        
        # Final stats
        mem_info = self.process.memory_info()
        final_memory = mem_info.rss / 1024 / 1024
        memory_growth = final_memory - self.start_memory
        
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   • Events processed: {self.event_count}")
        logger.info(f"   • Screenshots sent: {self.screenshot_count}")
        logger.info(f"   • Starting memory: {self.start_memory:.1f}MB")
        logger.info(f"   • Final memory: {final_memory:.1f}MB")
        logger.info(f"   • Memory growth: {memory_growth:.1f}MB")
        logger.info(f"   • MB per event: {memory_growth/max(1,self.event_count):.2f}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Memory-Safe FASTVLM Jarvis')
    parser.add_argument('--fps', type=int, default=5,
                       help='Capture FPS (default: 5)')
    parser.add_argument('--buffer', type=int, default=5,
                       help='Buffer seconds (default: 5)')
    parser.add_argument('--interval', type=float, default=3.0,
                       help='Analysis interval (default: 3.0)')
    parser.add_argument('--memory-limit', type=int, default=500,
                       help='Memory limit in MB (default: 500)')
    parser.add_argument('--use-vlm', action='store_true',
                       help='Use real FastVLM model for descriptions (uses 400MB+ memory)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    if args.use_vlm:
        print("🤖 FASTVLM JARVIS - FULL AI MODE")
        print("="*70)
        print()
        print("⚠️ WARNING: Using real VLM model")
        print("   • Memory usage will be 1.5-2GB higher")
        print("   • You get REAL AI descriptions")
        print("   • Monitor may crash if memory exceeds limits")
        print()
        print("💡 Tips:")
        print("   • Use --memory-limit 2000 for stability")
        print("   • Run ./install_vlm.sh to install mlx-vlm if needed")
    else:
        print("🛡️ MEMORY-SAFE FASTVLM JARVIS")
        print("="*70)
        print()
        print("✅ Fixes Applied:")
        print("   • Reduced FPS to 5")
        print("   • Smaller buffer (5 seconds)")
        print("   • Aggressive garbage collection")
        print("   • Screenshot compression (60% quality, 30% size)")
        print("   • Memory limit enforcement (500MB)")
        print("   • Frame deletion after processing")
    print()
    print("Press Ctrl+C to stop")
    print("="*70)
    print()
    
    # Create and run system
    jarvis = MemorySafeJarvisSystem(
        fps=args.fps,
        buffer_seconds=args.buffer,
        analysis_interval=args.interval,
        use_vlm=args.use_vlm
    )
    jarvis.memory_limit_mb = args.memory_limit
    
    try:
        asyncio.run(jarvis.run())
    except KeyboardInterrupt:
        print("\n⛔ Shutdown requested")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())