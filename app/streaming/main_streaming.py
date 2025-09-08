#!/usr/bin/env python3
"""
Main streaming pipeline integration.
Orchestrates all components for real-time intelligent screen monitoring.
"""

import asyncio
import logging
import time
import psutil
import os
import subprocess
import signal
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path
import sys
import base64
import cv2

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import streaming components
from simple_screencapture import SimpleScreenCapture
from screencapture_kit import AdaptiveFrameSampler
from context_fusion import ContextFusion

# Import server for WebSocket
from backend.server import DashboardServer

# Import VLM processor
try:
    from backend.vlm_processor import FastVLMProcessor
    VLM_AVAILABLE = True
    logger.info("FastVLM processor available")
except ImportError:
    logger.warning("FastVLM not available, using simplified processor")
    # Fall back to simple processor if VLM fails
    try:
        from backend.simple_processor import SimplifiedVLMProcessor as FastVLMProcessor
    except ImportError:
        # If simple_processor doesn't exist yet, use the one from vlm_processor
        from backend.vlm_processor import SimplifiedVLMProcessor as FastVLMProcessor
    VLM_AVAILABLE = False

class StreamingPipeline:
    """
    Main streaming pipeline that integrates all components.
    Implements: ScreenCaptureKit → Sampler → Saliency → Intelligence → Fusion
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize streaming pipeline with all components"""
        self.config = config or {}
        
        # Core components
        self.stream: Optional[SimpleScreenCapture] = None
        self.sampler = AdaptiveFrameSampler(base_fps=10)
        self.context_fusion = ContextFusion()
        
        # VLM processor
        self.vlm = None
        self.use_vlm = config.get('use_vlm', True)  # Enable VLM by default
        if self.use_vlm:
            try:
                self.vlm = FastVLMProcessor()
                logger.info(f"VLM processor initialized (Full={VLM_AVAILABLE})")
            except Exception as e:
                logger.warning(f"Failed to initialize VLM: {e}")
                self.vlm = None
        
        # Server for dashboard
        self.server = DashboardServer()
        
        # Performance monitoring
        self.process = psutil.Process(os.getpid())
        self.start_time = 0
        self.frames_processed = 0
        self.frames_analyzed = 0
        
        # Processing queues (reduced for memory)
        self.frame_queue = asyncio.Queue(maxsize=3)
        self.context_queue = asyncio.Queue(maxsize=3)
        
        # State
        self.running = False
        self.last_metrics_time = 0
        self.metrics_interval = 1.0  # Send metrics every second
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing streaming pipeline...")
        
        # Initialize stream
        fps = self.config.get('fps', 30)
        
        self.stream = SimpleScreenCapture(fps=fps)
        self.stream.set_frame_callback(self._handle_frame)
        
        # Initialize VLM if available
        if self.vlm:
            try:
                logger.info("Initializing VLM model...")
                await self.vlm.initialize()
                logger.info("VLM model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize VLM model: {e}")
                self.vlm = None
        
        # Start server
        await self.server.start()
        
        logger.info("Pipeline initialized successfully")
        
    def _handle_frame(self, frame: np.ndarray, fps: float):
        """Callback for new frames from ScreenCaptureKit"""
        current_time = time.time()
        
        # Adaptive sampling
        if self.sampler.should_sample(current_time):
            # Put frame in queue for processing
            try:
                self.frame_queue.put_nowait((frame, fps, current_time))
                self.frames_processed += 1
            except asyncio.QueueFull:
                logger.debug("Frame queue full, dropping frame")
    
    async def process_frames(self):
        """Process frames through the intelligence pipeline"""
        while self.running:
            try:
                # Get frame from queue
                frame, fps, timestamp = await asyncio.wait_for(
                    self.frame_queue.get(), timeout=1.0
                )
                
                # Process through VLM pipeline directly
                context = await self._analyze_frame(frame, timestamp)
                
                # Queue context for output
                await self.context_queue.put(context)
                self.frames_analyzed += 1
                
                # Send frame to frontend
                await self._send_frame(frame)
                
                # Update sampler based on context
                self.sampler.update_activity(
                    0.5,  # Default activity level
                    context.importance
                )
                
                # Send metrics periodically
                current_time = time.time()
                if current_time - self.last_metrics_time > self.metrics_interval:
                    await self._send_metrics(fps)
                    self.last_metrics_time = current_time
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
    
    async def _analyze_frame(self, frame: np.ndarray, timestamp: float):
        """Analyze frame through VLM only"""
        start_time = time.time()
        
        # VLM inference (using FastVLM)
        vlm_output = await self._simple_vlm_inference(frame)
        
        # Simple context fusion
        context = self.context_fusion.fuse_simple(
            vlm_output=vlm_output,
            frame_timestamp=timestamp
        )
        
        processing_time = time.time() - start_time
        logger.debug(f"Frame analysis complete in {processing_time*1000:.1f}ms")
        
        return context
    
    async def _simple_vlm_inference(self, frame: np.ndarray) -> str:
        """VLM inference using FastVLM"""
        # Try to use actual VLM if available
        if self.vlm and self.use_vlm:
            try:
                # Use FastVLM to generate real description
                description = await self.vlm.describe_screen(frame)
                if description and description != "No description available":
                    return description
            except Exception as e:
                logger.debug(f"VLM inference failed: {e}")
        
        # Simple fallback if VLM fails
        return "Processing screen content..."
    
    async def broadcast_contexts(self):
        """Broadcast analyzed contexts to dashboard"""
        while self.running:
            try:
                # Get context from queue
                context = await asyncio.wait_for(
                    self.context_queue.get(), timeout=1.0
                )
                
                # Send to dashboard
                await self.server.broadcast({
                    'type': 'context',
                    'data': context.to_dict()
                })
                
                # Removed OCR, UI tree, and saliency broadcasts - not needed with VLM only
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error broadcasting context: {e}")
    
    async def _send_frame(self, frame: np.ndarray):
        """Send frame to dashboard via WebSocket"""
        try:
            # Resize frame for faster transmission
            height, width = frame.shape[:2]
            max_width = 1280
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to JPEG for compression
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send via WebSocket
            await self.server.broadcast({
                'type': 'frame',
                'data': {
                    'image': f'data:image/jpeg;base64,{frame_base64}',
                    'timestamp': time.time(),
                    'width': frame.shape[1],
                    'height': frame.shape[0]
                }
            })
        except Exception as e:
            logger.debug(f"Failed to send frame: {e}")
    
    async def _send_metrics(self, current_fps: float):
        """Send performance metrics to dashboard"""
        memory = self.process.memory_info().rss / 1024 / 1024
        cpu = self.process.cpu_percent()
        
        # Calculate latency (simplified)
        if self.context_fusion.fusion_times:
            latency = float(np.mean(list(self.context_fusion.fusion_times))) * 1000
        else:
            latency = 0
        
        metrics = {
            'fps': float(current_fps),
            'latency': float(latency),
            'cpu': float(cpu),
            'memory': float(memory),
            'frames_processed': int(self.frames_processed),
            'frames_analyzed': int(self.frames_analyzed)
        }
        
        await self.server.broadcast({
            'type': 'metrics',
            'data': metrics
        })
        
        # Log metrics
        if self.frames_processed % 60 == 0:  # Every 60 frames
            logger.info(f"Pipeline metrics: FPS={current_fps:.1f}, Latency={latency:.1f}ms, "
                       f"CPU={cpu:.1f}%, Memory={memory:.1f}MB")
    
    async def start(self):
        """Start the streaming pipeline"""
        logger.info("Starting streaming pipeline...")
        
        await self.initialize()
        
        self.running = True
        self.start_time = time.time()
        self.frames_processed = 0
        self.frames_analyzed = 0
        
        # Start stream
        success = await self.stream.start()
        if not success:
            logger.error("Failed to start screen capture stream")
            return False
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self.process_frames()),
            asyncio.create_task(self.broadcast_contexts()),
            asyncio.create_task(self.stream.capture_frame_loop())  # Add frame capture loop
        ]
        
        logger.info("✅ Streaming pipeline started successfully!")
        logger.info("📊 Dashboard available at: http://localhost:8000/streaming")
        
        # Wait for tasks
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
        
        return True
    
    async def stop(self):
        """Stop the streaming pipeline"""
        logger.info("Stopping streaming pipeline...")
        
        self.running = False
        
        # Stop stream
        if self.stream:
            await self.stream.stop()
        
        # Stop server
        await self.server.stop()
        
        # Calculate final stats
        if self.frames_processed > 0:
            duration = time.time() - self.start_time
            avg_fps = self.frames_processed / duration
            analysis_rate = self.frames_analyzed / self.frames_processed * 100
            
            logger.info(f"Pipeline stopped. Stats:")
            logger.info(f"  • Duration: {duration:.1f}s")
            logger.info(f"  • Frames processed: {self.frames_processed}")
            logger.info(f"  • Frames analyzed: {self.frames_analyzed} ({analysis_rate:.1f}%)")
            logger.info(f"  • Average FPS: {avg_fps:.1f}")
    
    def get_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            'running': self.running,
            'frames_processed': self.frames_processed,
            'frames_analyzed': self.frames_analyzed,
            'uptime': time.time() - self.start_time if self.running else 0,
            'stream_metrics': self.stream.get_metrics() if self.stream else {},
            'fusion_metrics': self.context_fusion.get_metrics()
        }


def cleanup_port(port: int = 8001):
    """Kill any process using the specified port"""
    try:
        # Find process using the port
        result = subprocess.run(
            f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}'",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        logger.info(f"Killed process {pid} using port {port}")
                    except:
                        pass
            time.sleep(0.5)  # Wait for port to be released
    except Exception as e:
        logger.debug(f"Port cleanup: {e}")

async def main():
    """Main entry point for streaming pipeline"""
    # Clean up ports before starting
    cleanup_port(8001)
    cleanup_port(8000)
    
    print("\n" + "="*60)
    print("🚀 PORTER.AI STREAMING INTELLIGENCE")
    print("="*60)
    print("\nInitializing real-time streaming pipeline...")
    print("Components: ScreenCaptureKit → FastVLM → Context")
    print("-"*60 + "\n")
    
    # Create pipeline
    pipeline = StreamingPipeline({
        'width': 1920,
        'height': 1080,
        'fps': 10,  # Reduced from 60 to lower CPU usage
        'show_cursor': True
    })
    
    try:
        # Start pipeline
        await pipeline.start()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    finally:
        # Cleanup
        await pipeline.stop()
        print("\n✅ Pipeline shut down cleanly")


if __name__ == "__main__":
    # Run the streaming pipeline
    asyncio.run(main())