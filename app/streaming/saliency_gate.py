#!/usr/bin/env python3
"""
Saliency gate for intelligent frame filtering.
Determines which frames contain important changes worth processing.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import time
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class FrameMetrics:
    """Metrics for a single frame"""
    motion_score: float
    edge_density: float
    color_variance: float
    ui_activity: float
    saliency_score: float
    should_process: bool
    regions_of_interest: List[Tuple[int, int, int, int]]  # x, y, w, h

class SaliencyGate:
    """
    Smart gate that decides which frames to process based on saliency.
    Uses optical flow, edge detection, and UI activity detection.
    """
    
    def __init__(self, 
                 motion_threshold: float = 0.1,
                 edge_threshold: float = 0.15,
                 history_size: int = 10):
        """
        Initialize saliency gate.
        
        Args:
            motion_threshold: Minimum motion to trigger processing
            edge_threshold: Minimum edge density for UI changes
            history_size: Number of frames to keep in history
        """
        self.motion_threshold = motion_threshold
        self.edge_threshold = edge_threshold
        self.history_size = history_size
        
        # Frame history
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_history = deque(maxlen=history_size)
        
        # Optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Performance tracking
        self.frames_processed = 0
        self.frames_passed = 0
        self.processing_times = deque(maxlen=100)
        
        # Region tracking
        self.hot_regions = []  # Areas with frequent activity
        self.last_activity_map: Optional[np.ndarray] = None
        
    def should_process(self, frame: np.ndarray, force: bool = False) -> FrameMetrics:
        """
        Determine if frame should be processed.
        
        Args:
            frame: RGB frame
            force: Force processing regardless of saliency
            
        Returns:
            FrameMetrics with decision and scores
        """
        start_time = time.time()
        self.frames_processed += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Initialize metrics
        motion_score = 0.0
        edge_density = 0.0
        color_variance = 0.0
        ui_activity = 0.0
        regions_of_interest = []
        
        # Calculate motion if we have previous frame
        if self.prev_gray is not None:
            motion_score, motion_regions = self._calculate_motion(gray, self.prev_gray)
            regions_of_interest.extend(motion_regions)
        
        # Calculate edge density (UI changes)
        edge_density, edge_regions = self._calculate_edge_density(gray)
        regions_of_interest.extend(edge_regions)
        
        # Calculate color variance (content changes)
        color_variance = self._calculate_color_variance(frame)
        
        # Detect UI activity patterns
        ui_activity, ui_regions = self._detect_ui_activity(gray)
        regions_of_interest.extend(ui_regions)
        
        # Combine scores
        saliency_score = self._calculate_saliency(
            motion_score, edge_density, color_variance, ui_activity
        )
        
        # Make decision
        should_process = (
            force or
            motion_score > self.motion_threshold or
            edge_density > self.edge_threshold or
            saliency_score > 0.5
        )
        
        # Update state
        self.prev_frame = frame
        self.prev_gray = gray
        
        if should_process:
            self.frames_passed += 1
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Create metrics
        metrics = FrameMetrics(
            motion_score=motion_score,
            edge_density=edge_density,
            color_variance=color_variance,
            ui_activity=ui_activity,
            saliency_score=saliency_score,
            should_process=should_process,
            regions_of_interest=self._merge_regions(regions_of_interest)
        )
        
        # Add to history
        self.frame_history.append(metrics)
        
        logger.debug(f"Saliency gate: score={saliency_score:.2f}, process={should_process}, time={processing_time*1000:.1f}ms")
        
        return metrics
    
    def _calculate_motion(self, gray: np.ndarray, prev_gray: np.ndarray) -> Tuple[float, List]:
        """Calculate optical flow motion score"""
        regions = []
        
        try:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, **self.flow_params
            )
            
            # Calculate magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Find regions with significant motion
            motion_mask = magnitude > 2.0  # pixels with motion > 2 pixels
            
            if np.any(motion_mask):
                # Find contours of motion regions
                contours, _ = cv2.findContours(
                    motion_mask.astype(np.uint8) * 255,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        regions.append((x, y, w, h))
            
            # Calculate overall motion score
            motion_score = np.mean(magnitude) / 10.0  # Normalize
            motion_score = min(1.0, motion_score)
            
            return motion_score, regions
            
        except Exception as e:
            logger.debug(f"Motion calculation error: {e}")
            return 0.0, []
    
    def _calculate_edge_density(self, gray: np.ndarray) -> Tuple[float, List]:
        """Calculate edge density for UI change detection"""
        regions = []
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find regions with high edge density
        kernel_size = 50
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        edge_density_map = cv2.filter2D(edges.astype(float), -1, kernel)
        
        # Threshold to find UI regions
        threshold = 50
        ui_mask = edge_density_map > threshold
        
        if np.any(ui_mask):
            # Find contours
            contours, _ = cv2.findContours(
                ui_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area for UI element
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append((x, y, w, h))
        
        # Calculate overall edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        return edge_density, regions
    
    def _calculate_color_variance(self, frame: np.ndarray) -> float:
        """Calculate color variance to detect content changes"""
        # Downsample for efficiency
        small = cv2.resize(frame, (64, 64))
        
        # Calculate variance across color channels
        variance = np.var(small) / 255.0
        
        return min(1.0, variance / 100.0)  # Normalize
    
    def _detect_ui_activity(self, gray: np.ndarray) -> Tuple[float, List]:
        """Detect UI activity patterns (menus, dialogs, etc)"""
        regions = []
        activity_score = 0.0
        
        # Detect rectangular structures (windows, dialogs)
        edges = cv2.Canny(gray, 30, 100)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50,
            minLineLength=50, maxLineGap=10
        )
        
        if lines is not None:
            # Group lines into potential UI elements
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
                
                if angle < np.pi/6:  # Mostly horizontal
                    horizontal_lines.append((x1, y1, x2, y2))
                elif angle > np.pi/3:  # Mostly vertical
                    vertical_lines.append((x1, y1, x2, y2))
            
            # UI activity based on structured lines
            activity_score = min(1.0, (len(horizontal_lines) + len(vertical_lines)) / 100.0)
            
            # Find intersections as potential UI corners
            for h_line in horizontal_lines[:10]:  # Limit for performance
                for v_line in vertical_lines[:10]:
                    # Simple intersection check
                    hx1, hy1, hx2, hy2 = h_line
                    vx1, vy1, vx2, vy2 = v_line
                    
                    if (min(hx1, hx2) <= max(vx1, vx2) and 
                        max(hx1, hx2) >= min(vx1, vx2) and
                        min(vy1, vy2) <= max(hy1, hy2) and 
                        max(vy1, vy2) >= min(hy1, hy2)):
                        
                        # Potential UI element corner
                        x = (min(hx1, hx2) + min(vx1, vx2)) // 2
                        y = (min(hy1, hy2) + min(vy1, vy2)) // 2
                        regions.append((x - 25, y - 25, 50, 50))
        
        return activity_score, regions
    
    def _calculate_saliency(self, motion: float, edges: float, 
                           color: float, ui: float) -> float:
        """Combine metrics into overall saliency score"""
        # Weighted combination
        weights = {
            'motion': 0.4,
            'edges': 0.3,
            'color': 0.1,
            'ui': 0.2
        }
        
        saliency = (
            weights['motion'] * motion +
            weights['edges'] * edges +
            weights['color'] * color +
            weights['ui'] * ui
        )
        
        # Apply temporal smoothing
        if self.frame_history:
            recent_scores = [m.saliency_score for m in self.frame_history]
            temporal_avg = np.mean(recent_scores)
            saliency = 0.7 * saliency + 0.3 * temporal_avg
        
        return min(1.0, saliency)
    
    def _merge_regions(self, regions: List[Tuple[int, int, int, int]]) -> List:
        """Merge overlapping regions of interest"""
        if not regions:
            return []
        
        # Sort by area (largest first)
        regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)
        
        merged = []
        for region in regions:
            x, y, w, h = region
            
            # Check if overlaps with existing merged region
            overlaps = False
            for i, (mx, my, mw, mh) in enumerate(merged):
                # Calculate overlap
                overlap_x = max(0, min(x + w, mx + mw) - max(x, mx))
                overlap_y = max(0, min(y + h, my + mh) - max(y, my))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > 0:
                    # Merge regions
                    new_x = min(x, mx)
                    new_y = min(y, my)
                    new_w = max(x + w, mx + mw) - new_x
                    new_h = max(y + h, my + mh) - new_y
                    merged[i] = (new_x, new_y, new_w, new_h)
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(region)
        
        return merged[:5]  # Limit to top 5 regions
    
    def get_activity_heatmap(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate activity heatmap based on recent frames"""
        height, width = shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate activity from recent frames
        for metrics in self.frame_history:
            for x, y, w, h in metrics.regions_of_interest:
                # Add gaussian blob for each region
                y1, y2 = max(0, y), min(height, y + h)
                x1, x2 = max(0, x), min(width, x + w)
                heatmap[y1:y2, x1:x2] += 1.0
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Apply gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        return heatmap
    
    def get_metrics(self) -> Dict:
        """Get gate performance metrics"""
        metrics = {
            'frames_processed': self.frames_processed,
            'frames_passed': self.frames_passed,
            'pass_rate': self.frames_passed / self.frames_processed if self.frames_processed > 0 else 0,
            'avg_processing_time_ms': float(np.mean(self.processing_times)) * 1000 if self.processing_times else 0,
            'motion_threshold': self.motion_threshold,
            'edge_threshold': self.edge_threshold
        }
        
        if self.frame_history:
            recent = list(self.frame_history)[-10:]
            metrics['recent_avg_saliency'] = float(np.mean([m.saliency_score for m in recent]))
            metrics['recent_avg_motion'] = float(np.mean([m.motion_score for m in recent]))
        
        return metrics


# Demo usage
if __name__ == "__main__":
    import mss
    
    def demo():
        """Demonstrate saliency gate"""
        print("🎯 Saliency Gate Demo")
        print("=" * 40)
        
        # Create gate
        gate = SaliencyGate(motion_threshold=0.1, edge_threshold=0.15)
        
        # Capture frames
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            
            print("\nCapturing 10 frames...")
            print("-" * 40)
            
            for i in range(10):
                # Capture frame
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)[:, :, :3]  # Remove alpha
                
                # Check saliency
                metrics = gate.should_process(frame)
                
                print(f"\nFrame {i+1}:")
                print(f"  • Motion: {metrics.motion_score:.2%}")
                print(f"  • Edges: {metrics.edge_density:.2%}")
                print(f"  • Color: {metrics.color_variance:.2%}")
                print(f"  • UI Activity: {metrics.ui_activity:.2%}")
                print(f"  • Saliency: {metrics.saliency_score:.2%}")
                print(f"  • Process: {'✅ YES' if metrics.should_process else '❌ NO'}")
                print(f"  • ROIs: {len(metrics.regions_of_interest)}")
                
                # Small delay to allow for motion
                time.sleep(0.5)
            
            # Show metrics
            print("\n" + "=" * 40)
            print("📊 Gate Metrics:")
            for key, value in gate.get_metrics().items():
                if isinstance(value, float):
                    print(f"  • {key}: {value:.2f}")
                else:
                    print(f"  • {key}: {value}")
    
    demo()