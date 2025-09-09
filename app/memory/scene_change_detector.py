#!/usr/bin/env python3
"""
Scene Change Detection for Smart Context Management.
Detects when the user switches between different applications or contexts.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import time
import logging
from skimage.metrics import structural_similarity as ssim
import cv2
import platform
import subprocess

logger = logging.getLogger(__name__)


class SceneChangeDetector:
    """
    Detects scene changes using multiple signals:
    - SSIM (Structural Similarity Index)
    - Application changes
    - Color histogram shifts
    - Tab/window changes
    """
    
    def __init__(
        self,
        ssim_threshold: float = 0.4,
        histogram_threshold: float = 0.7,
        history_size: int = 100
    ):
        """
        Initialize the scene change detector.
        
        Args:
            ssim_threshold: Threshold for SSIM (lower = more different)
            histogram_threshold: Threshold for histogram difference
            history_size: Number of changes to keep in history
        """
        self.ssim_threshold = ssim_threshold
        self.histogram_threshold = histogram_threshold
        
        # Previous state
        self.prev_frame = None
        self.prev_app = None
        self.prev_histogram = None
        self.prev_tab = None
        
        # Change history
        self.change_history = deque(maxlen=history_size)
        
        # Performance tracking
        self.last_detection_time = 0
        
        logger.info(f"SceneChangeDetector initialized with SSIM={ssim_threshold}, Hist={histogram_threshold}")
    
    def detect_change(
        self,
        frame: Optional[np.ndarray],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect if a scene change has occurred.
        
        Args:
            frame: Current frame (numpy array)
            metadata: Metadata including active_app, tab, timestamp, etc.
            
        Returns:
            Dictionary with:
                - scene_changed: Boolean indicating if scene changed
                - change_type: Type of change (visual, application, theme, tab)
                - confidence: Confidence score (0-1)
                - prev_app: Previous application (if app change)
                - details: Additional details about the change
        """
        start_time = time.time()
        
        result = {
            'scene_changed': False,
            'change_type': None,
            'confidence': 0.0,
            'timestamp': metadata.get('timestamp', time.time()),
            'details': {}
        }
        
        # Handle None frame
        if frame is None:
            self.change_history.append(result)
            return result
        
        # First frame - no change
        if self.prev_frame is None:
            self._update_state(frame, metadata)
            self.change_history.append(result)
            return result
        
        confidence_scores = []
        change_types = []
        
        # 1. Check SSIM (visual similarity)
        visual_change, ssim_score = self._check_visual_change(frame)
        if visual_change:
            change_types.append('visual')
            confidence_scores.append(1.0 - ssim_score)  # Convert to difference
            result['details']['ssim_score'] = ssim_score
        
        # 2. Check application change
        app_change, prev_app = self._check_app_change(metadata)
        if app_change:
            change_types.append('application')
            confidence_scores.append(0.9)  # High confidence for app change
            result['prev_app'] = prev_app
            result['details']['new_app'] = metadata.get('active_app')
        
        # 3. Check color histogram (theme change)
        theme_change, hist_diff = self._check_histogram_change(frame)
        if theme_change:
            change_types.append('theme')
            confidence_scores.append(hist_diff)
            result['details']['histogram_diff'] = hist_diff
        
        # 4. Check tab change (for browsers)
        tab_change = self._check_tab_change(metadata)
        if tab_change:
            result['tab_changed'] = True
            result['details']['prev_tab'] = self.prev_tab
            result['details']['new_tab'] = metadata.get('tab')
            # Tab changes might not be full scene changes
            if metadata.get('active_app', '').lower() in ['chrome', 'firefox', 'safari', 'edge']:
                change_types.append('tab')
                confidence_scores.append(0.7)
        
        # Determine if scene changed
        if change_types:
            result['scene_changed'] = True
            # Prioritize visual changes when both visual and app changes occur
            # (usually means the app change caused a major visual change)
            if 'visual' in change_types:
                result['change_type'] = 'visual'
            elif 'application' in change_types:
                result['change_type'] = 'application'
            elif 'theme' in change_types:
                result['change_type'] = 'theme'
            else:
                result['change_type'] = change_types[0]
            
            # Calculate aggregate confidence
            if confidence_scores:
                result['confidence'] = min(1.0, max(confidence_scores))
        
        # Update state
        self._update_state(frame, metadata)
        
        # Track detection time
        self.last_detection_time = time.time() - start_time
        result['detection_time_ms'] = self.last_detection_time * 1000
        
        # Add to history
        self.change_history.append(result)
        
        logger.debug(f"Scene change detection: {result['scene_changed']} "
                    f"(type={result['change_type']}, confidence={result['confidence']:.2f})")
        
        return result
    
    def _check_visual_change(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check for visual changes using SSIM.
        
        Returns:
            Tuple of (is_changed, ssim_score)
        """
        if self.prev_frame is None:
            return False, 1.0
        
        try:
            # Resize frames for faster SSIM calculation
            small_prev = cv2.resize(self.prev_frame, (320, 180))
            small_curr = cv2.resize(frame, (320, 180))
            
            # Convert to grayscale for SSIM
            gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_RGB2GRAY)
            gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_RGB2GRAY)
            
            # Calculate SSIM
            score = ssim(gray_prev, gray_curr)
            
            # Check if change is significant
            is_changed = score < self.ssim_threshold
            
            return is_changed, score
            
        except Exception as e:
            logger.error(f"Error calculating SSIM: {e}")
            return False, 1.0
    
    def _check_app_change(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check for application change.
        
        Returns:
            Tuple of (is_changed, previous_app)
        """
        current_app = metadata.get('active_app')
        
        if current_app and self.prev_app and current_app != self.prev_app:
            return True, self.prev_app
        
        return False, None
    
    def _check_histogram_change(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check for significant color histogram changes (theme changes).
        
        Returns:
            Tuple of (is_changed, histogram_difference)
        """
        current_hist = self._calculate_histogram(frame)
        
        if self.prev_histogram is None:
            return False, 0.0
        
        try:
            # Compare histograms using correlation
            diff = 0.0
            for i in range(3):  # RGB channels
                correlation = cv2.compareHist(
                    current_hist[i],
                    self.prev_histogram[i],
                    cv2.HISTCMP_CORREL
                )
                diff += (1.0 - correlation) / 3.0
            
            # Check if change is significant
            is_changed = diff > self.histogram_threshold
            
            return is_changed, diff
            
        except Exception as e:
            logger.error(f"Error comparing histograms: {e}")
            return False, 0.0
    
    def _check_tab_change(self, metadata: Dict[str, Any]) -> bool:
        """Check for tab changes in browsers."""
        current_tab = metadata.get('tab')
        
        if current_tab and self.prev_tab and current_tab != self.prev_tab:
            return True
        
        return False
    
    def _calculate_histogram(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Calculate color histogram for the frame.
        
        Returns:
            List of histograms for each channel [R, G, B]
        """
        histograms = []
        
        for i in range(3):  # RGB channels
            hist = cv2.calcHist(
                [frame],
                [i],
                None,
                [64],  # Bins
                [0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        return histograms
    
    def _update_state(self, frame: np.ndarray, metadata: Dict[str, Any]):
        """Update internal state with current frame and metadata."""
        self.prev_frame = frame.copy() if frame is not None else None
        self.prev_app = metadata.get('active_app')
        self.prev_tab = metadata.get('tab')
        
        if frame is not None:
            self.prev_histogram = self._calculate_histogram(frame)
    
    def reset(self):
        """Reset detector state."""
        self.prev_frame = None
        self.prev_app = None
        self.prev_histogram = None
        self.prev_tab = None
        self.change_history.clear()
        logger.info("SceneChangeDetector reset")
    
    def get_recent_changes(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent scene changes.
        
        Args:
            n: Number of recent changes to return
            
        Returns:
            List of recent change events
        """
        recent = list(self.change_history)[-n:]
        return [c for c in recent if c['scene_changed']]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_detections = len(self.change_history)
        scene_changes = sum(1 for c in self.change_history if c['scene_changed'])
        
        change_types = {}
        for change in self.change_history:
            if change['scene_changed'] and change['change_type']:
                change_types[change['change_type']] = change_types.get(change['change_type'], 0) + 1
        
        return {
            'total_detections': total_detections,
            'scene_changes': scene_changes,
            'change_rate': scene_changes / total_detections if total_detections > 0 else 0,
            'change_types': change_types,
            'last_detection_time_ms': self.last_detection_time * 1000
        }


class ApplicationDetector:
    """
    Detects the currently active application across platforms.
    """
    
    @staticmethod
    def get_active_application() -> Optional[str]:
        """
        Get the currently active application name.
        
        Returns:
            Application name or None if detection fails
        """
        system = platform.system()
        
        try:
            if system == "Darwin":  # macOS
                script = '''
                tell application "System Events"
                    get name of first application process whose frontmost is true
                end tell
                '''
                result = subprocess.check_output(['osascript', '-e', script], text=True)
                return result.strip()
                
            elif system == "Windows":
                # Windows implementation (requires pywin32)
                try:
                    import win32gui
                    import win32process
                    
                    window = win32gui.GetForegroundWindow()
                    _, pid = win32process.GetWindowThreadProcessId(window)
                    
                    import psutil
                    process = psutil.Process(pid)
                    return process.name()
                    
                except ImportError:
                    logger.warning("pywin32 not installed, cannot detect active app on Windows")
                    return None
                    
            elif system == "Linux":
                # Linux implementation using xdotool
                try:
                    # Get active window ID
                    window_id = subprocess.check_output(['xdotool', 'getactivewindow'], text=True).strip()
                    
                    # Get window name
                    window_name = subprocess.check_output(
                        ['xdotool', 'getwindowname', window_id], text=True
                    ).strip()
                    
                    # Try to extract app name from window title
                    # This is heuristic and may need adjustment
                    if ' - ' in window_name:
                        return window_name.split(' - ')[-1]
                    return window_name
                    
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("xdotool not installed or no display, cannot detect active app on Linux")
                    return None
                    
        except Exception as e:
            logger.error(f"Error detecting active application: {e}")
            return None
    
    @staticmethod
    def get_browser_tab() -> Optional[str]:
        """
        Get the current browser tab title (if browser is active).
        
        Returns:
            Tab title or None
        """
        # This would require browser-specific extensions or accessibility APIs
        # For now, return None as a placeholder
        return None


# Utility functions
def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate simple frame difference metric.
    
    Args:
        frame1: First frame
        frame2: Second frame
        
    Returns:
        Normalized difference score (0-1)
    """
    if frame1.shape != frame2.shape:
        return 1.0
    
    diff = np.abs(frame1.astype(float) - frame2.astype(float))
    return np.mean(diff) / 255.0


def is_significant_change(
    ssim_score: float,
    hist_diff: float,
    app_changed: bool,
    thresholds: Optional[Dict[str, float]] = None
) -> bool:
    """
    Determine if a change is significant enough to trigger context reset.
    
    Args:
        ssim_score: SSIM similarity score
        hist_diff: Histogram difference
        app_changed: Whether application changed
        thresholds: Custom thresholds
        
    Returns:
        True if change is significant
    """
    if thresholds is None:
        thresholds = {
            'ssim': 0.4,
            'histogram': 0.7
        }
    
    # App change is always significant
    if app_changed:
        return True
    
    # Check visual similarity
    if ssim_score < thresholds['ssim']:
        return True
    
    # Check color distribution
    if hist_diff > thresholds['histogram']:
        return True
    
    return False